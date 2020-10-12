# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2020, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import math
import sys
import time
from collections import defaultdict

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torchvision import transforms

from nupic.research.frameworks.pytorch.dataset_utils.samplers import (
    TaskDistributedSampler,
    TaskRandomSampler,
)
from nupic.research.frameworks.pytorch.model_utils import evaluate_model, train_model
from nupic.research.frameworks.vernon import expansions
from nupic.research.frameworks.vernon.experiments.base_experiment import BaseExperiment

__all__ = [
    "ContinualLearningExperiment",
]


class ContinualLearningExperiment(expansions.ContinualLearningMetrics,
                                  expansions.Distributed,
                                  expansions.HasEpochs,
                                  expansions.HasOptimizer,
                                  expansions.HasModel,
                                  BaseExperiment):
    def setup_experiment(self, config):
        super().setup_experiment(config)

        # Apply DistributedDataParallel after all other model mutations
        if self.distributed:
            self.model = DistributedDataParallel(self.model)
        else:
            self.model = DataParallel(self.model)

        self.train_loader = self.create_train_dataloader(config)
        self.val_loader = self.create_validation_dataloader(config)
        self.total_batches = len(self.train_loader)

        self._loss_function = config.get(
            "loss_function", torch.nn.functional.cross_entropy
        )
        self.epochs = config.get("epochs", 1)
        self.batches_in_epoch = config.get("batches_in_epoch", sys.maxsize)
        self.batches_in_epoch_val = config.get("batches_in_epoch_val", sys.maxsize)

        self.current_task = 0

        # Defines how many classes should exist per task
        self.num_tasks = config.get("num_tasks", 1)

        self.num_classes = config.get("num_classes", None)
        assert self.num_classes is not None, "num_classes should be defined"

        self.num_classes_per_task = math.floor(self.num_classes / self.num_tasks)

        # Applying target transform depending on type of CL task
        # Task - we know the task, so the network is multihead
        # Class - we don't know the task, network has as many heads as classes
        self.cl_experiment_type = config.get("cl_experiment_type", "class")
        if self.cl_experiment_type == "task":
            self.logger.info("Overriding target transform")
            self.dataset_args["target_transform"] = (
                transforms.Lambda(lambda y: y % self.num_classes_per_task)
            )

        # Whitelist evaluation metrics
        self.evaluation_metrics = config.get(
            "evaluation_metrics", ["eval_all_visited_tasks"]
        )
        for metric in self.evaluation_metrics:
            if not hasattr(self, metric):
                raise ValueError(f"Metric {metric} not available.")

    def should_stop(self):
        """
        Whether or not the experiment should stop. Usually determined by the
        number of epochs but customizable to any other stopping criteria
        """
        return self.current_task >= self.num_tasks

    def run_iteration(self):
        return self.run_task()

    def run_task(self):
        """Run outer loop over tasks"""
        # configure the sampler to load only samples from current task
        self.logger.info("Training...")
        self.train_loader.sampler.set_active_tasks(self.current_task)

        # Run epochs, inner loop
        # TODO: get and return the results from run_epoch
        for _ in range(self.epochs):
            self.run_epoch()

        ret = self.evaluate_all_metrics()
        ret.update(
            learning_rate=self.get_lr()[0],
        )

        self.current_task += 1
        return ret

    def evaluate_all_metrics(self):
        ret = {}
        for metric in self.evaluation_metrics:
            eval_function = getattr(self, metric)
            temp_ret = eval_function()
            self.logger.debug(temp_ret)
            for k, v in temp_ret.items():
                ret[f"{metric}__{k}"] = v
        return ret

    def run_epoch(self):
        self.pre_epoch()
        self.train_epoch()
        self.post_epoch()

        self.current_epoch += 1

    def pre_epoch(self):
        super().pre_epoch()
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.current_epoch)

    def train_epoch(self):
        train_model(
            model=self.model,
            loader=self.train_loader,
            optimizer=self.optimizer,
            device=self.device,
            criterion=self.error_loss,
            complexity_loss_fn=self.complexity_loss,
            batches_in_epoch=self.batches_in_epoch,
            pre_batch_callback=self.pre_batch,
            post_batch_callback=self.post_batch_wrapper,
            transform_to_device_fn=self.transform_data_to_device,
        )

    def pre_batch(self, model, batch_idx):
        """
        Called before passing a batch into the model.
        """

    def post_batch(self, model, error_loss, complexity_loss, batch_idx,
                   num_images, time_string):
        """
        Called after the batch has been processed and learned from. Guaranteed
        to run after post_optimizer_step() has finished.
        """
        if self.progress and self.current_epoch == 0 and batch_idx == 0:
            self.logger.info("Launch time to end of first batch: %s",
                             time.time() - self.launch_time)

        if self.progress and (batch_idx % 40) == 0:
            total_batches = self.total_batches
            current_batch = batch_idx
            if self.distributed:
                # Compute actual batch size from distributed sampler
                total_batches *= self.train_loader.sampler.num_replicas
                current_batch *= self.train_loader.sampler.num_replicas
            self.logger.debug("End of batch for rank: %s. Epoch: %s, Batch: %s/%s, "
                              "loss: %s, Learning rate: %s num_images: %s",
                              self.rank, self.current_epoch, current_batch,
                              total_batches, error_loss, self.get_lr(),
                              num_images)
            self.logger.debug("Timing: %s", time_string)

    def post_batch_wrapper(self, **kwargs):
        self.post_optimizer_step()
        self.post_batch(**kwargs)

    def error_loss(self, output, target, reduction="mean"):
        """
        The error loss component of the loss function.
        """
        return self._loss_function(output, target, reduction=reduction)

    def complexity_loss(self, model):
        """
        The model complexity component of the loss function.
        """

    def transform_data_to_device(self, data, target, device, non_blocking):
        """
        This provides an extensibility point for performing any final
        transformations on the data or targets.
        """
        data = data.to(self.device, non_blocking=non_blocking)
        target = target.to(self.device, non_blocking=non_blocking)
        return data, target

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        return evaluate_model(
            model=self.model,
            loader=loader,
            device=self.device,
            criterion=self.error_loss,
            complexity_loss_fn=self.complexity_loss,
            batches_in_epoch=self.batches_in_epoch_val,
            transform_to_device_fn=self.transform_data_to_device,
        )

    def pre_experiment(self):
        if self.validate_immediately:
            return self.evaluate_all_metrics()

        return None

    def set_state(self, state):
        super().set_state(state)
        self.current_task = state["current_task"]

    def get_state(self):
        state = super().get_state()
        state["current_task"] = self.current_task

    @classmethod
    def load_dataset(cls, config, train=True):
        """
        :param config: Dictionary containing the configuration parameters

            - dataset_class: A callable that returns a pytorch Dataset
            - dataset_args: Args for dataset_class
        """
        dataset_class = config.get("dataset_class", None)
        if dataset_class is None:
            raise ValueError("Must specify 'dataset_class' in config.")

        dataset_args = config.get("dataset_args", {})
        dataset_args.update(train=train)
        return dataset_class(**dataset_args)

    @classmethod
    def create_train_dataloader(cls, config, dataset=None):
        """
        This method is a classmethod so that it can be used directly by analysis
        tools, while also being easily overrideable.
        """
        if dataset is None:
            dataset = cls.load_dataset(config, train=True)

        sampler = cls.create_task_sampler(config, dataset, train=True)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("batch_size", 1),
            shuffle=sampler is None,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            drop_last=config.get("train_loader_drop_last", True),
        )

    @classmethod
    def create_validation_dataloader(cls, config, dataset=None):
        """
        This method is a classmethod so that it can be used directly by analysis
        tools, while also being easily overrideable.
        """
        if dataset is None:
            dataset = cls.load_dataset(config, train=False)

        sampler = cls.create_task_sampler(config, dataset, train=False)
        return DataLoader(
            dataset=dataset,
            batch_size=config.get("val_batch_size",
                                  config.get("batch_size", 1)),
            shuffle=False,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
        )

    @classmethod
    def create_task_sampler(cls, config, dataset, train):
        # Assume dataloaders are already created
        class_indices = defaultdict(list)
        for idx, (_, target) in enumerate(dataset):
            class_indices[target].append(idx)

        # Defines how many classes should exist per task
        num_tasks = config.get("num_tasks", 1)
        num_classes = config.get("num_classes", None)
        assert num_classes is not None, "num_classes should be defined"
        num_classes_per_task = math.floor(num_classes / num_tasks)

        task_indices = defaultdict(list)
        for i in range(num_tasks):
            for j in range(num_classes_per_task):
                task_indices[i].extend(class_indices[j + (i * num_classes_per_task)])

        # Change the sampler in the train loader
        distributed = config.get("distributed", False)
        if distributed and train:
            sampler = TaskDistributedSampler(
                dataset,
                task_indices
            )
        else:
            # TODO: implement a TaskDistributedUnpaddedSampler
            # mplement the aggregate results
            # after above are implemented, remove this if else
            sampler = TaskRandomSampler(task_indices)

        return sampler

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "ContinualLearningExperiment"

        # Extended methods
        eo["setup_experiment"].append(exp + ".setup_experiment")
        eo["pre_epoch"].append(exp + ": Update distributed sampler")

        eo.update(
            # Overwritten methods
            should_stop=[exp + ".should_stop"],
            run_iteration=[exp + ".run_iteration"],

            # New methods
            run_task=[exp + ".run_task"],
            validate=[exp + ".validate"],
            train_epoch=[exp + ".train_epoch"],
            run_epoch=[exp + ".run_epoch"],
            pre_batch=[],
            post_batch=[exp + ": Logging"],
            transform_data_to_device=[exp + ".transform_data_to_device"],
            error_loss=[exp + ".error_loss"],
            complexity_loss=[],
        )
        return eo
