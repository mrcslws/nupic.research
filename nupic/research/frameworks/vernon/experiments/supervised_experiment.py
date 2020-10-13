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

import copy
import time

import torch
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from nupic.research.frameworks.pytorch.distributed_sampler import (
    UnpaddedDistributedSampler,
)
from nupic.research.frameworks.pytorch.model_utils import aggregate_eval_results
from nupic.research.frameworks.vernon import expansions
from nupic.research.frameworks.vernon.experiments.base_experiment import BaseExperiment

__all__ = [
    "SupervisedExperiment",
]


class SupervisedExperiment(expansions.SupervisedLearning,
                           expansions.HasEpochs,
                           expansions.HasOptimizer,
                           expansions.HasModel,
                           expansions.Distributed,
                           BaseExperiment):
    """
    General experiment class used to train neural networks in supervised learning tasks.
    """
    def setup_experiment(self, config):
        """
        Configure the experiment for training

        :param config: Dictionary containing the configuration parameters

            - data: Dataset path
            - train_dir: Dataset training data relative path
            - batch_size: Training batch size
            - val_dir: Dataset validation data relative path
            - val_batch_size: Validation batch size
            - workers: how many data loading processes to use
            - train_loader_drop_last: Whether to skip last batch if it is
                                      smaller than the batch size
            - num_classes: Limit the dataset size to the given number of classes
            - epochs: Number of epochs to train
            - log_timestep_freq: Configures mixins and subclasses that log every
                                 timestep to only log every nth timestep (in
                                 addition to the final timestep of each epoch).
                                 Set to 0 to log only at the end of each epoch.
            - sample_transform: Transform acting on the training samples. To be used
                                additively after default transform or auto-augment.
            - target_transform: Transform acting on the training targets.
            - replicas_per_sample: Number of replicas to create per sample in the batch.
                                   (each replica is transformed independently)
                                   Used in maxup.
            - epochs_to_validate: list of epochs to run validate(). A -1 asks
                                  to run validate before any training occurs.
                                  Default: every epoch.
            - validate_immediately: Whether to validate before the first epoch.
            - extra_validations_per_epoch: number of additional validations to
                                           perform mid-epoch. Additional
                                           validations are distributed evenly
                                           across training batches.
        """
        super().setup_experiment(config)

        # Apply DistributedDataParallel after all other model mutations
        if self.distributed:
            self.model = DistributedDataParallel(self.model)
        else:
            self.model = DataParallel(self.model)

        self.num_classes = config.get("num_classes", 1000)
        self.epochs = config.get("epochs", 1)

        # Configure data loaders
        self.train_loader = self.create_train_dataloader(config)
        self.val_loader = self.create_validation_dataloader(config)
        self.total_batches = len(self.train_loader)

        self.epochs_to_validate = config.get("epochs_to_validate",
                                             range(self.epochs))
        self.validate_immediately = config.get("validate_immediately",
                                               (-1 in self.epochs_to_validate))

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

        if config.get("distributed", False):
            sampler = DistributedSampler(dataset)
        else:
            sampler = None

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

        if config.get("distributed", False):
            sampler = UnpaddedDistributedSampler(dataset, shuffle=False)
        else:
            sampler = None

        return DataLoader(
            dataset=dataset,
            batch_size=config.get("val_batch_size",
                                  config.get("batch_size", 1)),
            shuffle=False,
            num_workers=config.get("workers", 0),
            sampler=sampler,
            pin_memory=torch.cuda.is_available(),
        )

    def run_iteration(self):
        return self.run_epoch()

    def run_epoch(self):
        self.train_epoch(self.train_loader)

        t1 = time.time()

        if self.current_epoch in self.epochs_to_validate:
            ret = self.validate(self.val_loader)
        else:
            ret = {
                "total_correct": 0,
                "total_tested": 0,
                "mean_loss": 0.0,
                "mean_accuracy": 0.0,
            }

        ret.update(
            learning_rate=self.get_lr()[0],
        )

        if self.rank == 0:
            self.logger.debug("validate time: %s", time.time() - t1)
            self.logger.debug("---------- End of run epoch ------------")
            self.logger.debug("")

        self.current_epoch += 1
        return ret

    def pre_experiment(self):
        """Run validation before training."""
        if self.validate_immediately:
            self.logger.debug("Validating before any training:")
            return self.validate()

    def post_batch(self, error_loss, batch_idx, num_images, time_string,
                   **kwargs):
        super().post_batch(error_loss=error_loss, batch_idx=batch_idx,
                           num_images=num_images, time_string=time_string,
                           **kwargs)

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

    def pre_epoch(self):
        super().pre_epoch()
        if self.distributed:
            self.train_loader.sampler.set_epoch(self.current_epoch)

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        return super().validate(loader)

    @classmethod
    def _aggregate_validation_results(cls, results):
        result = copy.copy(results[0])
        result.update(aggregate_eval_results(results))
        return result

    @classmethod
    def aggregate_results(cls, results):
        """
        Aggregate multiple processes' "run_epoch" results into a single result.

        :param results:
            A list of return values from run_epoch from different processes.
        :type results: list

        :return:
            A single result dict with results aggregated.
        :rtype: dict
        """
        return cls._aggregate_validation_results(results)

    @classmethod
    def aggregate_pre_experiment_results(cls, results):
        if results[0] is not None:
            return cls._aggregate_validation_results(results)

        return None

    def should_stop(self):
        return self.current_epoch >= self.epochs

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()

        exp = "SupervisedExperiment"

        # Extended methods
        eo["setup_experiment"].append(exp + ".setup_experiment")
        eo["pre_epoch"].append(exp + ": Update distributed sampler")
        eo["post_batch"].append(exp + ": Logging")
        eo["validate"].insert(0, exp + ": Specify val loader by default")

        eo.update(
            # Overwritten methods
            aggregate_results=[exp + ": Aggregate validation results"],
            aggregate_pre_experiment_results=[exp + ": Aggregate validation results"],
            run_iteration=[exp + ".run_iteration"],

            # New methods
            run_epoch=[exp + ".run_epoch"],
            stop_experiment=[exp + ".stop_experiment"],
            load_dataset=[exp + ".load_dataset"],
        )

        return eo
