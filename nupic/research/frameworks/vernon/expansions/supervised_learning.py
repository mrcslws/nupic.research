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

import sys

import torch

from nupic.research.frameworks.pytorch.model_utils import evaluate_model, train_model

__all__ = [
    "SupervisedLearning",
]


class SupervisedLearning:
    """
    Implements training and validation loops.
    """
    def setup_experiment(self, config):
        """
        :param config: Dictionary containing the configuration parameters

            - loss_function: Loss function. See "torch.nn.functional"
            - batches_in_epoch: Number of batches per epoch.
                                Useful for debugging
            - batches_in_epoch_val: Number of batches per epoch in validation.
                                   Useful for debugging
        """
        super().setup_experiment(config)

        assert hasattr(self, "model"), (
            "Must use HasModel or similar expansion"
        )
        assert hasattr(self, "optimizer"), (
            "Must use HasOptimizer or similar expansion"
        )

        self._loss_function = config.get(
            "loss_function", torch.nn.functional.cross_entropy
        )

        self.batches_in_epoch = config.get("batches_in_epoch", sys.maxsize)
        self.batches_in_epoch_val = config.get("batches_in_epoch_val",
                                               sys.maxsize)

        if "train_model_func" in config:
            self.logger.warning(
                "'train_model_func' is deprecated and will be removed soon."
                "Instead, override the train_model method."
            )
            self._train_model = config["train_model_func"]
        else:
            self._train_model = train_model

        if "evaluate_model_func" in config:
            self.logger.warning(
                "'evaluate_model_func' is deprecated and will be removed soon."
                "Instead, override the validate method."
            )
            self._evaluate_model = config["evaluate_model_func"]
        else:
            self._evaluate_model = evaluate_model

    def error_loss(self, output, target, reduction="mean"):
        """
        The error loss component of the loss function.
        """
        return self._loss_function(output, target, reduction=reduction)

    def complexity_loss(self, model):
        """
        The model complexity component of the loss function.
        """

    def train_model(self, loader):
        """
        Train the model by making one pass through the provided loader.
        """
        self._train_model(
            model=self.model,
            loader=loader,
            optimizer=self.optimizer,
            device=self.device,
            criterion=self.error_loss,
            complexity_loss_fn=self.complexity_loss,
            batches_in_epoch=self.batches_in_epoch,
            pre_batch_callback=self.pre_batch,
            post_batch_callback=self.post_batch_wrapper,
            transform_to_device_fn=self.transform_data_to_device,
        )

    def train_epoch(self, loader):
        self.pre_epoch()
        self.train_model(loader)
        self.post_epoch()

    def validate(self, loader):
        """
        Evaluate the model using the provided dataloader.
        """
        return self._evaluate_model(
            model=self.model,
            loader=loader,
            device=self.device,
            criterion=self.error_loss,
            complexity_loss_fn=self.complexity_loss,
            batches_in_epoch=self.batches_in_epoch_val,
            transform_to_device_fn=self.transform_data_to_device,
        )

    def post_batch_wrapper(self, **kwargs):
        self.post_optimizer_step()
        self.post_batch(**kwargs)

    def transform_data_to_device(self, data, target, device, non_blocking):
        """
        This provides an extensibility point for performing any final
        transformations on the data or targets.
        """
        data = data.to(self.device, non_blocking=non_blocking)
        target = target.to(self.device, non_blocking=non_blocking)
        return data, target

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

    @classmethod
    def get_printable_result(cls, result):
        """
        Return a stripped down version of result that has its large data structures
        removed so that the result can be printed to the console.
        """
        keys = ["total_correct", "total_tested", "mean_loss", "mean_accuracy",
                "learning_rate"]
        return {key: result[key]
                for key in keys
                if key in result}

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()

        name = "SupervisedLearning"

        # Extended methods
        eo["setup_experiment"].append(name + ".setup_experiment")

        eo.update(
            # Overwritten methods
            get_printable_result=[name + ": Basic keys"],

            # New methods
            validate=[name + ".validate"],
            train_model=[name + ".train_model"],
            pre_batch=[],
            post_batch=[],
            transform_data_to_device=[name + ".transform_data_to_device"],
            error_loss=[name + ".error_loss"],
            complexity_loss=[],
        )

        return eo
