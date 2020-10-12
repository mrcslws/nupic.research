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

from nupic.research.frameworks.pytorch import datasets
from nupic.research.frameworks.pytorch.model_utils import evaluate_model, train_model
from nupic.research.frameworks.vernon import expansions, mixins
from nupic.research.frameworks.vernon.experiments.supervised_experiment import (
    SupervisedExperiment,
)

__all__ = [
    "ImagenetExperiment",
]


class ImagenetExperiment(mixins.SelectiveWeightDecay,
                         mixins.MixedPrecision,
                         mixins.ExtraValidations,
                         mixins.FixedLRSchedule,
                         expansions.StepBasedLogging,
                         SupervisedExperiment):
    """
    Experiment class used to train Sparse and dense versions of Resnet50 v1.5
    models on Imagenet dataset
    """
    def setup_experiment(self, config):
        """
            - train_model_func: Optional user defined function to train the model,
                                expected to behave similarly to `train_model`
                                in terms of input parameters and return values
            - evaluate_model_func: Optional user defined function to validate the model
                                   expected to behave similarly to `evaluate_model`
                                   in terms of input parameters and return values
        """
        config = copy.copy(config)
        config.setdefault("epochs", 1)  # Necessary for next line.
        config.setdefault("epochs_to_validate", range(config["epochs"] - 3,
                                                      config["epochs"]))

        super().setup_experiment(config)

        self.train_model = config.get("train_model_func", train_model)
        self.evaluate_model = config.get("evaluate_model_func", evaluate_model)

    def train_epoch(self):
        self.train_model(
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

    def validate(self, loader=None):
        if loader is None:
            loader = self.val_loader

        return self.evaluate_model(
            model=self.model,
            loader=loader,
            device=self.device,
            criterion=self.error_loss,
            complexity_loss_fn=self.complexity_loss,
            batches_in_epoch=self.batches_in_epoch_val,
            transform_to_device_fn=self.transform_data_to_device,
        )

    @classmethod
    def load_dataset(cls, config, train=True):
        config = copy.copy(config)
        config.setdefault("dataset_class", datasets.imagenet)
        if "dataset_args" not in config:
            config["dataset_args"] = dict(
                data_path=config["data"],
                train_dir=config.get("train_dir", "train"),
                val_dir=config.get("val_dir", "val"),
                num_classes=config.get("num_classes", 1000),
                use_auto_augment=config.get("use_auto_augment", False),
                sample_transform=config.get("sample_transform", None),
                target_transform=config.get("target_transform", None),
                replicas_per_sample=config.get("replicas_per_sample", 1),
            )

        return super().load_dataset(config, train)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        exp = "ImagenetExperiment"

        # Extended methods
        eo["setup_experiment"].insert(0, exp + ": Compatibility shims")
        eo["setup_experiment"].append(exp + ": Additional setup")
        eo["load_dataset"].insert(0, exp + ": Set default dataset")

        eo.update(
            # Overwritten methods
            train_model=[exp + ": Call train_model_func"],
            validate=[exp + ": Call evaluate_model_func"],
        )

        return eo
