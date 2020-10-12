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

import io

import torch

from nupic.research.frameworks.pytorch.model_utils import (
    deserialize_state_dict,
    serialize_state_dict,
)
from nupic.research.frameworks.vernon.network_utils import (
    create_model,
    get_compatible_state_dict,
)

__all__ = [
    "HasModel",
]


class HasModel:
    """
    Creates a self.model, serializes it, and adds extensibility methods for
    customizing model creation.
    """
    def setup_experiment(self, config):
        """
        :param config: Dictionary containing the configuration parameters

            - model_class: Model class. Must inherit from "torch.nn.Module"
            - model_args: model model class arguments passed to the constructor
            - init_batch_norm: Whether or not to Initialize running batch norm
                               mean to 0.
            - checkpoint_file: if not None, will start from this model. The model
                               must have the same model_args and model_class as the
                               current experiment.
            - load_checkpoint_args: args to be passed to `load_state_from_checkpoint`
        """
        super().setup_experiment(config)
        self.device = config.get("device",
                                 (torch.device("cuda"
                                  if torch.cuda.is_available()
                                  else "cpu")))
        self.model = self.create_model(config, self.device)
        self.transform_model()
        self.logger.debug(self.model)

    @classmethod
    def create_model(cls, config, device):
        """
        Create imagenet model from an ImagenetExperiment config
        :param config:
            - model_class: Model class. Must inherit from "torch.nn.Module"
            - model_args: model model class arguments passed to the constructor
            - init_batch_norm: Whether or not to Initialize running batch norm
                               mean to 0.
            - checkpoint_file: if not None, will start from this model. The
                               model must have the same model_args and
                               model_class as the current experiment.
            - load_checkpoint_args: args to be passed to `load_state_from_checkpoint`
        :param device:
            Pytorch device

        :return:
                Model instance
        """
        return create_model(
            model_class=config["model_class"],
            model_args=config.get("model_args", {}),
            init_batch_norm=config.get("init_batch_norm", False),
            device=device,
            checkpoint_file=config.get("checkpoint_file", None),
            load_checkpoint_args=config.get("load_checkpoint_args", {}),
        )

    def transform_model(self):
        """Placeholder for any model transformation required prior to training"""

    def get_state(self):
        """
        Get experiment serialized state as a dictionary of  byte arrays
        :return: dictionary with "model", "optimizer" and "lr_scheduler" states
        """
        state = super().get_state()

        # Save state into a byte array to avoid ray's GPU serialization issues
        # See https://github.com/ray-project/ray/issues/5519
        with io.BytesIO() as buffer:
            serialize_state_dict(buffer, self.model.module.state_dict())
            state["model"] = buffer.getvalue()

        return state

    def set_state(self, state):
        """
        Restore the experiment from the state returned by `get_state`
        :param state: dictionary with "model", "optimizer", "lr_scheduler"
        states
        """
        super().set_state(state)
        if "model" in state:
            with io.BytesIO(state["model"]) as buffer:
                state_dict = deserialize_state_dict(buffer, self.device)
            state_dict = get_compatible_state_dict(state_dict, self.model.module)
            self.model.module.load_state_dict(state_dict)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()

        name = "HasModel"

        # Extended methods
        eo["setup_experiment"].append(name + ".setup_experiment")
        eo["get_state"].append(name + ": Get model")
        eo["set_state"].append(name + ": Set model")

        eo.update(
            # New methods
            create_model=[name + ".create_model"],
            transform_model=[],
        )

        return eo
