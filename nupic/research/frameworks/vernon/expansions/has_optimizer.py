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

__all__ = [
    "HasOptimizer",
]


class HasOptimizer:
    """
    Creates a self.optimizer and handles its serialization. Requires HasModel or
    similar expansion. Classes inheriting from HasOptimizer should call
    post_optimizer_step after calling self.optimizer.step().
    """
    def setup_experiment(self, config):
        """
        :param config: Dictionary containing the configuration parameters

            - optimizer_class: Optimizer class.
                               Must inherit from "torch.optim.Optimizer"
            - optimizer_args: Optimizer class class arguments passed to the
                              constructor
        """
        super().setup_experiment(config)

        assert hasattr(self, "model"), (
            "Must use HasModel or similar expansion, and must place it "
            "early in the setup_experiment chain."
        )

        self.optimizer = self.create_optimizer(self.model, config)

    @classmethod
    def create_optimizer(cls, model, config):
        """
        :param config: Dictionary containing the configuration parameters
            - optimizer_class: Optimizer class.
                               Must inherit from "torch.optim.Optimizer"
            - optimizer_args: Optimizer class class arguments passed to the
                              constructor
        """
        optimizer_class = config.get("optimizer_class", torch.optim.SGD)
        optimizer_args = config.get("optimizer_args", {})
        return optimizer_class(model.parameters(), **optimizer_args)

    def post_optimizer_step(self):
        """
        An extension point for running code immediately after each optimizer
        step.
        """

    def get_state(self):
        state = super().get_state()

        with io.BytesIO() as buffer:
            serialize_state_dict(buffer, self.optimizer.state_dict())
            state["optimizer"] = buffer.getvalue()

        return state

    def set_state(self, state):
        super().set_state(state)
        if "optimizer" in state:
            with io.BytesIO(state["optimizer"]) as buffer:
                state_dict = deserialize_state_dict(buffer, self.device)
            self.optimizer.load_state_dict(state_dict)

    def get_lr(self):
        """
        Returns the current learning rate
        :return: list of learning rates used by the optimizer
        """
        return [p["lr"] for p in self.optimizer.param_groups]

    def get_weight_decay(self):
        """
        Returns the current weight decay
        :return: list of weight decays used by the optimizer
        """
        return [p["weight_decay"] for p in self.optimizer.param_groups]

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()

        name = "HasOptimizer"

        # Extended methods
        eo["setup_experiment"].append(name + ".setup_experiment")
        eo["get_state"].append(name + ": Get optimizer")
        eo["set_state"].append(name + ": Set optimizer")

        eo.update(
            # New methods
            post_optimizer_step=[],
        )

        return eo
