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

__all__ = [
    "HasEpochs",
]


class HasEpochs:
    """
    Adds a self.current_epoch, incrementing logic, serialization, and hooks that
    can be extended. Classes inheriting from HasEpochs must call pre_epoch and
    post_epoch.
    """
    def setup_experiment(self, config):
        super().setup_experiment(config)
        self.current_epoch = 0

    def get_current_epoch(self):
        """
        Returns the current epoch of the running experiment
        """
        return self.current_epoch

    def pre_epoch(self):
        """
        A hook for running code before every epoch.
        """

    def post_epoch(self):
        """
        A hook for running code after every epoch.
        """
        self.current_epoch += 1

    def set_state(self, state):
        super().set_state(state)
        self.current_epoch = state["current_epoch"]

    def get_state(self):
        state = super().get_state()
        state["current_epoch"] = self.current_epoch
        return state

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()

        name = "HasEpochs"

        # Extended methods
        eo["setup_experiment"].append(name + ": Set current epoch to 0")
        eo["set_state"].append(name + ": Set current epoch")
        eo["get_state"].append(name + ": Get current epoch")

        eo.update(
            # New methods
            pre_epoch=[],
            post_epoch=[name + ": Increment current epoch"],
        )

        return eo
