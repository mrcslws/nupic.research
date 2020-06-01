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

import torch


class LogEveryLoss:
    """
    Include the training loss for every batch in the result dict.

    This class must be placed earlier in the mixin order than other mixins that
    modify the loss.

    class MyExperiment(mixins.LogEveryLoss,
                       ...
                       mixins.RegularizeLoss,
                       ImagenetExperiment):
        pass

    """
    def setup_experiment(self, config):
        super().setup_experiment(config)
        self.error_loss_by_batch = []
        self.complexity_loss_by_batch = []
        self.test_complexity_loss = None

    def error_loss(self, *args, **kwargs):
        loss = super().error_loss(*args, **kwargs)
        if self.model.training:
            self.error_loss_by_batch.append(loss.detach().clone())
        return loss

    def complexity_loss(self, *args, **kwargs):
        loss = super().complexity_loss(*args, **kwargs)
        if loss is not None:
            if self.model.training:
                self.complexity_loss_by_batch.append(loss.detach().clone())
            else:
                self.test_complexity_loss = loss.detach().clone()

        return loss

    def run_epoch(self):
        result = super().run_epoch()

        log = torch.stack(self.error_loss_by_batch)
        result["error_loss_by_batch"] = log.cpu().tolist()
        self.error_loss_by_batch = []

        if len(self.complexity_loss_by_batch) > 0:
            log = torch.stack(self.complexity_loss_by_batch)
            result["complexity_loss_by_batch"] = log.cpu().tolist()
            self.complexity_loss_by_batch = []

        if self.test_complexity_loss is not None:
            result["test_complexity_loss"] = self.test_complexity_loss.item()
            self.test_complexity_loss = None

        return result

    @classmethod
    def aggregate_results(cls, results):
        aggregated = super().aggregate_results(results)

        k = "error_loss_by_batch"
        loss_by_process_and_batch = torch.Tensor(len(results),
                                                 len(results[0][k]))
        for rank, result in enumerate(results):
            loss_by_process_and_batch[rank, :] = torch.tensor(result[k])
        aggregated[k] = loss_by_process_and_batch.mean(dim=0).tolist()

        # "complexity_loss_by_batch" and "test_complexity_loss" don't need to be
        # aggregated, since they're the same on every process.

        return aggregated

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo["setup_experiment"].append("LogEveryLoss: initialize")
        eo["error_loss"].append("LogEveryLoss: copy loss")
        eo["complexity_loss"].append("LogEveryLoss: copy loss")
        eo["run_epoch"].append("LogEveryLoss: to result dict")
        eo["aggregate_results"].append("LogEveryLoss: Aggregate")
        return eo
