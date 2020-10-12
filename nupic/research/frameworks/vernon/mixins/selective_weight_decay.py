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
from torch.nn.modules.batchnorm import _BatchNorm


class SelectiveWeightDecay:
    """
    Adds functionality for performing weight decay on only certain parameters.
    """
    @classmethod
    def create_optimizer(cls, model, config):
        """
            - batch_norm_weight_decay: Whether or not to apply weight decay to
                                       batch norm modules parameters
                                       See https://arxiv.org/abs/1807.11205
            - bias_weight_decay: Whether or not to apply weight decay to
                                       bias parameters
        """
        group_decay, group_no_decay = [], []
        for module in model.modules():
            for name, param in module.named_parameters(recurse=False):
                if isinstance(module, _BatchNorm):
                    decay = config.get("batch_norm_weight_decay", True)
                elif name == "bias":
                    decay = config.get("bias_weight_decay", True)
                else:
                    decay = True

                if decay:
                    group_decay.append(param)
                else:
                    group_no_decay.append(param)

        optimizer_class = config.get("optimizer_class", torch.optim.SGD)
        optimizer_args = config.get("optimizer_args", {})
        return optimizer_class([dict(params=group_decay),
                                dict(params=group_no_decay,
                                     weight_decay=0.)],
                               **optimizer_args)

    @classmethod
    def get_execution_order(cls):
        eo = super().get_execution_order()
        eo.update(
            # Replaced methods
            create_optimizer=["SelectiveWeightDecay.create_optimizer"]
        )
        return eo
