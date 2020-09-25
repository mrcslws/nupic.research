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

"""
A simple implementation of dendrite segments. This is meant to offer, at the least,
a template for revision and further modifications.
"""

import math

import numpy as np
import torch
from torch.nn import init

torch.nn.Linear

class DendriteSegments(torch.nn.Module):
    """
    This implements dendrite segments over a set of units. Each unit has a set of
    segments modeled by a linear transformation from a context vector to output value
    for each segment.

    TODO: Include a optional bias (dim num_segments)
    """

    def __init__(self, num_units, num_segments, dim_context, sparsity, bias=None):
        """
        :param num_units: number of units i.e. neurons;
                        each unit will have it's own set of dendrite segments
        :param dim_context: length of the context vector;
                            the same context will be applied to each segment
        :param num_segments: number of dendrite segments per unit
        :param sparsity: sparsity of connections;
                        this is over each linear transformation from
                        dim_context to num_segments
        """
        super().__init__()

        # Save params.
        self.num_units = num_units
        self.num_segments = num_segments
        self.dim_context = dim_context
        self.sparsity = sparsity

        # TODO: Use named dimensions.
        segment_weights = torch.Tensor(num_units, num_segments, dim_context)
        self.segment_weights = torch.nn.Parameter(segment_weights)

        # Create a bias per unit per segment.
        if bias:
            segment_biases = torch.Tensor(num_units, num_segments)
            self.segment_biases = torch.nn.Parameter(segment_biases)
        else:
            self.register_parameter("segment_biases", None)
        self.reset_parameters()

        # Create a random mask for each unit (dim=0)
        # TODO: Need sparsity per unit per segment.
        zero_mask = random_mask(self.segment_weights.shape, sparsity=sparsity, dim=0)

        # Use float16 because pytorch distributed nccl doesn't support bools.
        self.register_buffer("zero_mask", zero_mask.half())

        self.rezero_weights()

    def reset_parameters(self):
        """Initialize the linear transformation for each unit."""
        for unit in range(self.num_units):
            weight = self.segment_weights[unit, ...]
            bias = self.segment_biases[unit, ...]
            init_linear_(weight, bias)

    def rezero_weights(self):
        self.segment_weights.data[self.zero_mask.bool()] = 0

    def forward(self, context):
        """
        Matrix-multiply the context with the weight tensor for each dendrite segment.
        This is done for each unit and so the output is of length num_units.
        """

        # Matrix multiply using einsum:
        #    * b => the batch dimension
        #    * k => the context dimension; multiplication will be along this dimension
        #    * ij => the units and segment dimensions, respectively
        # W^C * M^C * C -> num_units x num_segments
        output = torch.einsum("ijk,bk->bij", self.segment_weights, context)

        if self.segment_biases is not None:
            output += self.segment_biases
        return output


def init_linear_(weight, bias=None):
    """
    Performs the default initilization of a weight and bias parameter
    of a linear layaer; done in-place.
    """
    init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)


def random_mask(size, sparsity, dim=None, **kwargs):
    """
    This creates a random off-mask (True => off) of 'size' with the
    specified 'sparsity' level along 'dim'. If 'dim' is 1, for instance,
    then `mask[:, d, ...]` has the desired sparsity for all d. If None,
    the sparsity is applied over the whole tensor.

    :param size: shape of tensor
    :param sparsity: fraction of non-zeros
    :param dim: which dimension to apply the sparsity
    :param kwargs: keywords args passed to torch.ones;
                   helpful for specifying device, for instace
    """

    assert 0 <= sparsity <= 1

    # Start with all elements off.
    mask = torch.ones(size, **kwargs)

    # Find sparse submasks along dim; recursively call 'random_mask'.
    if dim is not None:
        len_of_dim = mask.shape[dim]
        for d in range(len_of_dim):
            dim_slice = [slice(None)] * len(mask.shape)
            dim_slice[dim] = d
            sub_mask = mask[dim_slice]
            sub_mask[:] = random_mask(
                sub_mask.shape,
                sparsity, **kwargs, dim=None
            )
        return mask

    # Randomly choose indices to make non-zero ("nz").
    mask_flat = mask.view(-1)  # flattened view
    num_total = mask_flat.shape[0]
    num_nz = int(round((1 - sparsity) * num_total))
    on_indices = np.random.choice(num_total, num_nz, replace=False)
    mask_flat[on_indices] = False

    return mask


if __name__ == "__main__":

    dendrite_segment = DendriteSegments(
        num_units=10, num_segments=20, dim_context=15, sparsity=0.7, bias=True
    )
    dendrite_segment.rezero_weights()

    batch_size = 8
    context = torch.rand(batch_size, dendrite_segment.dim_context)
    out = dendrite_segment(context)

    print(f"out.shape={out.shape}")
