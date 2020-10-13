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

from .constrain_parameters import ConstrainParameters
from .export_model import ExportModel
from .extra_validations import ExtraValidations
from .knowledge_distillation import KnowledgeDistillation, KnowledgeDistillationCL
from .load_preprocessed_data import LoadPreprocessedData
from .log_backprop_structure import LogBackpropStructure
from .log_covariance import LogCovariance
from .log_every_learning_rate import LogEveryLearningRate
from .log_every_loss import LogEveryLoss
from .lr_range_test import LRRangeTest, create_lr_test_experiment
from .maxup import MaxupStandard, MaxupPerSample
from .mixed_precision import MixedPrecision
from .multi_cycle_lr import MultiCycleLR
from .profile import Profile
from .profile_autograd import ProfileAutograd
from .prune_low_magnitude import PruneLowMagnitude
from .prune_low_snr import PruneLowSNR
from .regularize_loss import RegularizeLoss
from .rezero_weights import RezeroWeights
from .save_final_checkpoint import SaveFinalCheckpoint
from .selective_weight_decay import SelectiveWeightDecay
from .update_boost_strength import UpdateBoostStrength
from .cutmix import CutMix, CutMixKnowledgeDistillation
from .quantization_aware import QuantizationAware
from .reduce_lr_after_task import ReduceLRAfterTask
from .vary_batch_size import VaryBatchSize
from .ewc import ElasticWeightConsolidation
from .oml import OnlineMetaLearning
