# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import ml_collections


def get_params(name):
  aug_params = ml_collections.ConfigDict()

  aug_params.name = name

  """Parameters controlling data augmentation."""
  aug_params.crop_height = 320
  aug_params.crop_width = 448
  aug_params.eval_crop_height = 320
  aug_params.eval_crop_width = 768

  aug_params.noise_std_range = 0.06  # range for sampling std of additive noise
  aug_params.crop_range_delta = 0.03  # range of relative translation of image 2
  aug_params.flow_interpolation = "BILINEAR"  # "NEAREST"

  # control params
  aug_params.is_schedule_coeff = True  # schedule aug coeff for image 2
  aug_params.schedule_coeff = 1.0
  aug_params.is_channel_swapping = False  # True: random swapping color channels
  aug_params.is_augment_colors = True
  aug_params.is_augment_spatial = True
  aug_params.disable_ground_truth = False  # True: set ground truth to invalid for semi-supervised training
  aug_params.black = False  # True: allow out-of-boundary cropping (Chairs)
  aug_params.prob_hard_sample = 1.0  # probability that we use the hard sample technique, see line 87 in https://github.com/gengshan-y/VCN/blob/master/dataloader/robloader.py
  aug_params.is_random_erasing = False

  # spatial params
  aug_params.min_scale = 0.2
  aug_params.max_scale = 1.0
  aug_params.vflip_prob = 0.0
  aug_params.rot1 = 0.4
  aug_params.squeeze1 = 0.3
  aug_params.scale1 = 0.3
  aug_params.tran1 = 0.4
  aug_params.scale2 = 0.1
  aug_params.lmult_factor = 1.
  aug_params.sat_factor = 1.
  aug_params.col_factor = 1.
  aug_params.ladd_factor = 1.
  aug_params.col_rot_factor = 1.
  return aug_params
