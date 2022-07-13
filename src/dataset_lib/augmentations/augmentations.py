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

import functools

from augmentations import pwc_augmentation
from augmentations import raft_augmentation
from augmentations import simple_augmentation


def no_aug(element, aug_params):
  del aug_params
  return element


ALL_AUGMENTATIONS = {
    'raft': raft_augmentation.apply,
    'pwc': pwc_augmentation.apply,
    'crop': simple_augmentation.apply_crop,
    'resize': simple_augmentation.apply_resize,
    'none': no_aug,
}


def get_augmentation_fn(aug_params):
  aug_name = aug_params.name
  if aug_name not in ALL_AUGMENTATIONS.keys():
    raise NotImplementedError(
        'Unrecognized augmentation: {}'.format(aug_name))
  aug_fn = ALL_AUGMENTATIONS[aug_name]
  return functools.partial(aug_fn, aug_params=aug_params)
