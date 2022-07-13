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

import tensorflow as tf

from augmentations import color_aug
from augmentations import crop_aug
from augmentations import spatial_aug


FLOW_SCALE_FACTOR = 20.0

INVALID_FLOW_VALUE = 1e9


def apply(element, aug_params):
  images, forward_flow = element['inputs'], element['label']

  is_hard_augment = tf.greater_equal(aug_params.prob_hard_sample,
                                     tf.random.uniform([], 0, 1))
  is_hard_augment = tf.cast(is_hard_augment, dtype=tf.bool)

  images, forward_flow = tf.cond(tf.cast(aug_params.is_augment_spatial, dtype=tf.bool),
                                 lambda: spatial_aug.apply(images, forward_flow, aug_params, is_hard_augment),
                                 lambda: no_spatial_op(images, forward_flow, aug_params, is_hard_augment))

  return tf.cond(
      tf.logical_and(is_hard_augment, aug_params.is_augment_colors),
      lambda: chromatic_aug(images, forward_flow, aug_params),
      lambda: no_op(images, forward_flow))


def no_spatial_op(images, forward_flow, augmentation_params, is_hard_augment):
  _, height, width, _ = tf.unstack(tf.shape(images), num=4)

  crop_start_y = tf.random.uniform([],
                                   minval=0,
                                   maxval=height -
                                   augmentation_params.crop_height + 1,
                                   dtype=tf.dtypes.int32)
  crop_start_x = tf.random.uniform([],
                                   minval=0,
                                   maxval=width -
                                   augmentation_params.crop_width + 1,
                                   dtype=tf.dtypes.int32)
  images, forward_flow = crop_aug.crop_to_box(images,
                                              forward_flow,
                                              crop_start_y,
                                              crop_start_x,
                                              augmentation_params.crop_height,
                                              augmentation_params.crop_width)

  if augmentation_params.disable_ground_truth:
    # Set GT to invalid values for semi-supervised training
    forward_flow = tf.ones(forward_flow.get_shape())*INVALID_FLOW_VALUE

  return images, forward_flow


def chromatic_aug(images, forward_flow, augmentation_params):
  # Should be faster than color->spatial and have same effect
  images = color_aug.apply(images, augmentation_params)
  return {'inputs': images, 'label': forward_flow}


def no_op(images, forward_flow):
  return {'inputs': images, 'label': forward_flow}



