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

"""Contains various data augmentation utilities for PWC-Net."""
import math

import tensorflow as tf


def apply_horizontal_flip(images, forward_flow):
  """Applies a horizontal flip to images and forward_flow."""
  images = tf.image.flip_left_right(images)
  forward_flow = tf.image.flip_left_right(forward_flow)

  # Invert the horizontal component of flow. This is because an object moving to
  # the right in the input images (positive horizontal flow) will be moving to
  # the left in the flipped images (negative horizontal flow) and vice-versa.
  flow_scale_factors = tf.constant([-1, 1], dtype=tf.float32)
  forward_flow = forward_flow * flow_scale_factors
  return images, forward_flow


def random_vertical_flip(img1, img2, flow, prob=0.1):
  pred = tf.random.uniform([]) < prob
  def true_fn(img1, img2, flow):
    img1 = tf.image.flip_up_down(img1)
    img2 = tf.image.flip_up_down(img2)
    flow = tf.image.flip_up_down(flow) * [1.0, -1.0]
    return img1, img2, flow
  def false_fn(img1, img2, flow):
    return img1, img2, flow
  return tf.cond(pred, lambda: true_fn(img1, img2, flow),
                 lambda: false_fn(img1, img2, flow))


def rotated_box_size(rotation_degrees, augmentation_params):
  """Returns the bounding box size after accounting for its rotation."""
  box_diagonal = math.hypot(augmentation_params.crop_width,
                            augmentation_params.crop_height)
  diagonal_angle = math.atan2(augmentation_params.crop_height,
                              augmentation_params.crop_width)
  absolute_rotation_radians = tf.math.abs(rotation_degrees * math.pi / 180)
  rotated_height = box_diagonal * tf.sin(diagonal_angle +
                                         absolute_rotation_radians)
  rotated_width = box_diagonal * tf.cos(diagonal_angle -
                                        absolute_rotation_radians)
  return rotated_height, rotated_width

