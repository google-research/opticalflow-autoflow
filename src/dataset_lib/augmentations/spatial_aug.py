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

import numpy as np

import tensorflow as tf

from tensorflow.contrib import image as contrib_image

from augmentations import crop_aug
from augmentations import image_aug


MAX_SAMPLES = 50  # max number of rejection samples
INVALID_THRESHOLD = 1e3
INVALID_FLOW_VALUE = 1e9
FLOW_SCALE_FACTOR = 20


class SpatialAug(object):
  """Sample spatial agumentation parameters, borrowed from VCN's pytorch implementation of FlowNet/PWC-Net."""

  def __init__(self,
               crop,
               scale=None,
               rot=None,
               trans=None,
               squeeze=None,
               schedule_coeff=1,
               order=1,
               black=False):
    self.crop = crop
    self.scale = scale
    self.rot = rot
    self.trans = trans
    self.squeeze = squeeze
    self.t_0 = tf.zeros(1)
    self.t_1 = tf.zeros(1)
    self.t_2 = tf.zeros(1)
    self.t_3 = tf.zeros(1)
    self.t_4 = tf.zeros(1)
    self.t_5 = tf.zeros(1)
    self.schedule_coeff = schedule_coeff
    self.order = order
    self.black = black

  def to_identity(self):
    """Identity transformation."""
    self.t_0 = tf.constant([1], dtype=tf.float32)
    self.t_1 = tf.constant([0], dtype=tf.float32)
    self.t_2 = tf.constant([0], dtype=tf.float32)
    self.t_3 = tf.constant([1], dtype=tf.float32)
    self.t_4 = tf.constant([0], dtype=tf.float32)
    self.t_5 = tf.constant([0], dtype=tf.float32)

  def left_multiply(self, u0, u1, u2, u3, u4, u5):
    """Composite transformations."""
    result_0 = self.t_0 * u0 + self.t_1 * u2
    result_1 = self.t_0 * u1 + self.t_1 * u3

    result_2 = self.t_2 * u0 + self.t_3 * u2
    result_3 = self.t_2 * u1 + self.t_3 * u3

    result_4 = self.t_4 * u0 + self.t_5 * u2 + u4
    result_5 = self.t_4 * u1 + self.t_5 * u3 + u5

    self.t_0 = result_0
    self.t_1 = result_1
    self.t_2 = result_2
    self.t_3 = result_3
    self.t_4 = result_4
    self.t_5 = result_5

  def inverse(self):
    """Compute inverse transformation."""
    a = self.t_0
    c = self.t_2
    e = self.t_4
    b = self.t_1
    d = self.t_3
    f = self.t_5

    denom = a * d - b * c

    result_0 = d / denom
    result_1 = -b / denom
    result_2 = -c / denom
    result_3 = a / denom
    result_4 = (c * f - d * e) / denom
    result_5 = (b * e - a * f) / denom

    return tf.stack(
        [result_0, result_1, result_2, result_3, result_4, result_5])

  def grid_transform(self, meshgrid, t, normalize=True, gridsize=None):
    """Transform grid according to transformation."""
    if gridsize is None:
      h, w = meshgrid[0].shape
    else:
      h, w = gridsize

    vgrid = tf.stack([(meshgrid[0] * t[0] + meshgrid[1] * t[2] + t[4]),
                      (meshgrid[0] * t[1] + meshgrid[1] * t[3] + t[5])], 2)
    if normalize:
      # normlize for pytorch style [-1, 1]
      vgridx = 2.0 * vgrid[:, :, 0] / tf.math.maximum(w - 1, 1) - 1.0
      vgridy = 2.0 * vgrid[:, :, 1] / tf.math.maximum(h - 1, 1) - 1.0
      return  tf.stack([vgridx, vgridy], 2)
    else:
      return vgrid

  def __call__(self, h, w):
    th, tw = self.crop
    # meshgrid = np.meshgrid(range(th), range(tw))[::-1]
    cornergrid = np.meshgrid([0, th - 1], [0, tw - 1])[::-1]

    def cond(out, not_found, i, max_iters):
      del out

      return tf.math.logical_and(not_found, tf.less(i, max_iters))

    def body(out, not_found, i, max_iters):
      del not_found

      # Compute transformation for first image.
      self.to_identity()
      # Center.
      self.left_multiply(1, 0, 0, 1, -.5 * tw, -.5 * th)
      scale0 = 1
      scale1 = 1
      squeeze0 = 1
      squeeze1 = 1
      # Sample rotation.
      if self.rot is None:
        rot0 = 0.0
        rot1 = 0.0
      else:
        rot0 = tf.random.uniform([], minval=-self.rot[0], maxval=self.rot[0])
        rot1 = tf.random.uniform(
            [],
            minval=-self.rot[1] * self.schedule_coeff,
            maxval=self.rot[1] * self.schedule_coeff) + rot0
        self.left_multiply(
            tf.math.cos(rot0), tf.math.sin(rot0), -tf.math.sin(rot0),
            tf.math.cos(rot0), 0, 0)

      # Sample scale & squeeze.
      if self.squeeze is None:
        squeeze0 = 1.0
        squeeze1 = 1.0
      else:
        squeeze0 = tf.math.exp(
            tf.random.uniform([],
                              minval=-self.squeeze[0],
                              maxval=self.squeeze[0]))
        squeeze1 = tf.math.exp(
            tf.random.uniform(
                [],
                minval=-self.squeeze[1] * self.schedule_coeff,
                maxval=self.squeeze[1] * self.schedule_coeff)) * squeeze0

      if self.scale is None:
        scale0 = 1.0
        scale1 = 1.0
      else:
        scale0 = tf.math.exp(
            tf.random.uniform([],
                              minval=self.scale[2] - self.scale[0],
                              maxval=self.scale[2] + self.scale[0]))
        scale1 = tf.math.exp(
            tf.random.uniform(
                [],
                minval=-self.scale[1] * self.schedule_coeff,
                maxval=self.scale[1] * self.schedule_coeff)) * scale0

      self.left_multiply(1.0 / (scale0 * squeeze0), 0, 0,
                         1.0 / (scale0 / squeeze0), 0, 0)

      # Sample translation.
      if self.trans is None:
        trans0 = [0.0, 0.0]
        trans1 = [0.0, 0.0]
      else:
        trans0 = tf.random.uniform([2],
                                   minval=-self.trans[0],
                                   maxval=self.trans[0])
        trans1 = tf.random.uniform(
            [2],
            minval=-self.trans[1] * self.schedule_coeff,
            maxval=self.trans[1] * self.schedule_coeff) + trans0

        self.left_multiply(1, 0, 0, 1, trans0[0] * tw, trans0[1] * th)

      self.left_multiply(1, 0, 0, 1, .5 * float(w), .5 * float(h))
      transmat0_0 = tf.identity(self.t_0)
      transmat0_1 = tf.identity(self.t_1)
      transmat0_2 = tf.identity(self.t_2)
      transmat0_3 = tf.identity(self.t_3)
      transmat0_4 = tf.identity(self.t_4)
      transmat0_5 = tf.identity(self.t_5)
      transmat0 = [
          transmat0_0, transmat0_1, transmat0_2, transmat0_3, transmat0_4,
          transmat0_5
      ]

      # Compute transformation for second image.
      self.to_identity()
      self.left_multiply(1, 0, 0, 1, -.5 * tw, -.5 * th)
      if self.rot is not None:
        self.left_multiply(
            tf.math.cos(rot1), tf.math.sin(rot1), -tf.math.sin(rot1),
            tf.math.cos(rot1), 0, 0)
      if self.trans is not None:
        self.left_multiply(1, 0, 0, 1, trans1[0] * tw, trans1[1] * th)
      self.left_multiply(1.0 / (scale1 * squeeze1), 0, 0,
                         1.0 / (scale1 / squeeze1), 0, 0)
      self.left_multiply(1, 0, 0, 1, .5 * float(w), .5 * float(h))
      transmat1_0 = tf.identity(self.t_0)
      transmat1_1 = tf.identity(self.t_1)
      transmat1_2 = tf.identity(self.t_2)
      transmat1_3 = tf.identity(self.t_3)
      transmat1_4 = tf.identity(self.t_4)
      transmat1_5 = tf.identity(self.t_5)
      transmat1 = [
          transmat1_0, transmat1_1, transmat1_2, transmat1_3, transmat1_4,
          transmat1_5
      ]

      sum_val0 = tf.math.reduce_sum(
          tf.to_float(
              tf.math.abs(
                  self.grid_transform(
                      cornergrid, transmat0, gridsize=[float(h),
                                                       float(w)])) > 1))
      sum_val1 = tf.math.reduce_sum(
          tf.to_float(
              tf.math.abs(
                  self.grid_transform(
                      cornergrid, transmat1, gridsize=[float(h),
                                                       float(w)])) > 1))
      bool_val = tf.logical_or(
          tf.math.equal((sum_val0 + sum_val1), 0), self.black)

      out = (
          (rot0 * 180 / 3.14),
          (scale0 * squeeze0),
          (scale0 / squeeze0),
          (rot1 * 180 / 3.14),
          (scale1 * squeeze1),
          (scale1 / squeeze1),
          )

      return [out, tf.math.logical_not(bool_val), tf.add(i, 1), max_iters]

    identity_val = tf.constant([0.], shape=()), tf.constant(
        [1.], shape=()), tf.constant([1.], shape=()), tf.constant(
            [0.], shape=()), tf.constant([1.], shape=()), tf.constant([1.],
                                                                      shape=())
    not_found = tf.ones([], dtype=tf.bool)
    ret_val, not_found, _, _ = tf.while_loop(
        cond, body, [identity_val, not_found, 0, MAX_SAMPLES])

    return tf.cond(not_found, lambda: identity_val, lambda: ret_val)

def apply(images, forward_flow, augmentation_params, is_hard_augment):
  """Augments the inputs by applying random spatial transformations.

  Args:
    images: A tensor of size [2, height, width, 3] representing the two images.
    forward_flow: A tensor of size [height, width, 2] representing the flow from
      the first image to the second.
    augmentation_params: An AugmentationParams controlling the augmentations to
      be performed.

  Returns:
    (images, forward_flow) after any spatial transformations.
  """
  images = tf.convert_to_tensor(images)
  input_image_height = images.shape[1]
  input_image_width = images.shape[2]
  crop_height = augmentation_params.crop_height
  crop_width = augmentation_params.crop_width

  # For quick experiment: sample a valid one, re-sample crop-center
  _, input_image_height, input_image_width, _ = tf.unstack(
      tf.shape(images), num=4)

  # Sample rotations and stretches for each images.
  rotation_degrees, stretch_factor_y, stretch_factor_x, rotation_degrees2, stretch_factor_y2, stretch_factor_x2 = _rejection_sample_spatial_aug_parameters(
      input_image_height, input_image_width, augmentation_params,
      is_hard_augment)

  # Sample crop_centers for each images.
  crop_center_y, crop_center_x, crop_center_y2, crop_center_x2 = crop_aug.sample_cropping_centers(
      input_image_height, input_image_width, stretch_factor_y, stretch_factor_x,
      rotation_degrees, stretch_factor_y2, stretch_factor_x2, rotation_degrees2,
      augmentation_params)

  # Transform first image.
  crop_center_y += 1
  crop_center_x += 1
  transform = crop_aug.compose_cropping_transformation(
      stretch_factor_y, stretch_factor_x, crop_center_y, crop_center_x,
      rotation_degrees, crop_height, crop_width)

  transform = tf.reshape(transform, [-1])[:8]  # tf is row-based
  output_shape = tf.stack([crop_height, crop_width])
  aug_image = contrib_image.transform(
      images[0, :, :, :],
      transform,
      interpolation="BILINEAR",
      output_shape=output_shape)

  # Transform flow
  aug_flow = contrib_image.transform(
      forward_flow,
      transform,
      interpolation=augmentation_params.flow_interpolation,
      output_shape=output_shape)
  # print('forward_flow316', np.max(forward_flow))

  all_ones = tf.ones(tf.shape(forward_flow))
  aug_all_ones = contrib_image.transform(
      all_ones,
      transform,
      interpolation=augmentation_params.flow_interpolation,
      output_shape=output_shape)
  # Mark invalid pixels (extreme value or out-of-boundary)
  invalid_mask = tf.logical_or(
      tf.abs(aug_flow) > tf.to_float(INVALID_THRESHOLD), aug_all_ones < 1.0)
  # invalid_mask = tf.abs(aug_flow) > tf.to_float(INVALID_THRESHOLD)
  invalid_flow = tf.ones(aug_flow.get_shape()) * INVALID_FLOW_VALUE

  # Transform second image
  crop_center_y2 += 1
  crop_center_x2 += 1
  transform2 = crop_aug.compose_cropping_transformation(
      stretch_factor_y2, stretch_factor_x2, crop_center_y2, crop_center_x2,
      rotation_degrees2, crop_height, crop_width)

  # Compute reverse transform for augmenting flow
  transform2_inv = tf.linalg.inv(transform2)
  transform2_inv = tf.reshape(transform2_inv, [-1])[:8]
  transform2 = tf.reshape(transform2, [-1])[:8]
  aug_image2 = contrib_image.transform(
      images[1, :, :, :],
      transform2,
      interpolation="BILINEAR",
      output_shape=output_shape)

  # Composite augmented image pairs.
  images = tf.stack([aug_image, aug_image2])

  # Compute augmented optical flow.
  # Compute position in transformed first image.
  x, y = tf.meshgrid(tf.range(crop_width), tf.range(crop_height))
  x = tf.to_float(x)
  y = tf.to_float(y)
  # Map to coordinates of first image.
  x0 = x * transform[0] + y * transform[1] + transform[2]
  y0 = x * transform[3] + y * transform[4] + transform[5]
  # Map to coordinates of second image.
  x1 = x0 + aug_flow[:, :, 0] * FLOW_SCALE_FACTOR
  y1 = y0 + aug_flow[:, :, 1] * FLOW_SCALE_FACTOR
  # Map to coordinates of augmented second image.
  x11 = x1 * transform2_inv[0] + y1 * transform2_inv[1] + transform2_inv[2]
  y11 = x1 * transform2_inv[3] + y1 * transform2_inv[4] + transform2_inv[5]
  # Compute flow for augmented image paris & scale for training.
  forward_flow = tf.stack([x11 - x, y11 - y], -1) / FLOW_SCALE_FACTOR
  # print('forward_flow368', np.max(forward_flow.numpy()))

  # Remark invalid flow.
  forward_flow = tf.where(invalid_mask, invalid_flow, forward_flow)
  # print('forward_flow370', np.max(forward_flow.numpy()))

  # Apply a horizontal flip with 50% probability.
  should_flip = tf.less(tf.random_uniform([]), 0.5)
  # pyformat: disable
  images, forward_flow = tf.cond(
      should_flip,
      lambda: image_aug.apply_horizontal_flip(images, forward_flow),
      lambda: (images, forward_flow))
  # pyformat: enable
  # print('380', np.max(forward_flow.numpy()))

  image0, image1, forward_flow = image_aug.random_vertical_flip(
      images[0, :, :, :], images[1, :, :, :], forward_flow,
      augmentation_params.vflip_prob)
  images = tf.stack([image0, image1])
  # print('forward_flow385', np.max(forward_flow.numpy()))

  # Augmentation make flip signs of flow. Flip to the same extreme value
  invalid_mask = tf.abs(forward_flow) > tf.to_float(INVALID_THRESHOLD)
  invalid_flow = tf.ones(forward_flow.get_shape())*INVALID_FLOW_VALUE
  forward_flow = tf.where(invalid_mask, invalid_flow, forward_flow)
  # print('forward_flow390', np.max(forward_flow.numpy()))

  if augmentation_params.disable_ground_truth:
    # Set GT to invalid values for semi-supervised training
    forward_flow = tf.ones(forward_flow.get_shape())*INVALID_FLOW_VALUE

  return images, forward_flow


def _rejection_sample_spatial_aug_parameters(input_image_height,
                                             input_image_width,
                                             augmentation_params,
                                             is_hard_augment):
  """Rejection sample rotation and scaling factors."""
  th = augmentation_params.crop_height
  tw = augmentation_params.crop_width

  def hard_augment():
    spa = SpatialAug(
        [th, tw],
        scale=[augmentation_params.scale1, 0.03, augmentation_params.scale2],
        rot=[augmentation_params.rot1, 0.03],
        trans=[augmentation_params.tran1, 0.03],
        squeeze=[augmentation_params.squeeze1, 0.],
        black=augmentation_params.black)
    return spa(input_image_height, input_image_width)

  def easy_augment():
    spa = SpatialAug([th, tw],
                     trans=[0.4, 0.03],
                     black=augmentation_params.black)
    return spa(input_image_height, input_image_width)

  rotation_degrees, stretch_factor_x, stretch_factor_y, rotation_degrees2, stretch_factor_x2, stretch_factor_y2 = tf.cond(
      is_hard_augment, hard_augment, easy_augment)

  # Schedule parameters for second image.
  s1 = tf.math.log(tf.sqrt(stretch_factor_x * stretch_factor_y + 1e-9) +
                   1e-9)  # scale
  z1 = tf.math.log(tf.sqrt(stretch_factor_x / stretch_factor_y + 1e-9) +
                   1e-9)  # squeeze

  s2 = tf.math.log(
      tf.sqrt(stretch_factor_x2 * stretch_factor_y2 + 1e-9) + 1e-9)  # scale
  z2 = tf.math.log(
      tf.sqrt(stretch_factor_x2 / stretch_factor_y2 + 1e-9) + 1e-9)  # squeeze

  s2 = tf.to_float(s1) + tf.to_float(s2 -
                                     s1) * augmentation_params.schedule_coeff
  z2 = tf.to_float(z1) + tf.to_float(z2 -
                                     z1) * augmentation_params.schedule_coeff

  stretch_factor_x2 = tf.exp(s2 + z2)
  stretch_factor_y2 = tf.exp(s2 - z2)

  return rotation_degrees, stretch_factor_y, stretch_factor_x, rotation_degrees2, stretch_factor_y2, stretch_factor_x2

