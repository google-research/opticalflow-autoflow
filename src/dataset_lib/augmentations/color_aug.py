import numpy as np

import tensorflow as tf


class PCAAug(object):
  """Chromatic Eigen Augmentation, translated from VCN

    https://github.com/gengshan-y/VCN, which translates
    https://github.com/lmb-freiburg/flownet2/blob/master/src/caffe/layers/data_augmentation_layer.cu
  """

  def __init__(self,
               lmult_pow=[0.4, 0, -0.2],
               lmult_mult=[0.4, 0, 0],
               lmult_add=[0.03, 0, 0],
               sat_pow=[0.4, 0, 0],
               sat_mult=[0.5, 0, -0.3],
               sat_add=[0.03, 0, 0],
               col_pow=[0.4, 0, 0],
               col_mult=[0.2, 0, 0],
               col_add=[0.02, 0, 0],
               ladd_pow=[0.4, 0, 0],
               ladd_mult=[0.4, 0, 0],
               ladd_add=[0.04, 0, 0],
               col_rotate=[1., 0, 0],
               schedule_coeff=1):
    # no mean
    self.pow_nomean = [1, 1, 1]
    self.add_nomean = [0, 0, 0]
    self.mult_nomean = [1, 1, 1]
    self.pow_withmean = [1, 1, 1]
    self.add_withmean = [0, 0, 0]
    self.mult_withmean = [1, 1, 1]
    self.lmult_pow = 1
    self.lmult_mult = 1
    self.lmult_add = 0
    self.col_angle = 0
    if ladd_pow is not None:
      self.pow_nomean[0] = tf.exp(
          tf.random.normal([], ladd_pow[2], ladd_pow[0]))
    if col_pow is not None:
      self.pow_nomean[1] = tf.exp(tf.random.normal([], col_pow[2], col_pow[0]))
      self.pow_nomean[2] = tf.exp(tf.random.normal([], col_pow[2], col_pow[0]))

    if ladd_add is not None:
      self.add_nomean[0] = tf.random.normal([], ladd_add[2], ladd_add[0])
    if col_add is not None:
      self.add_nomean[1] = tf.random.normal([], col_add[2], col_add[0])
      self.add_nomean[2] = tf.random.normal([], col_add[2], col_add[0])

    if ladd_mult is not None:
      self.mult_nomean[0] = tf.exp(
          tf.random.normal([], ladd_mult[2], ladd_mult[0]))
    if col_mult is not None:
      self.mult_nomean[1] = tf.exp(
          tf.random.normal([], col_mult[2], col_mult[0]))
      self.mult_nomean[2] = tf.exp(
          tf.random.normal([], col_mult[2], col_mult[0]))

    # with mean
    if sat_pow is not None:
      self.pow_withmean[1] = tf.exp(
          tf.random.uniform([], sat_pow[2] - sat_pow[0],
                            sat_pow[2] + sat_pow[0]))
      self.pow_withmean[2] = self.pow_withmean[1]
    if sat_add is not None:
      self.add_withmean[1] = tf.random.uniform([], sat_add[2] - sat_add[0],
                                               sat_add[2] + sat_add[0])
      self.add_withmean[2] = self.add_withmean[1]
    if sat_mult is not None:
      self.mult_withmean[1] = tf.exp(
          tf.random.uniform([], sat_mult[2] - sat_mult[0],
                            sat_mult[2] + sat_mult[0]))
      self.mult_withmean[2] = self.mult_withmean[1]

    if lmult_pow is not None:
      self.lmult_pow = tf.exp(
          tf.random.uniform([], lmult_pow[2] - lmult_pow[0],
                            lmult_pow[2] + lmult_pow[0]))
    if lmult_mult is not None:
      self.lmult_mult = tf.exp(
          tf.random.uniform([], lmult_mult[2] - lmult_mult[0],
                            lmult_mult[2] + lmult_mult[0]))
    if lmult_add is not None:
      self.lmult_add = tf.random.uniform([], lmult_add[2] - lmult_add[0],
                                         lmult_add[2] + lmult_add[0])
    if col_rotate is not None:
      self.col_angle = tf.random.uniform([], col_rotate[2] - col_rotate[0],
                                         col_rotate[2] + col_rotate[0])

    # eigen vectors
    self.eigvec = tf.transpose(
        tf.reshape([0.51, 0.56, 0.65, 0.79, 0.01, -0.62, 0.35, -0.83, 0.44],
                   [3, 3]))

  def __call__(self, inputs, target):
    inputs = tf.stack([
        self.pca_image(inputs[0, :, :, :]),
        self.pca_image(inputs[1, :, :, :])
    ])
    return inputs, target

  def _apply_eig_nomean(self, eig, c, max_abs_eig):
    """tf.cond true_fn for eig_nomean."""
    eig_result = eig[:, :, c] / max_abs_eig[c]
    a = tf.pow(tf.abs(eig_result), self.pow_nomean[c])
    b = (tf.to_float(eig_result > 0) - 0.5) * 2
    eig_result = tf.multiply(a, b)
    eig_result = eig_result + self.add_nomean[c]
    eig_result = eig_result * self.mult_nomean[c]
    return eig_result

  def _apply_eig_withmean(self, eig, max_abs_eig):
    """tf.cond true_fn."""
    a = tf.pow(tf.abs(eig[:, :, 0]), self.pow_withmean[0])
    b = (tf.to_float(eig[:, :, 0] > 0) - 0.5) * 2
    eig0 = tf.multiply(a, b)
    eig0 += self.add_withmean[0]
    eig0 *= self.mult_withmean[0]
    return eig0

  def _apply_color_angle(self, eig):
    """tf.cond true_fn."""
    temp1 = tf.math.cos(self.col_angle) * eig[:, :, 1] - tf.math.sin(
        self.col_angle) * eig[:, :, 2]
    temp2 = tf.math.sin(self.col_angle) * eig[:, :, 1] + tf.math.cos(
        self.col_angle) * eig[:, :, 2]
    return tf.stack([eig[:, :, 0], temp1, temp2], -1)

  def _apply_final_step(self, eig, l1, max_abs_eig, max_l):
    """tf.cond true_fn."""
    l = tf.sqrt(eig[:, :, 0] * eig[:, :, 0] + eig[:, :, 1] * eig[:, :, 1] +
                eig[:, :, 2] * eig[:, :, 2] + 1e-9)
    l1 = tf.pow(l1, self.lmult_pow)
    l1 = tf.clip_by_value(l1 + self.lmult_add, 0, np.inf)
    l1 = l1 * self.lmult_mult
    l1 = l1 * max_l
    lmask = tf.to_float(l > 1e-2)
    eig = eig * tf.expand_dims((1 - lmask), -1) + tf.multiply(
        tf.divide(eig, tf.expand_dims(l, -1)), tf.expand_dims(
            l1, -1)) * tf.expand_dims(lmask, -1)
    eig_list = []
    for c in range(3):
      tmp = eig[:, :, c] * (1 - lmask) + tf.clip_by_value(
          eig[:, :, c], -np.inf, max_abs_eig[c]) * lmask
      eig_list.append(tmp)
    eig = tf.stack(eig_list, -1)
    return eig

  def pca_image(self, rgb):
    eig = tf.matmul(rgb, self.eigvec)

    eig = tf.matmul(rgb, self.eigvec)
    mean_rgb = tf.reduce_mean(rgb, [0, 1])

    max_abs_eig = tf.reduce_max(tf.abs(eig), [0, 1])
    max_l = tf.norm(max_abs_eig)
    mean_eig = tf.linalg.matvec(self.eigvec, mean_rgb, transpose_a=True)
    # no-mean stuff
    eig -= tf.expand_dims(tf.expand_dims(mean_eig, 0), 0)

    mean_eig_list = []
    eig_list = []
    for c in range(3):
      is_apply = tf.greater(max_abs_eig[c], 1e-2)
      mean_eig0 = tf.cond(is_apply, lambda: mean_eig[c] / max_abs_eig[c],
                          lambda: mean_eig[c])
      eig0 = tf.cond(is_apply,
                     lambda: self._apply_eig_nomean(eig, c, max_abs_eig),
                     lambda: eig[:, :, c])
      mean_eig_list.append(mean_eig0)
      eig_list.append(eig0)
    mean_eig = tf.stack(mean_eig_list)
    eig = tf.stack(eig_list, -1)

    eig += tf.expand_dims(tf.expand_dims(mean_eig, 0), 0)  # match:-)

    # withmean stuff
    is_apply = tf.greater(max_abs_eig[0], 1e-2)
    eig0 = tf.cond(is_apply, lambda: self._apply_eig_withmean(eig, max_abs_eig),
                   lambda: eig[:, :, 0])
    eig = tf.stack([eig0, eig[:, :, 1], eig[:, :, 2]], -1)

    s = tf.sqrt(eig[:, :, 1] * eig[:, :, 1] + eig[:, :, 2] * eig[:, :, 2] +
                1e-9)
    smask = tf.to_float(s > 1e-2)
    s1 = tf.pow(s, self.pow_withmean[1])
    s1 = tf.clip_by_value(s1 + self.add_withmean[1], 0, np.inf)
    s1 = s1 * self.mult_withmean[1]
    s1 = s1 * smask + s * (1 - smask)

    # color angle
    is_apply = tf.math.not_equal(self.col_angle, 0)
    eig = tf.cond(is_apply, lambda: self._apply_color_angle(eig), lambda: eig)

    # to origin magnitude
    eig_list = []
    for c in range(3):
      is_apply = tf.greater(max_abs_eig[c], 1e-2)
      tmp = tf.cond(is_apply, lambda: eig[:, :, c] * max_abs_eig[c],
                    lambda: eig[:, :, c])
      eig_list.append(tmp)

    is_apply = tf.greater(max_l, 1e-2)
    tmp = tf.to_float(
        tf.sqrt(
            tf.multiply(eig_list[0], eig_list[0]) +
            tf.multiply(eig_list[1], eig_list[1]) +
            tf.multiply(eig_list[2], eig_list[2])) / max_l + 1e-9)
    l1 = tf.cond(is_apply, lambda: tmp, lambda: tmp * 0)

    eig_list[1] = eig_list[1] * (1 - smask) + eig_list[1] / s * s1 * smask
    eig_list[2] = eig_list[2] * (1 - smask) + eig_list[2] / s * s1 * smask
    eig = tf.stack(eig_list, -1)

    is_apply = tf.greater(max_l, 1e-2)
    eig = tf.cond(is_apply,
                  lambda: self._apply_final_step(eig, l1, max_abs_eig, max_l),
                  lambda: eig)
    eig = tf.clip_by_value(tf.matmul(eig, tf.transpose(self.eigvec)), 0, 1)
    return eig


def apply(images, aug_params):
  """Augments the input images by applying random color transformations.

  Args:
    images: A tensor of size [2, height, width, 3] representing the two RGB
      input images with range [-1, 1].
    aug_params: An instance of AugmentationParams to control the color
      transformations.

  Returns:
    output: A tensor of size [2, height, width, 3] holding the augmented image.
  """
  # PCA agumentation
  # Convert to [0,1] from [-1,1]
  images = (images + 1.)/2.
  pcaaug = PCAAug(
      lmult_pow=[0.4 * aug_params.lmult_factor, 0, -0.2],
      lmult_mult=[0.4 * aug_params.lmult_factor, 0, 0],
      lmult_add=[0.03 * aug_params.lmult_factor, 0, 0],
      sat_pow=[0.4 * aug_params.sat_factor, 0, 0],
      sat_mult=[0.5 * aug_params.sat_factor, 0, -0.3],
      sat_add=[0.03 * aug_params.sat_factor, 0, 0],
      col_pow=[0.4 * aug_params.col_factor, 0, 0],
      col_mult=[0.2 * aug_params.col_factor, 0, 0],
      col_add=[0.02 * aug_params.col_factor, 0, 0],
      ladd_pow=[
          0.4 * aug_params.ladd_factor,
          0,
          0,
      ],
      ladd_mult=[0.4 * aug_params.ladd_factor, 0, 0],
      ladd_add=[0.04 * aug_params.ladd_factor, 0, 0],
      col_rotate=[1. * aug_params.col_rot_factor, 0, 0],
      schedule_coeff=1)
  images, _ = pcaaug(images, [])

  # Chromatic augmentation applied to image 2
  # TODO(deqingsun): make the range parameters of each augmentation CONSTANT
  image1 = images[1, :, :, :]
  mean_in = tf.reduce_sum(image1, -1)
  color = tf.math.exp(
      tf.random.normal([3], 0., 0.02 * aug_params.schedule_coeff))
  image1 = image1 * color
  brightness_coeff = tf.divide(mean_in, tf.reduce_sum(image1, -1) + 0.01)
  image1 = tf.math.multiply(image1, tf.expand_dims(brightness_coeff, -1))
  image1 = tf.clip_by_value(image1, 0., 1.)
  # Gamma
  gamma = tf.exp(
      tf.random.normal([], 0., 0.02 * aug_params.schedule_coeff))
  image1 = tf.pow(image1, gamma)
  # Brightness
  image1 += tf.random.normal([], 0, 0.02 * aug_params.schedule_coeff)
  # Contrast
  image1 = 0.5 + (image1 - 0.5) * tf.exp(
      tf.random.normal([], 0, 0.02 * aug_params.schedule_coeff))
  image1 = tf.clip_by_value(image1, 0., 1.)
  images = tf.stack([images[0, :, :, :], image1])

  # Add noise
  noise_std = tf.random.uniform([], 0., aug_params.noise_std_range)
  noise = tf.random.normal(tf.shape(images), 0, noise_std)
  images += noise

  # Clip to [0,1] & scale to [-1,1]
  images = tf.clip_by_value(images, 0., 1.) * 2. - 1

  if aug_params.is_channel_swapping:
    channel_permutation = tf.constant([[0, 1, 2], [0, 2, 1], [1, 0, 2],
                                       [1, 2, 0], [2, 0, 1], [2, 1, 0]])
    rand_i = tf.random_uniform([], minval=0, maxval=6, dtype=tf.int32)
    perm = channel_permutation[rand_i]
    images = tf.stack([
        images[:, :, :, perm[0]], images[:, :, :, perm[1]], images[:, :, :,
                                                                   perm[2]]
    ], axis=-1)

  if aug_params.is_random_erasing:
    image0, image1 = eraser_transform(images[0, :, :, :], images[1, :, :, :],
                                      bounds=[50, 100], eraser_aug_prob=0.5)
    images = tf.stack([image0, image1])
    # images = tf.stack([images[0, :, :, :], random_erasing(images[1, :, :, :])])

  return images

def eraser_transform(img1, img2, bounds, eraser_aug_prob=0.5):
  ht, wd, _ = tf.unstack(tf.shape(img1))
  pred = tf.random.uniform([]) < eraser_aug_prob
  def true_fn(img1, img2):
    mean_color = tf.reduce_mean(tf.reshape(img2, (-1, 3)), axis=0)
    mean_color = tf.expand_dims(tf.expand_dims(mean_color, axis=0), axis=0)
    def body(var_img, mean_color):
      x0 = tf.random.uniform([], 0, wd, dtype=tf.int32)
      y0 = tf.random.uniform([], 0, ht, dtype=tf.int32)
      dx = tf.random.uniform([], bounds[0], bounds[1], dtype=tf.int32)
      dy = tf.random.uniform([], bounds[0], bounds[1], dtype=tf.int32)
      x = tf.range(wd)
      x_mask = (x0 <= x) & (x < x0+dx)
      y = tf.range(ht)
      y_mask = (y0 <= y) & (y < y0+dy)
      mask = x_mask & y_mask[:, tf.newaxis]
      mask = tf.cast(mask[:, :, tf.newaxis], img1.dtype)
      mean_slice = tf.tile(mean_color, multiples=[ht, wd, 1])
      result = var_img * (1 - mask) + mean_slice * mask
      return result
    max_num = tf.random.uniform([], 1, 3, dtype=tf.int32)
    img2 = body(img2, mean_color)
    img2 = tf.cond(2 <= max_num, lambda: body(img2, mean_color), lambda: img2)
    return img1, img2
  def false_fn(img1, img2):
    return img1, img2

  return tf.cond(pred, lambda: true_fn(img1, img2),
                 lambda: false_fn(img1, img2))
