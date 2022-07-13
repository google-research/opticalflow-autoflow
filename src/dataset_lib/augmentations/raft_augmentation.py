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

CLIP_MAX = 1e3
DEFAULT_ERASER_BOUNDS = (50, 100)


class Augment(tf.keras.layers.Layer):
  """Augment object for RAFT"""

  def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5):
    super(Augment, self).__init__()
    self.crop_size = crop_size

    self.brightness = (0.6, 1.4)
    self.contrast = (0.6, 1.4)
    self.saturation = (0.6, 1.4)
    self.hue = 0.5 / 3.14

    self.asymmetric_color_aug_prob = 0.2
    self.spatial_aug_prob = 0.8
    self.eraser_aug_prob = 0.5

    self.min_scale = min_scale
    self.max_scale = max_scale
    self.max_stretch = 0.2
    self.stretch_prob = 0.8
    self.margin = 20

  def augment_color(self, images):
    brightness_scale = tf.random.uniform([],
                                         self.brightness[0],
                                         self.brightness[1],
                                         dtype=tf.float32)
    images = images * brightness_scale
    # images = tf.clip_by_value(images, 0, 1) # float limits
    images = tf.image.random_contrast(images, self.contrast[0],
                                      self.contrast[1])
    # images = tf.clip_by_value(images, 0, 1) # float limits
    images = tf.image.random_saturation(images, self.saturation[0],
                                        self.saturation[1])
    # images = tf.clip_by_value(images, 0, 1) # float limits
    images = tf.image.random_hue(images, self.hue)
    images = tf.clip_by_value(images, 0, 1)  # float limits
    return images

  def color_transform(self, img1, img2):
    pred = tf.random.uniform([]) < self.asymmetric_color_aug_prob
    def true_fn(img1, img2):
      img1 = self.augment_color(img1)
      img2 = self.augment_color(img2)
      return [img1, img2]
    def false_fn(img1, img2):
      imgs = tf.concat((img1, img2), axis=0)
      imgs = self.augment_color(imgs)
      return tf.split(imgs, num_or_size_splits=2)

    return tf.cond(pred, lambda: true_fn(img1, img2),
                   lambda: false_fn(img1, img2))

  def eraser_transform(self, img1, img2, bounds=DEFAULT_ERASER_BOUNDS):
    ht, wd, _ = tf.unstack(tf.shape(img1), num=3)
    pred = tf.random.uniform([]) < self.eraser_aug_prob
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

  def random_vertical_flip(self, img1, img2, flow, prob=0.1):
    pred = tf.random.uniform([]) < prob
    def true_fn(img1, img2, flow):
      img1 = tf.image.flip_up_down(img1)
      img2 = tf.image.flip_up_down(img2)
      flow = tf.image.flip_up_down(flow) * [1.0, -1.0]
      return img1, img2, flow
    def false_fn(img1, img2, flow):
      return img1, img2, flow
    return tf.cond(pred,
                   lambda: true_fn(img1, img2, flow),
                   lambda: false_fn(img1, img2, flow))

  def random_horizontal_flip(self, img1, img2, flow, prob=0.5):
    pred = tf.random.uniform([]) < prob
    def true_fn(img1, img2, flow):
      img1 = tf.image.flip_left_right(img1)
      img2 = tf.image.flip_left_right(img2)
      flow = tf.image.flip_left_right(flow) * [-1.0, 1.0]
      return img1, img2, flow
    def false_fn(img1, img2, flow):
      return img1, img2, flow
    return tf.cond(pred,
                   lambda: true_fn(img1, img2, flow),
                   lambda: false_fn(img1, img2, flow))

  def random_scale(self, img1, img2, flow, scale_x, scale_y):
    pred = tf.random.uniform([]) < self.spatial_aug_prob
    ht, wd, _ = tf.unstack(tf.shape(img1), num=3)
    def true_fn(img1, img2, flow, scale_x, scale_y):
      # rescale the images
      new_ht = scale_x * tf.cast(ht, dtype=tf.float32)
      new_wd = scale_y * tf.cast(wd, dtype=tf.float32)
      new_shape = tf.cast(tf.concat([new_ht, new_wd], axis=0), dtype=tf.int32)
      img1 = tf.compat.v1.image.resize(
          img1,
          new_shape,
          tf.compat.v1.image.ResizeMethod.BILINEAR,
          align_corners=True)
      img2 = tf.compat.v1.image.resize(
          img2,
          new_shape,
          tf.compat.v1.image.ResizeMethod.BILINEAR,
          align_corners=True)
      flow = tf.compat.v1.image.resize(
          flow,
          new_shape,
          tf.compat.v1.image.ResizeMethod.BILINEAR,
          align_corners=True)

      flow = flow * tf.expand_dims(
          tf.expand_dims(tf.concat([scale_x, scale_y], axis=0), axis=0), axis=0)
      return img1, img2, flow

    def false_fn(img1, img2, flow):
      return img1, img2, flow
    return tf.cond(pred,
                   lambda: true_fn(img1, img2, flow, scale_x, scale_y),
                   lambda: false_fn(img1, img2, flow))

  def spatial_transform(self, img1, img2, flow):
    # randomly sample scale
    ht, wd, _ = tf.unstack(tf.shape(img1), num=3)
    min_scale = tf.math.maximum(
        (self.crop_size[0] + 1) / ht,
        (self.crop_size[1] + 1) / wd)

    max_scale = self.max_scale
    min_scale = tf.math.maximum(min_scale, self.min_scale)

    scale = 2 ** tf.random.uniform([], self.min_scale, self.max_scale)
    scale_x = scale
    scale_y = scale
    pred = tf.random.uniform([]) < self.stretch_prob
    def true_fn(scale_x, scale_y):
      scale_x *= 2 ** tf.random.uniform([], -self.max_stretch, self.max_stretch)
      scale_y *= 2 ** tf.random.uniform([], -self.max_stretch, self.max_stretch)
      return tf.stack((scale_x, scale_y), axis=0)
    def false_fn(scale_x, scale_y):
      return tf.stack((scale_x, scale_y), axis=0)
    scales = tf.cond(pred,
                     lambda: true_fn(scale_x, scale_y),
                     lambda: false_fn(scale_x, scale_y))
    scale_x, scale_y = tf.split(scales, num_or_size_splits=2)

    clip_max = tf.cast(CLIP_MAX, dtype=tf.float32)
    min_scale = tf.cast(min_scale, dtype=tf.float32)
    scale_x = tf.clip_by_value(scale_x, min_scale, clip_max)
    scale_y = tf.clip_by_value(scale_y, min_scale, clip_max)

    img1, img2, flow = self.random_scale(img1, img2, flow, scale_x, scale_y)

    # random flips
    img1, img2, flow = self.random_horizontal_flip(img1, img2, flow, prob=0.5)
    img1, img2, flow = self.random_vertical_flip(img1, img2, flow, prob=0.1)

    # clip_by_value
    ht, wd, _ = tf.unstack(tf.shape(img1), num=3)
    y0 = tf.random.uniform([],
                           -self.margin,
                           ht - self.crop_size[0] + self.margin,
                           dtype=tf.int32)
    x0 = tf.random.uniform([],
                           -self.margin,
                           wd - self.crop_size[1] + self.margin,
                           dtype=tf.int32)

    y0 = tf.clip_by_value(y0, 0, ht - self.crop_size[0])
    x0 = tf.clip_by_value(x0, 0, wd - self.crop_size[1])

    # crop
    img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]:, :]
    img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]:, :]
    flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]:, :]

    return img1, img2, flow

  def __call__(self, images, flow):
    images = (images + 1) / 2.0  # switch from [-1,1] to [0,1]

    img1, img2 = tf.unstack(images, num=2)
    img1, img2 = self.color_transform(img1, img2)
    img1, img2 = self.eraser_transform(img1, img2)
    img1, img2, flow = self.spatial_transform(img1, img2, flow)
    images = tf.stack((img1, img2), axis=0)
    images = tf.ensure_shape(images,
                             (2, self.crop_size[0], self.crop_size[1], 3))
    flow = tf.ensure_shape(flow, (self.crop_size[0], self.crop_size[1], 2))

    images = 2 * images - 1  # switch from [0,1] to [-1,1]

    return images, flow


def apply(element, aug_params):
  crop_size = (aug_params.crop_height, aug_params.crop_width)
  min_scale = aug_params.min_scale
  max_scale = aug_params.max_scale
  aug = Augment(crop_size=crop_size, min_scale=min_scale, max_scale=max_scale)
  images, flow = aug(element['inputs'], element['label'])
  return {'inputs': images, 'label':flow}
