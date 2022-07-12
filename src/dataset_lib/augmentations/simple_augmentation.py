import tensorflow as tf

from augmentations import crop_aug


def apply_crop(element, aug_params):
  crop_height = aug_params.crop_height
  crop_width = aug_params.crop_width
  return crop_aug.top_left_crop(element, crop_height, crop_width)


def compute_upsample_flow(flow, size):
  upsampled_flow = tf.compat.v1.image.resize(
      flow, size, tf.compat.v1.image.ResizeMethod.BILINEAR, align_corners=True)
  upsampled_x = upsampled_flow[:, :, 0] * tf.cast(
      size[1], dtype=tf.float32) / tf.cast(
          tf.shape(flow)[1], dtype=tf.float32)
  upsampled_y = upsampled_flow[:, :, 1] * tf.cast(
      size[0], dtype=tf.float32) / tf.cast(
          tf.shape(flow)[0], dtype=tf.float32)
  return tf.stack((upsampled_x, upsampled_y), axis=-1)


def apply_resize(element, aug_params):
  _, height, width, _ = tf.unstack(tf.shape(element['inputs']))
  divisor = 64
  adapt_height = tf.to_int32(
      tf.math.ceil(height / divisor) * divisor)
  adapt_width = tf.to_int32(tf.math.ceil(width / divisor) * divisor)

  images = tf.compat.v1.image.resize(
      element['inputs'],
      [adapt_height, adapt_width],
      tf.compat.v1.image.ResizeMethod.BILINEAR,
      align_corners=True)
  flows = compute_upsample_flow(element['label'], (adapt_height, adapt_width))
  return {'inputs': images, 'label': flows}
