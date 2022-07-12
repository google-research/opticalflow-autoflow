import math
import tensorflow as tf

from augmentations import image_aug

def crop_to_box(images, forward_flow, y, x, height, width):
  """Applies a crop to images and forward_flow."""
  images = tf.image.crop_to_bounding_box(images, y, x, height, width)
  forward_flow = tf.image.crop_to_bounding_box(forward_flow, y, x, height,
                                               width)
  return images, forward_flow


def top_left_crop(element, crop_height, crop_width):
  """Crop validation data to be multiples of 64."""
  # crop_height = tf.math.floordiv(element.images.shape[1], 64) * 64
  # crop_width = tf.math.floordiv(element.images.shape[2], 64) * 64

  inputs = element['inputs'][:, 0:crop_height, 0:crop_width, :]
  label = element['label'][0:crop_height, 0:crop_width, :]
  return {'inputs': inputs, 'label': label}


def sample_cropping_centers(image_height, image_width, image_stretch_y,
                            image_stretch_x, rotation_degrees,
                            image_stretch_y2, image_stretch_x2,
                            rotation_degrees2, augmentation_params):
  """Sample crop_centers for two images."""
  rotated_box_height, rotated_box_width = image_aug.rotated_box_size(
      rotation_degrees, augmentation_params)
  rotated_box_height2, rotated_box_width2 = image_aug.rotated_box_size(
      rotation_degrees2, augmentation_params)

  stretched_image_height = image_stretch_y * tf.cast(image_height, tf.float32)
  stretched_image_width = image_stretch_x * tf.cast(image_width, tf.float32)
  stretched_image_height2 = image_stretch_y2 * tf.cast(image_height, tf.float32)
  stretched_image_width2 = image_stretch_x2 * tf.cast(image_width, tf.float32)

  y_min = tf.maximum(rotated_box_height / 2, rotated_box_height2 / 2)
  y_max = tf.minimum(stretched_image_height - rotated_box_height / 2,
                     stretched_image_height2 - rotated_box_height2 / 2)

  x_min = tf.maximum(rotated_box_width / 2, rotated_box_width2 / 2)
  x_max = tf.minimum(stretched_image_width - rotated_box_width / 2,
                     stretched_image_width2 - rotated_box_width2 / 2)

  center_y = tf.random_uniform([], y_min, y_max)
  center_x = tf.random_uniform([], x_min, x_max)

  # Sample crop center of second image conditioned that of first image.
  delta_y = augmentation_params.crop_range_delta * augmentation_params.crop_height * augmentation_params.schedule_coeff
  y2_min = tf.maximum(rotated_box_height2 / 2, center_y - delta_y)
  y2_max = tf.minimum(stretched_image_height2 - rotated_box_height2 / 2,
                      center_y + delta_y)

  delta_x = augmentation_params.crop_range_delta * augmentation_params.crop_width * augmentation_params.schedule_coeff
  x2_min = tf.maximum(rotated_box_width2 / 2, center_x - delta_x)
  x2_max = tf.minimum(stretched_image_width2 - rotated_box_width2 / 2,
                      center_x + delta_x)
  center_y2 = tf.random_uniform([], y2_min, y2_max)
  center_x2 = tf.random_uniform([], x2_min, x2_max)

  return center_y / image_stretch_y, center_x / image_stretch_x, center_y2 / image_stretch_y2, center_x2 / image_stretch_x2


def compose_cropping_transformation(stretch_factor_y, stretch_factor_x,
                                    crop_center_y, crop_center_x,
                                    rotation_degrees, crop_height, crop_width):
  """Composes stretching, rotation, and cropping into one 3x3 transformation."""
  # Transforms coordinates from the output space to a "centered" output space
  # (i.e., relative to the center of the output).
  centering_matrix = tf.stack([(1, 0, -0.5 * (crop_width - 1)),
                               (0, 1, -0.5 * (crop_height - 1)), (0, 0, 1)],
                              axis=0)

  # Performs a rotation by |rotation_degrees|.
  cos_value = tf.math.cos(rotation_degrees * math.pi / 180)
  sin_value = tf.math.sin(rotation_degrees * math.pi / 180)
  rotation_matrix = tf.stack([(cos_value, -sin_value, 0),
                              (sin_value, cos_value, 0), (0, 0, 1)],
                             axis=0)

  # Performs translation to account for the requested cropping location. Note
  # that this translates to the location in the stretched image.
  translation_matrix = tf.stack([(1, 0, stretch_factor_x * crop_center_x),
                                 (0, 1, stretch_factor_y * crop_center_y),
                                 (0, 0, 1)],
                                axis=0)

  # Scales from stretched coordinates to un-stretched coordinates.
  scaling_matrix = tf.stack([(1 / stretch_factor_x, 0, 0),
                             (0, 1 / stretch_factor_y, 0), (0, 0, 1)],
                            axis=0)

  # Compose our various transformations into one overall transformation. The
  # transformation T should be such that T * output_coord = input_coord. The
  # action of our composed transformation on output_coord is equivalent to:
  #  1) Centering (e.g., mapping x=0 to roughly -width/2).
  #  2) Applying the rotation. Due to (1), this rotates about the center rather
  #     than the top-left.
  #  3) Translate to the location of (crop_center_x, crop_center_y) in the
  #     stretched version of the image.
  #  4) Apply inverse stretching to get to original coordinates from |image|.
  transform = centering_matrix
  transform = tf.matmul(rotation_matrix, transform)
  transform = tf.matmul(translation_matrix, transform)
  transform = tf.matmul(scaling_matrix, transform)
  return transform
