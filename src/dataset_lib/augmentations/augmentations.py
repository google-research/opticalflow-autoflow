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
