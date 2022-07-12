load("//learning/brain/frameworks/blaze:py3.bzl", "py3_library")

py3_library(
    name = "augmentations",
    srcs = [
        "aug_params.py",
        "augmentations.py",
        "color_aug.py",
        "crop_aug.py",
        "image_aug.py",
        "pwc_augmentation.py",
        "raft_augmentation.py",
        "simple_augmentation.py",
        "spatial_aug.py",
    ],
    deps = [
        "//third_party/py/absl/logging",
        "//third_party/py/jax",
        "//third_party/py/ml_collections",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow:tensorflow_google",
        "//third_party/tensorflow/contrib/image:image_py",
    ],
)
