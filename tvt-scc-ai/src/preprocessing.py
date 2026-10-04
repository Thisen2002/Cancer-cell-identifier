"""Image preprocessing and normalization routines."""

"""
preprocessing.py

Image preprocessing and augmentation for the Canine TVT vs SCC project.

Important rule
--------------
Raw images are never overwritten. These transformations operate in memory when
images are loaded by PyTorch.

Training pipeline:
    RGB -> resize -> conservative augmentation -> tensor -> normalization

Validation/test/inference pipeline:
    RGB -> resize -> tensor -> normalization
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from PIL import Image
from torchvision import transforms


# 224x224 is widely used by pretrained CNNs such as ResNet.
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)

# Standard ImageNet normalization. This is appropriate when fine-tuning
# torchvision models pretrained on ImageNet.
IMAGENET_MEAN: Sequence[float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Sequence[float] = (0.229, 0.224, 0.225)


def ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert any PIL image mode to standard three-channel RGB."""
    return image.convert("RGB")


def get_train_transforms(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    """
    Build preprocessing and conservative augmentation for training images.

    The augmentations are deliberately mild because cytological morphology is
    diagnostically important. Strong distortions could create unrealistic cells.
    """
    return transforms.Compose(
        [
            transforms.Lambda(ensure_rgb),

            # Standardize input dimensions without touching the original file.
            transforms.Resize(image_size),

            # Cytology fields usually have no meaningful left/right orientation.
            transforms.RandomHorizontalFlip(p=0.5),

            # Top/bottom orientation is also generally arbitrary in microscopy.
            transforms.RandomVerticalFlip(p=0.5),

            # Small rotation improves robustness while limiting distortion.
            transforms.RandomRotation(degrees=10),

            # Mild brightness/contrast variation helps reduce dependence on small
            # microscope/camera differences. Keep these changes conservative.
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
            ),

            # Convert PIL image [0,255] values to a float tensor in [0,1].
            transforms.ToTensor(),

            # Match the preprocessing expected by ImageNet-pretrained CNNs.
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_eval_transforms(
    image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> transforms.Compose:
    """
    Build deterministic preprocessing for validation and test images.

    No random augmentation is used during evaluation. The same image should always
    result in the same model input so that performance measurements are reproducible.
    """
    return transforms.Compose(
        [
            transforms.Lambda(ensure_rgb),