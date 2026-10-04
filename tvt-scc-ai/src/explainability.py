"""
explainability.py

Grad-CAM utilities for visualizing image regions that influenced the classifier.

Important interpretation note
-----------------------------
Grad-CAM is an explainability aid, not proof that the network learned medically
correct features. Heatmaps should be interpreted with veterinary/pathology expertise.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from evaluate import load_checkpoint
from preprocessing import DEFAULT_IMAGE_SIZE, get_inference_transforms
from train import get_device


def prepare_original_for_overlay(image: Image.Image, image_size: Tuple[int, int]) -> np.ndarray:
    """Resize an RGB image and convert it to floating-point [0,1] for Grad-CAM overlay."""
    image = image.convert("RGB").resize((image_size[1], image_size[0]))
    return np.asarray(image, dtype=np.float32) / 255.0


def generate_gradcam(
    checkpoint_path: str | Path,
    image_path: str | Path,
    output_path: str | Path,
    target_class: int | None = None,
) -> Tuple[int, float, Path]:
    """
    Generate and save a Grad-CAM overlay for one image.

    Parameters
    ----------
    target_class:
        0 for TVT, 1 for SCC, or None to explain the model's predicted class.
    """
    device = get_device()
    model, checkpoint = load_checkpoint(checkpoint_path, device)

    image_size = tuple(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
    transform = get_inference_transforms(image_size=image_size)

    with Image.open(image_path) as pil_image:
        rgb_image = pil_image.convert("RGB")
        input_tensor = transform(rgb_image).unsqueeze(0).to(device)
        overlay_base = prepare_original_for_overlay(rgb_image, image_size)

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = int(probabilities.argmax().item())
        predicted_confidence = float(probabilities[predicted_class].item())

    class_to_explain = predicted_class if target_class is None else int(target_class)
    if class_to_explain not in (0, 1):
        raise ValueError("target_class must be 0 (TVT), 1 (SCC), or None")

    # For torchvision ResNet-18, layer4[-1] is a common final convolutional target.
    target_layers = [model.layer4[-1]]

    # Import target lazily to keep the Grad-CAM-specific dependency isolated here.
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    targets = [ClassifierOutputTarget(class_to_explain)]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    visualization = show_cam_on_image(
        overlay_base,
        grayscale_cam,
        use_rgb=True,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(visualization).save(output_path)

    return predicted_class, predicted_confidence, output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for one cytology image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="results/gradcam/gradcam.png")
    parser.add_argument("--target-class", type=int, choices=[0, 1], default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predicted_class, confidence, output_path = generate_gradcam(
        checkpoint_path=args.checkpoint,
        image_path=args.image,
        output_path=args.output,
        target_class=args.target_class,
    )
    label = "TVT" if predicted_class == 0 else "SCC"
    print(f"Prediction: {label} ({confidence:.1%})")
    print(f"Grad-CAM saved to: {output_path}")
