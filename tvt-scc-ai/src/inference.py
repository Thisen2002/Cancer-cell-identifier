"""
inference.py

Single-image inference utilities for the TVT vs SCC classifier.

This module is used by both command-line prediction and the Streamlit application.
The returned confidence is a model probability, not a clinical certainty estimate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
from PIL import Image

from dataset import INDEX_TO_LABEL
from evaluate import load_checkpoint
from preprocessing import DEFAULT_IMAGE_SIZE, get_inference_transforms
from train import get_device


class TVTSCCPredictor:
    """Load a checkpoint once and reuse it for multiple image predictions."""

    def __init__(self, checkpoint_path: str | Path) -> None:
        self.device = get_device()
        self.model, self.checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.image_size = tuple(self.checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))
        self.transform = get_inference_transforms(image_size=self.image_size)

    def predict_pil(self, image: Image.Image) -> Dict[str, object]:
        """Predict TVT/SCC probabilities for a PIL image."""
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu()

        predicted_index = int(probabilities.argmax().item())

        return {
            "predicted_index": predicted_index,
            "predicted_label": INDEX_TO_LABEL[predicted_index],
            "confidence": float(probabilities[predicted_index].item()),
            "probabilities": {
                "TVT": float(probabilities[0].item()),
                "SCC": float(probabilities[1].item()),
            },
        }

    def predict_file(self, image_path: str | Path) -> Dict[str, object]:
        """Load an image from disk and return its prediction."""
        with Image.open(image_path) as image:
            return self.predict_pil(image)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict TVT or SCC for one image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = TVTSCCPredictor(args.checkpoint)
    result = predictor.predict_file(args.image)

    print(f"Prediction: {result['predicted_label']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"TVT probability: {result['probabilities']['TVT']:.1%}")
    print(f"SCC probability: {result['probabilities']['SCC']:.1%}")
