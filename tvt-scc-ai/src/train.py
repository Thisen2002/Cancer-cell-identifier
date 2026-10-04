"""
train.py

Training entry point for the Canine TVT vs SCC cytology classifier.

This module is intentionally usable before the real dataset arrives: once a valid
metadata/cases.csv and images exist, the same training code can be used without
changing the core pipeline.

Default model
-------------
ResNet-18 pretrained on ImageNet is used as a simple, reproducible transfer-learning
baseline. A baseline should be kept deliberately simple so that later experiments
can be compared against it fairly.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from dataset import LABEL_TO_INDEX, prepare_datasets
from preprocessing import DEFAULT_IMAGE_SIZE, get_eval_transforms, get_train_transforms


def set_seed(seed: int = 42) -> None:
    """Set common random seeds to make experiments more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Use CUDA when available; otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(pretrained: bool = True, num_classes: int = 2) -> nn.Module:
    """
    Build the baseline ResNet-18 classifier.

    The original ImageNet output layer predicts 1000 classes. We replace it with a
    two-class layer for TVT and SCC.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> Tuple[float, float]:
    """
    Run one training or validation epoch.

    When optimizer is supplied, gradients are calculated and weights are updated.
    When optimizer is None, the function behaves as validation/evaluation.
    """
    is_training = optimizer is not None
    model.train(mode=is_training)

    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, leave=False, desc="train" if is_training else "val")

    for images, labels, _metadata in progress:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        # Gradients are only needed during training.
        with torch.set_grad_enabled(is_training):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

        progress.set_postfix(loss=f"{loss.item():.4f}")

    if total == 0:
        raise ValueError("DataLoader contained zero images.")

    return running_loss / total, correct / total


def train_model(
    metadata_csv: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    num_workers: int = 0,
    seed: int = 42,
    pretrained: bool = True,
) -> Path:
    """Train the baseline model and save the best validation checkpoint."""
    set_seed(seed)
    device = get_device()

    project_root = Path(project_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, _test_dataset, split_metadata = prepare_datasets(
        metadata_csv=metadata_csv,
        train_transform=get_train_transforms(),
        eval_transform=get_eval_transforms(),
        project_root=project_root,
        random_state=seed,
    )

    # Persist the exact split assignments used for this experiment.
    split_metadata.to_csv(output_dir / "dataset_splits.csv", index=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    history: List[Dict[str, float]] = []
    checkpoint_path = output_dir / "best_model.pt"

    print(f"Device: {device}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.3f}"
        )

        # Choose the checkpoint using validation loss only.
        # The held-out test set must not guide model selection.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "resnet18",
                    "num_classes": 2,
                    "class_to_idx": LABEL_TO_INDEX,
                    "image_size": list(DEFAULT_IMAGE_SIZE),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                },
                checkpoint_path,
            )

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Best model saved to: {checkpoint_path}")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TVT vs SCC baseline classifier")
    parser.add_argument("--metadata", default="metadata/cases.csv")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="models/baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not use ImageNet pretrained weights.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(
        metadata_csv=args.metadata,
        project_root=args.project_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained=not args.no_pretrained,
    )
