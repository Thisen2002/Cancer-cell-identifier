"""
evaluate.py

Evaluation utilities for the TVT vs SCC classifier.

Two levels are reported:
1. Image-level performance.
2. Case-level performance, where probabilities from all images of the same dog/case
   are averaged before assigning a final class.

Case-level results are especially important because the independent biological unit
in this project is the dog/case, not each microscope image.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from dataset import INDEX_TO_LABEL, get_split_dataframe, load_metadata, validate_no_case_leakage, CytologyDataset
from preprocessing import get_eval_transforms
from train import build_model, get_device


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    """Load a saved model checkpoint and reconstruct the baseline architecture."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model_name = checkpoint.get("model_name", "resnet18")
    if model_name != "resnet18":
        raise ValueError(f"Unsupported model_name in checkpoint: {model_name}")

    model = build_model(pretrained=False, num_classes=checkpoint.get("num_classes", 2))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def collect_predictions(model, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    """Run inference over a DataLoader and return one result row per image."""
    rows = []

    with torch.no_grad():
        for images, labels, metadata in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)

            # PyTorch's default collate function converts the metadata dictionary
            # into a dictionary of lists, one item per sample in the batch.
            for i in range(len(labels)):
                rows.append(
                    {
                        "case_id": metadata["case_id"][i],
                        "image_id": metadata["image_id"][i],
                        "true_label": int(labels[i]),
                        "predicted_label": int(predictions[i]),
                        "prob_tvt": float(probabilities[i, 0]),
                        "prob_scc": float(probabilities[i, 1]),
                    }
                )

    return pd.DataFrame(rows)


def aggregate_to_case_level(image_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Average image probabilities for each independent case.

    This is a simple baseline aggregation strategy. More advanced approaches can be
    investigated later, but the test-set aggregation rule should be defined before
    examining final test performance.
    """
    # Verify that all images from one case share the same ground-truth diagnosis.
    inconsistent = image_predictions.groupby("case_id")["true_label"].nunique()
    if (inconsistent > 1).any():
        bad = inconsistent[inconsistent > 1].index.tolist()
        raise ValueError(f"Cases with conflicting true labels: {bad}")

    case_df = (
        image_predictions.groupby("case_id", as_index=False)
        .agg(
            true_label=("true_label", "first"),
            prob_tvt=("prob_tvt", "mean"),
            prob_scc=("prob_scc", "mean"),
            n_images=("image_id", "count"),
        )
    )

    case_df["predicted_label"] = (case_df["prob_scc"] >= 0.5).astype(int)
    return case_df


def calculate_metrics(df: pd.DataFrame) -> Dict[str, float | None]:
    """
    Calculate binary classification metrics with SCC (class 1) as positive class.

    In the paper, clearly state which class is treated as positive when reporting
    sensitivity/recall and specificity.
    """
    y_true = df["true_label"].to_numpy()
    y_pred = df["predicted_label"].to_numpy()
    y_score = df["prob_scc"].to_numpy()

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else None

    metrics: Dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_scc": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "sensitivity_recall_scc": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity_scc": float(specificity) if specificity is not None else None,
        "f1_scc": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }

    # ROC-AUC is undefined when the evaluated set contains only one class.
    metrics["roc_auc_scc"] = (
        float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else None
    )

    return metrics


def save_confusion_matrix(df: pd.DataFrame, output_path: str | Path, title: str) -> None:
    """Save a simple confusion-matrix figure without modifying the prediction data."""
    cm = confusion_matrix(df["true_label"], df["predicted_label"], labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=[INDEX_TO_LABEL[0], INDEX_TO_LABEL[1]],
        yticklabels=[INDEX_TO_LABEL[0], INDEX_TO_LABEL[1]],
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    metadata_csv: str | Path,
    project_root: str | Path,
    output_dir: str | Path,
    split: str = "test",
    batch_size: int = 16,
    num_workers: int = 0,
) -> Dict[str, Dict]:
    """Evaluate a trained checkpoint and save metrics/predictions to disk."""
    device = get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(metadata_csv)
    if "split" not in df.columns or df["split"].isna().any():
        raise ValueError(
            "Evaluation requires fixed split assignments in cases.csv. "
            "Use the dataset_splits.csv produced during training or copy its split "
            "column into your metadata."
        )

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    validate_no_case_leakage(df)
    split_df = get_split_dataframe(df, split)

    dataset = CytologyDataset(
        split_df,
        transform=get_eval_transforms(),
        project_root=project_root,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model, _checkpoint = load_checkpoint(checkpoint_path, device)
    image_predictions = collect_predictions(model, loader, device)
    case_predictions = aggregate_to_case_level(image_predictions)

    image_metrics = calculate_metrics(image_predictions)
    case_metrics = calculate_metrics(case_predictions)

    image_predictions.to_csv(output_dir / f"{split}_image_predictions.csv", index=False)
    case_predictions.to_csv(output_dir / f"{split}_case_predictions.csv", index=False)

    save_confusion_matrix(
        image_predictions,
        output_dir / f"{split}_image_confusion_matrix.png",
        f"{split.title()} — image-level",
    )
    save_confusion_matrix(
        case_predictions,
        output_dir / f"{split}_case_confusion_matrix.png",
        f"{split.title()} — case-level",
    )

    results = {
        "image_level": image_metrics,
        "case_level": case_metrics,
        "n_images": int(len(image_predictions)),
        "n_cases": int(len(case_predictions)),
    }

    with open(output_dir / f"{split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TVT vs SCC checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata", default="metadata/cases.csv")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/metrics/baseline")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        metadata_csv=args.metadata,
        project_root=args.project_root,
        output_dir=args.output_dir,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(json.dumps(results, indent=2))
