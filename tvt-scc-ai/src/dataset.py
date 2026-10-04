"""Dataset loading and augmentation pipelines."""

"""
dataset.py

Dataset utilities for the Canine TVT vs SCC cytology image classification project.

Responsibilities
----------------
1. Read metadata from metadata/cases.csv.
2. Validate required columns and diagnostic labels.
3. Create train/validation/test splits at CASE LEVEL.
4. Prevent data leakage: one dog/case must never appear in multiple splits.
5. Provide a PyTorch Dataset class for loading cytology images.

Research rule
-------------
The independent biological unit is the dog/case, not the image. A case may contain
many microscope fields, but every image from that case must remain in the same split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


VALID_LABELS = {"TVT", "SCC"}

LABEL_TO_INDEX: Dict[str, int] = {
    "TVT": 0,
    "SCC": 1,
}

INDEX_TO_LABEL: Dict[int, str] = {
    value: key for key, value in LABEL_TO_INDEX.items()
}

REQUIRED_COLUMNS = {
    "case_id",
    "diagnosis",
    "image_id",
    "image_path",
}


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate the dataset metadata CSV."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(how="all").copy()

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Metadata is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    required_non_null = ["case_id", "diagnosis", "image_id", "image_path"]
    if df[required_non_null].isnull().any().any():
        bad_rows = df[df[required_non_null].isnull().any(axis=1)].index.tolist()
        raise ValueError(f"Missing required metadata values in row(s): {bad_rows}")

    # Normalize text fields so entries such as 'tvt' and ' TVT ' are treated identically.
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip().str.upper()
    df["case_id"] = df["case_id"].astype(str).str.strip()
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["image_path"] = df["image_path"].astype(str).str.strip()

    invalid_labels = sorted(set(df["diagnosis"]) - VALID_LABELS)
    if invalid_labels:
        raise ValueError(
            f"Unsupported diagnosis label(s): {invalid_labels}. "
            f"Expected only: {sorted(VALID_LABELS)}"
        )

    # image_id should identify exactly one image.
    duplicate_image_ids = df.loc[
        df["image_id"].duplicated(keep=False), "image_id"
    ].unique()
    if len(duplicate_image_ids) > 0:
        raise ValueError(
            "Duplicate image_id value(s) found: "
            + ", ".join(map(str, duplicate_image_ids[:10]))
        )

    # One case must have one diagnosis only.
    diagnoses_per_case = df.groupby("case_id")["diagnosis"].nunique()