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
    conflicting_cases = diagnoses_per_case[diagnoses_per_case > 1].index.tolist()
    if conflicting_cases:
        raise ValueError(
            "The following case_id values have more than one diagnosis: "
            + ", ".join(conflicting_cases[:10])
        )

    return df.reset_index(drop=True)


def create_case_level_splits(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Assign train/validation/test splits at CASE LEVEL.

    Splitting is stratified by diagnosis. One row per case is split first, and then
    that case assignment is mapped back to all images belonging to the case.
    """
    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"train_size + val_size + test_size must equal 1.0, got {total:.4f}"
        )

    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("All split fractions must be greater than zero.")

    # Reduce the metadata to one row per independent dog/case.
    case_df = (
        df[["case_id", "diagnosis"]]
        .drop_duplicates(subset=["case_id"])
        .reset_index(drop=True)
    )

    # First split: training cases vs the remaining validation+test cases.
    train_cases, remaining_cases = train_test_split(
        case_df,
        train_size=train_size,
        random_state=random_state,
        stratify=case_df["diagnosis"],
    )

    # Second split: divide the remaining cases into validation and test sets.
    remaining_fraction = val_size + test_size
    relative_val_size = val_size / remaining_fraction

    val_cases, test_cases = train_test_split(
        remaining_cases,
        train_size=relative_val_size,
        random_state=random_state,
        stratify=remaining_cases["diagnosis"],
    )

    split_map = {}
    split_map.update({case_id: "train" for case_id in train_cases["case_id"]})
    split_map.update({case_id: "val" for case_id in val_cases["case_id"]})
    split_map.update({case_id: "test" for case_id in test_cases["case_id"]})

    result = df.copy()
    result["split"] = result["case_id"].map(split_map)

    validate_no_case_leakage(result)
    return result


def validate_no_case_leakage(df: pd.DataFrame) -> None:
    """Raise an error if any case appears in more than one dataset split."""
    if "split" not in df.columns:
        raise ValueError("Metadata does not contain a 'split' column.")

    valid_splits = {"train", "val", "test"}
    unknown_splits = set(df["split"].dropna()) - valid_splits
    if unknown_splits:
        raise ValueError(f"Unknown split value(s): {sorted(unknown_splits)}")

    split_counts = df.groupby("case_id")["split"].nunique()
    leaked_cases = split_counts[split_counts > 1].index.tolist()

    if leaked_cases:
        raise ValueError(
            "DATA LEAKAGE DETECTED. These cases occur in multiple splits: "
            + ", ".join(leaked_cases[:20])
        )


def get_split_dataframe(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Return metadata rows for one split: train, val, or test."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be one of: 'train', 'val', 'test'")

    if "split" not in df.columns:
        raise ValueError(
            "Metadata has no 'split' column. Run create_case_level_splits() first."
        )

    return df[df["split"] == split].reset_index(drop=True)


class CytologyDataset(Dataset):
    """PyTorch Dataset for TVT/SCC cytology images."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
        project_root: Optional[str | Path] = None,
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.project_root = Path(project_root) if project_root else None

        if self.df.empty:
            raise ValueError("CytologyDataset received an empty DataFrame.")

        missing = REQUIRED_COLUMNS - set(self.df.columns)
        if missing:
            raise ValueError(
                "Dataset DataFrame is missing required columns: "
                + ", ".join(sorted(missing))
            )

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, image_path: str) -> Path:
        """Resolve relative image paths against project_root when provided."""
        path = Path(image_path)
        if not path.is_absolute() and self.project_root is not None:
            path = self.project_root / path
        return path

    def __getitem__(self, index: int) -> Tuple[object, int, Dict[str, str]]:
        row = self.df.iloc[index]
        image_path = self._resolve_image_path(row["image_path"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found for image_id={row['image_id']}: {image_path}"
            )

        # Force 3-channel RGB input even if the original file is grayscale/RGBA.
        with Image.open(image_path) as pil_image:
            image = pil_image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        diagnosis = row["diagnosis"]
        label = LABEL_TO_INDEX[diagnosis]

        # Return identifiers too; these are useful for case-level evaluation later.
        metadata = {
            "case_id": str(row["case_id"]),
            "image_id": str(row["image_id"]),
            "diagnosis": diagnosis,
            "image_path": str(image_path),
        }

        return image, label, metadata


def prepare_datasets(
    metadata_csv: str | Path,
    train_transform: Optional[Callable],
    eval_transform: Optional[Callable],
    project_root: Optional[str | Path] = None,
    random_state: int = 42,
) -> Tuple[CytologyDataset, CytologyDataset, CytologyDataset, pd.DataFrame]:
    """
    Convenience helper that loads metadata, creates/validates case-level splits,
    and returns PyTorch train, validation, and test datasets.
    """
    df = load_metadata(metadata_csv)

    # Respect a complete split column if one already exists; otherwise generate one.
    if "split" in df.columns and df["split"].notna().all():
        df["split"] = df["split"].astype(str).str.strip().str.lower()
        validate_no_case_leakage(df)
    else:
        df = create_case_level_splits(df, random_state=random_state)

    train_df = get_split_dataframe(df, "train")
    val_df = get_split_dataframe(df, "val")
    test_df = get_split_dataframe(df, "test")

    train_dataset = CytologyDataset(
        train_df, transform=train_transform, project_root=project_root
    )
    val_dataset = CytologyDataset(
        val_df, transform=eval_transform, project_root=project_root
    )
    test_dataset = CytologyDataset(
        test_df, transform=eval_transform, project_root=project_root
    )

    return train_dataset, val_dataset, test_dataset, df
