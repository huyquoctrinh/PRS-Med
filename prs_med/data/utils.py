from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from sklearn.utils import shuffle


def load_annotation(annotation_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate annotation CSV files from a directory."""
    annotation_path = Path(annotation_dir)
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    csv_files = sorted(
        path for path in annotation_path.iterdir() if path.suffix.lower() == ".csv"
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in annotation directory: {annotation_dir}")

    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file).dropna().reset_index(drop=True)
        frames.append(df)

    annotations = pd.concat(frames, ignore_index=True)
    annotations = shuffle(annotations).reset_index(drop=True)

    train_df = annotations[annotations["split"] == "train"].reset_index(drop=True)
    test_df = annotations[annotations["split"] == "test"].reset_index(drop=True)
    return train_df, test_df


def load_image(image_path: str) -> Image.Image:
    """Load an RGB image from disk or URL."""
    if image_path.startswith(("http://", "https://")):
        response = requests.get(image_path, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

    return Image.open(image_path).convert("RGB")


def load_mask(mask_path: str) -> Image.Image:
    """Load a segmentation mask as a single-channel PIL image."""
    with Path(mask_path).open("rb") as file_handle:
        return Image.open(file_handle).convert("L")


binary_loader = load_mask


