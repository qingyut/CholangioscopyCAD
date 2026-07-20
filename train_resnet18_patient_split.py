#!/usr/bin/env python3
"""
Patient-disjoint ResNet18 training and external validation for DSOC CADx.

Design principles
-----------------
1. Development images are split at the PATIENT level, not the image level.
2. The internal validation partition is used for checkpoint selection, early stopping,
   and threshold selection; it is therefore not described as an independent internal test set.
3. External data are never used for model selection, hyperparameter selection, threshold
   selection, or early stopping.
4. Image-level and patient-level outputs are both saved. Patient-level aggregation defaults
   to the mean of the highest-scoring 10% of images for each patient.
5. Every run saves the exact split manifest, package versions, command-line configuration,
   thresholds, predictions, and summary counts needed for the manuscript.

Input CSV
---------
Required columns:
    image_path   Path to an image. Relative paths are resolved relative to the CSV file.
    patient_id   Patient identifier.
    label        Final patient-level reference standard: 0=non-neoplasia, 1=neoplasia.

Optional columns:
    video_id     Video identifier.
    frame_id     Frame/image identifier.
    center       Institution/center identifier.

Typical commands
----------------
Training + internal validation + independent external evaluation:

    python train_resnet18_patient_split.py \
        --development-csv development_metadata.csv \
        --external-csv external_metadata.csv \
        --output-dir resnet18_run_01 \
        --val-fraction 0.10 \
        --seed 42 \
        --epochs 30 \
        --patience 8

CPU smoke test without downloading pretrained weights:

    python train_resnet18_patient_split.py \
        --development-csv development_metadata.csv \
        --output-dir smoke_test \
        --epochs 1 \
        --patience 1 \
        --no-pretrained \
        --device cpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torchvision
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


REQUIRED_COLUMNS = {"image_path", "patient_id", "label"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class RunConfig:
    development_csv: str
    external_csv: Optional[str]
    output_dir: str
    val_fraction: float
    seed: int
    epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    num_workers: int
    image_size: int
    pretrained: bool
    device: str
    patient_aggregation: str
    top_fraction: float
    bootstrap_replicates: int
    deterministic: bool


class ImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform: transforms.Compose):
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = Path(row["resolved_image_path"])
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Unable to open image: {image_path}") from exc

        image_tensor = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return image_tensor, label, index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet18 using a patient-disjoint internal split."
    )
    parser.add_argument("--development-csv", type=Path, required=True)
    parser.add_argument("--external-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet pretrained weights. Use --no-pretrained for an offline smoke test.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--patient-aggregation",
        choices=("top10_mean", "p95", "mean", "max"),
        default="top10_mean",
    )
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Request deterministic PyTorch algorithms. This may substantially slow training "
            "and is not guaranteed to reproduce results across different platforms/releases."
        ),
    )
    parser.add_argument(
        "--skip-file-check",
        action="store_true",
        help="Skip the preflight check that opens every image.",
    )
    return parser.parse_args()


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        # Optional because deterministic kernels can be markedly slower.
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metadata(csv_path: Path, dataset_name: str) -> pd.DataFrame:
    csv_path = csv_path.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"{dataset_name} CSV does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{dataset_name} CSV is missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{dataset_name} CSV is empty.")

    df = df.copy()
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    if (df["patient_id"] == "").any():
        raise ValueError(f"{dataset_name}: empty patient_id detected.")

    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{dataset_name}: label must contain only 0 and 1.")

    patient_label_counts = df.groupby("patient_id")["label"].nunique()
    inconsistent = patient_label_counts[patient_label_counts != 1]
    if not inconsistent.empty:
        raise ValueError(
            f"{dataset_name}: inconsistent labels within patient(s): "
            f"{inconsistent.index.tolist()[:10]}"
        )

    csv_parent = csv_path.parent
    def resolve_path(value: object) -> str:
        path = Path(str(value))
        if not path.is_absolute():
            path = csv_parent / path
        return str(path.resolve())

    df["resolved_image_path"] = df["image_path"].map(resolve_path)

    if "frame_id" not in df.columns:
        df["frame_id"] = [f"{dataset_name}_image_{i:06d}" for i in range(len(df))]
    else:
        df["frame_id"] = df["frame_id"].astype(str)

    if "video_id" in df.columns:
        df["video_id"] = df["video_id"].astype(str)
    if "center" in df.columns:
        df["center"] = df["center"].astype(str)

    if df["resolved_image_path"].duplicated().any():
        duplicated = df.loc[df["resolved_image_path"].duplicated(), "resolved_image_path"]
        raise ValueError(
            f"{dataset_name}: duplicated image paths detected, for example: "
            f"{duplicated.iloc[0]}"
        )
    return df


def preflight_images(df: pd.DataFrame, dataset_name: str) -> None:
    errors: List[str] = []
    for path_text in df["resolved_image_path"]:
        path = Path(path_text)
        if not path.exists():
            errors.append(f"Missing: {path}")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            errors.append(f"Unreadable: {path} ({exc})")
        if len(errors) >= 20:
            break
    if errors:
        joined = "\n".join(errors)
        raise RuntimeError(f"{dataset_name} image preflight failed:\n{joined}")


def create_patient_split(
    development_df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5.")

    patient_table = (
        development_df[["patient_id", "label"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    if patient_table["label"].value_counts().min() < 2:
        raise ValueError("Each class must contain at least two patients.")

    train_patients, val_patients = train_test_split(
        patient_table,
        test_size=val_fraction,
        random_state=seed,
        stratify=patient_table["label"],
        shuffle=True,
    )

    train_ids = set(train_patients["patient_id"])
    val_ids = set(val_patients["patient_id"])
    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError(f"Patient leakage detected: {sorted(overlap)[:10]}")

    train_df = development_df[development_df["patient_id"].isin(train_ids)].copy()
    val_df = development_df[development_df["patient_id"].isin(val_ids)].copy()

    split_manifest = patient_table.copy()
    split_manifest["split"] = split_manifest["patient_id"].map(
        lambda value: "train" if value in train_ids else "internal_validation"
    )
    return train_df, val_df, split_manifest


def split_summary(df: pd.DataFrame, name: str) -> Dict[str, object]:
    patient_table = df[["patient_id", "label"]].drop_duplicates()
    return {
        "split": name,
        "patients_total": int(patient_table.shape[0]),
        "patients_neoplasia": int((patient_table["label"] == 1).sum()),
        "patients_non_neoplasia": int((patient_table["label"] == 0).sum()),
        "images_total": int(df.shape[0]),
        "images_neoplasia": int((df["label"] == 1).sum()),
        "images_non_neoplasia": int((df["label"] == 0).sum()),
    }


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.85, 1.0),
                ratio=(0.9, 1.1),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(round(image_size * 256 / 224))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_model(pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)
    return model


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def compute_pos_weight(train_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    positives = int((train_df["label"] == 1).sum())
    negatives = int((train_df["label"] == 0).sum())
    if positives == 0 or negatives == 0:
        raise ValueError("Training images must include both classes.")
    return torch.tensor([negatives / positives], dtype=torch.float32, device=device)


def create_grad_scaler(amp_enabled: bool):
    """Create a GradScaler compatible with both newer and older supported PyTorch releases."""
    try:
        return torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def load_checkpoint(path: Path, device: torch.device):
    """Load a full training checkpoint across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=torch.float16,
        ):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.shape[0]
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    return running_loss / sample_count


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    dataframe: pd.DataFrame,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[pd.DataFrame, float]:
    model.eval()
    probabilities = np.zeros(len(dataframe), dtype=np.float64)
    labels_out = np.zeros(len(dataframe), dtype=np.int64)
    running_loss = 0.0
    sample_count = 0

    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
            dtype=torch.float16,
        ):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels_device)

        probs = torch.sigmoid(logits).cpu().numpy()
        indices_np = indices.numpy()
        probabilities[indices_np] = probs
        labels_out[indices_np] = labels.numpy().astype(int)

        batch_size = labels.shape[0]
        running_loss += float(loss.item()) * batch_size
        sample_count += batch_size

    output = dataframe.reset_index(drop=True).copy()
    output["true_label"] = labels_out
    output["malignancy_probability"] = probabilities
    return output, running_loss / sample_count


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def select_youden_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        raise ValueError("Threshold selection requires both classes.")
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    finite = np.isfinite(thresholds)
    fpr, tpr, thresholds = fpr[finite], tpr[finite], thresholds[finite]
    youden = tpr - fpr
    best_indices = np.flatnonzero(youden == np.max(youden))
    # Tie-breaker: highest sensitivity, then highest specificity, then highest threshold.
    candidates = []
    for index in best_indices:
        candidates.append(
            (
                float(tpr[index]),
                float(1 - fpr[index]),
                float(thresholds[index]),
            )
        )
    return max(candidates)[2]


def confusion_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def divide(num: int, den: int) -> float:
        return float(num / den) if den else float("nan")

    return {
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "auc": safe_auc(y_true, scores),
        "sensitivity": divide(tp, tp + fn),
        "specificity": divide(tn, tn + fp),
        "accuracy": divide(tp + tn, len(y_true)),
        "ppv": divide(tp, tp + fp),
        "npv": divide(tn, tn + fn),
    }


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def aggregate_scores(
    scores: Iterable[float],
    method: str,
    top_fraction: float,
) -> float:
    values = np.asarray(list(scores), dtype=float)
    if values.size == 0:
        raise ValueError("Cannot aggregate an empty score array.")
    if method == "mean":
        return float(values.mean())
    if method == "max":
        return float(values.max())
    if method == "p95":
        return float(np.quantile(values, 0.95))
    if method == "top10_mean":
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be within (0, 1].")
        count = max(1, int(math.ceil(values.size * top_fraction)))
        return float(np.sort(values)[-count:].mean())
    raise ValueError(f"Unknown aggregation method: {method}")


def aggregate_patient_predictions(
    image_predictions: pd.DataFrame,
    method: str,
    top_fraction: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for patient_id, group in image_predictions.groupby("patient_id", sort=True):
        rows.append(
            {
                "patient_id": patient_id,
                "label": int(group["true_label"].iloc[0]),
                "n_images": int(len(group)),
                "n_videos": (
                    int(group["video_id"].nunique())
                    if "video_id" in group.columns
                    else np.nan
                ),
                "patient_score": aggregate_scores(
                    group["malignancy_probability"],
                    method,
                    top_fraction,
                ),
            }
        )
    return pd.DataFrame(rows)


def patient_metric_table(
    patient_predictions: pd.DataFrame,
    threshold: float,
    bootstrap_replicates: int,
    seed: int,
) -> pd.DataFrame:
    y_true = patient_predictions["label"].to_numpy(dtype=int)
    scores = patient_predictions["patient_score"].to_numpy(dtype=float)
    point = confusion_metrics(y_true, scores, threshold)

    counts = {
        "sensitivity": (point["TP"], point["TP"] + point["FN"]),
        "specificity": (point["TN"], point["TN"] + point["FP"]),
        "accuracy": (point["TP"] + point["TN"], len(y_true)),
        "ppv": (point["TP"], point["TP"] + point["FP"]),
        "npv": (point["TN"], point["TN"] + point["FN"]),
    }

    rows: List[Dict[str, object]] = []
    for metric, (numerator, denominator) in counts.items():
        low, high = wilson_interval(int(numerator), int(denominator))
        rows.append(
            {
                "metric": metric,
                "estimate": point[metric],
                "ci_low": low,
                "ci_high": high,
                "ci_method": "Wilson",
                "numerator": int(numerator),
                "denominator": int(denominator),
            }
        )

    auc_low, auc_high = bootstrap_auc_patient(
        y_true,
        scores,
        bootstrap_replicates,
        seed,
    )
    rows.append(
        {
            "metric": "auc",
            "estimate": point["auc"],
            "ci_low": auc_low,
            "ci_high": auc_high,
            "ci_method": "Stratified patient bootstrap",
            "numerator": np.nan,
            "denominator": len(y_true),
        }
    )
    return pd.DataFrame(rows)


def bootstrap_auc_patient(
    y_true: np.ndarray,
    scores: np.ndarray,
    replicates: int,
    seed: int,
) -> Tuple[float, float]:
    positive_indices = np.flatnonzero(y_true == 1)
    negative_indices = np.flatnonzero(y_true == 0)
    if len(positive_indices) == 0 or len(negative_indices) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    auc_values: List[float] = []
    for _ in range(replicates):
        sampled_indices = np.concatenate(
            [
                rng.choice(
                    positive_indices,
                    size=len(positive_indices),
                    replace=True,
                ),
                rng.choice(
                    negative_indices,
                    size=len(negative_indices),
                    replace=True,
                ),
            ]
        )
        auc_values.append(
            roc_auc_score(y_true[sampled_indices], scores[sampled_indices])
        )
    low, high = np.quantile(auc_values, [0.025, 0.975])
    return float(low), float(high)


def cluster_bootstrap_image_metrics(
    image_predictions: pd.DataFrame,
    threshold: float,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    grouped = {
        patient_id: group
        for patient_id, group in image_predictions.groupby("patient_id")
    }
    patient_table = image_predictions[["patient_id", "true_label"]].drop_duplicates()
    positive_ids = patient_table.loc[
        patient_table["true_label"] == 1, "patient_id"
    ].to_numpy()
    negative_ids = patient_table.loc[
        patient_table["true_label"] == 0, "patient_id"
    ].to_numpy()

    point = confusion_metrics(
        image_predictions["true_label"].to_numpy(dtype=int),
        image_predictions["malignancy_probability"].to_numpy(dtype=float),
        threshold,
    )

    rng = np.random.default_rng(seed)
    metric_names = ("auc", "sensitivity", "specificity", "accuracy", "ppv", "npv")
    distributions: Dict[str, List[float]] = {name: [] for name in metric_names}

    for _ in range(replicates):
        sampled_ids = list(
            rng.choice(positive_ids, size=len(positive_ids), replace=True)
        )
        sampled_ids += list(
            rng.choice(negative_ids, size=len(negative_ids), replace=True)
        )
        sampled = pd.concat(
            [grouped[patient_id] for patient_id in sampled_ids],
            ignore_index=True,
        )
        values = confusion_metrics(
            sampled["true_label"].to_numpy(dtype=int),
            sampled["malignancy_probability"].to_numpy(dtype=float),
            threshold,
        )
        for name in metric_names:
            if not np.isnan(values[name]):
                distributions[name].append(values[name])

    rows: List[Dict[str, object]] = []
    for name in metric_names:
        low, high = np.quantile(distributions[name], [0.025, 0.975])
        rows.append(
            {
                "metric": name,
                "estimate": point[name],
                "ci_low": float(low),
                "ci_high": float(high),
                "ci_method": "Stratified patient-cluster bootstrap",
            }
        )
    return pd.DataFrame(rows)


def save_confusion_plot(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    title: str,
    output_path: Path,
) -> None:
    y_pred = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix)
    axis.set_xticks([0, 1], labels=["Non-neoplasia", "Neoplasia"])
    axis.set_yticks([0, 1], labels=["Non-neoplasia", "Neoplasia"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Reference label")
    axis.set_title(title)
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_roc_plot(
    y_true: np.ndarray,
    scores: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_value = roc_auc_score(y_true, scores)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot(fpr, tpr, label=f"AUC = {auc_value:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title(title)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def evaluate_and_save(
    name: str,
    image_predictions: pd.DataFrame,
    image_threshold: float,
    patient_threshold: float,
    aggregation: str,
    top_fraction: float,
    output_dir: Path,
    bootstrap_replicates: int,
    seed: int,
) -> Dict[str, object]:
    name_dir = output_dir / name
    name_dir.mkdir(parents=True, exist_ok=True)

    image_predictions = image_predictions.copy()
    image_predictions["predicted_label"] = (
        image_predictions["malignancy_probability"] >= image_threshold
    ).astype(int)
    image_predictions.to_csv(name_dir / "image_predictions.csv", index=False)

    image_metrics = cluster_bootstrap_image_metrics(
        image_predictions,
        image_threshold,
        bootstrap_replicates,
        seed,
    )
    image_metrics.to_csv(name_dir / "image_metrics_clustered.csv", index=False)

    patient_predictions = aggregate_patient_predictions(
        image_predictions,
        aggregation,
        top_fraction,
    )
    patient_predictions["predicted_label"] = (
        patient_predictions["patient_score"] >= patient_threshold
    ).astype(int)
    patient_predictions.to_csv(name_dir / "patient_predictions.csv", index=False)

    patient_metrics = patient_metric_table(
        patient_predictions,
        patient_threshold,
        bootstrap_replicates,
        seed + 1,
    )
    patient_metrics.to_csv(name_dir / "patient_metrics.csv", index=False)

    save_confusion_plot(
        patient_predictions["label"].to_numpy(dtype=int),
        patient_predictions["patient_score"].to_numpy(dtype=float),
        patient_threshold,
        f"{name}: patient-level confusion matrix",
        name_dir / "patient_confusion_matrix.png",
    )
    save_roc_plot(
        patient_predictions["label"].to_numpy(dtype=int),
        patient_predictions["patient_score"].to_numpy(dtype=float),
        f"{name}: patient-level ROC",
        name_dir / "patient_roc_curve.png",
    )

    point_image = confusion_metrics(
        image_predictions["true_label"].to_numpy(dtype=int),
        image_predictions["malignancy_probability"].to_numpy(dtype=float),
        image_threshold,
    )
    point_patient = confusion_metrics(
        patient_predictions["label"].to_numpy(dtype=int),
        patient_predictions["patient_score"].to_numpy(dtype=float),
        patient_threshold,
    )

    return {
        "image_level": point_image,
        "patient_level": point_patient,
        "patients": int(patient_predictions.shape[0]),
        "images": int(image_predictions.shape[0]),
    }


def build_manuscript_text(
    split_summaries: List[Dict[str, object]],
    best_epoch: int,
    image_threshold: float,
    patient_threshold: float,
    aggregation: str,
    top_fraction: float,
) -> str:
    summary_map = {item["split"]: item for item in split_summaries}
    train = summary_map["training"]
    validation = summary_map["internal_validation"]

    text = (
        "Stage II (CADx—diagnosis classification): ResNet18 was used for binary "
        "image classification. The development cohort was stratified at the patient "
        f"level into a training set of {train['patients_total']} patients "
        f"({train['patients_neoplasia']} neoplastic and "
        f"{train['patients_non_neoplasia']} non-neoplastic; "
        f"{train['images_total']} images) and an internal validation set of "
        f"{validation['patients_total']} patients "
        f"({validation['patients_neoplasia']} neoplastic and "
        f"{validation['patients_non_neoplasia']} non-neoplastic; "
        f"{validation['images_total']} images). All images from a given patient were "
        "restricted to one partition. The internal validation set was evaluated after "
        "each epoch and was used for early stopping and selection of the checkpoint "
        f"with the highest validation AUC (best epoch: {best_epoch}); it therefore "
        "served as a model-selection set rather than an independent internal test set. "
        "No external image or patient was used for model selection or threshold "
        f"calibration. The image-level operating threshold ({image_threshold:.6f}) "
        "and the patient-level threshold "
        f"({patient_threshold:.6f}) were selected on the internal validation set and "
        "locked before external evaluation. Patient scores were calculated using "
        f"{aggregation} aggregation"
    )
    if aggregation == "top10_mean":
        text += f" with a top-frame fraction of {top_fraction:.2f}"
    text += (
        ". No k-fold cross-validation was performed; the use of a single patient-level "
        "internal split reduces the precision of the internal performance estimate and "
        "is acknowledged as a limitation."
    )
    return text


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed, args.deterministic)
    device = choose_device(args.device)
    amp_enabled = device.type == "cuda"

    config = RunConfig(
        development_csv=str(args.development_csv.resolve()),
        external_csv=(
            str(args.external_csv.resolve()) if args.external_csv is not None else None
        ),
        output_dir=str(args.output_dir.resolve()),
        val_fraction=args.val_fraction,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        image_size=args.image_size,
        pretrained=args.pretrained,
        device=str(device),
        patient_aggregation=args.patient_aggregation,
        top_fraction=args.top_fraction,
        bootstrap_replicates=args.bootstrap_replicates,
        deterministic=args.deterministic,
    )

    development_df = load_metadata(args.development_csv, "development")
    external_df = (
        load_metadata(args.external_csv, "external")
        if args.external_csv is not None
        else None
    )

    if not args.skip_file_check:
        preflight_images(development_df, "development")
        if external_df is not None:
            preflight_images(external_df, "external")

    train_df, val_df, split_manifest = create_patient_split(
        development_df,
        args.val_fraction,
        args.seed,
    )

    if external_df is not None:
        development_ids = set(development_df["patient_id"])
        external_ids = set(external_df["patient_id"])
        overlap = development_ids & external_ids
        if overlap:
            raise RuntimeError(
                f"External patient leakage detected: {sorted(overlap)[:10]}"
            )
        duplicated_paths = set(development_df["resolved_image_path"]) & set(
            external_df["resolved_image_path"]
        )
        if duplicated_paths:
            raise RuntimeError(
                "Development/external image path duplication detected, for example: "
                f"{next(iter(duplicated_paths))}"
            )

    split_manifest.to_csv(args.output_dir / "patient_split_manifest.csv", index=False)
    train_df.to_csv(args.output_dir / "training_images_manifest.csv", index=False)
    val_df.to_csv(args.output_dir / "internal_validation_images_manifest.csv", index=False)
    if external_df is not None:
        external_df.to_csv(args.output_dir / "external_images_manifest.csv", index=False)

    summaries = [
        split_summary(train_df, "training"),
        split_summary(val_df, "internal_validation"),
    ]
    if external_df is not None:
        summaries.append(split_summary(external_df, "external"))
    pd.DataFrame(summaries).to_csv(
        args.output_dir / "split_summary.csv",
        index=False,
    )

    provenance = {
        "config": asdict(config),
        "development_csv_sha256": sha256_file(args.development_csv.resolve()),
        "external_csv_sha256": (
            sha256_file(args.external_csv.resolve())
            if args.external_csv is not None
            else None
        ),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None
        ),
    }
    (args.output_dir / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    train_transform, eval_transform = build_transforms(args.image_size)
    train_dataset = ImageDataset(train_df, train_transform)
    val_dataset = ImageDataset(val_df, eval_transform)

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        True,
        args.num_workers,
        args.seed,
        pin_memory,
    )
    val_loader = make_loader(
        val_dataset,
        args.batch_size,
        False,
        args.num_workers,
        args.seed + 1,
        pin_memory,
    )

    model = build_model(args.pretrained).to(device)
    pos_weight = compute_pos_weight(train_df, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = create_grad_scaler(amp_enabled)

    best_auc = -float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: List[Dict[str, float]] = []
    checkpoint_path = args.output_dir / "best_model.pt"

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            amp_enabled,
        )
        val_predictions, val_loss = predict(
            model,
            val_loader,
            val_df,
            criterion,
            device,
            amp_enabled,
        )
        val_auc = safe_auc(
            val_predictions["true_label"].to_numpy(dtype=int),
            val_predictions["malignancy_probability"].to_numpy(dtype=float),
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "validation_auc": val_auc,
            }
        )
        pd.DataFrame(history).to_csv(
            args.output_dir / "training_history.csv",
            index=False,
        )

        improved = val_auc > best_auc + 1e-6
        if improved:
            best_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_auc": val_auc,
                    "config": asdict(config),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss={train_loss:.5f} | "
            f"val loss={val_loss:.5f} | "
            f"val AUC={val_auc:.5f} | "
            f"best={best_auc:.5f}"
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping after epoch {epoch}; "
                f"best epoch was {best_epoch}."
            )
            break

    if not checkpoint_path.exists():
        raise RuntimeError("No checkpoint was saved.")

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_predictions, _ = predict(
        model,
        val_loader,
        val_df,
        criterion,
        device,
        amp_enabled,
    )
    image_threshold = select_youden_threshold(
        val_predictions["true_label"].to_numpy(dtype=int),
        val_predictions["malignancy_probability"].to_numpy(dtype=float),
    )

    val_patient_predictions = aggregate_patient_predictions(
        val_predictions,
        args.patient_aggregation,
        args.top_fraction,
    )
    patient_threshold = select_youden_threshold(
        val_patient_predictions["label"].to_numpy(dtype=int),
        val_patient_predictions["patient_score"].to_numpy(dtype=float),
    )

    thresholds = {
        "selected_on": "internal_validation",
        "image_threshold": image_threshold,
        "patient_threshold": patient_threshold,
        "patient_aggregation": args.patient_aggregation,
        "top_fraction": args.top_fraction,
    }
    (args.output_dir / "locked_thresholds.json").write_text(
        json.dumps(thresholds, indent=2),
        encoding="utf-8",
    )

    results: Dict[str, object] = {}
    results["internal_validation"] = evaluate_and_save(
        "internal_validation",
        val_predictions,
        image_threshold,
        patient_threshold,
        args.patient_aggregation,
        args.top_fraction,
        args.output_dir,
        args.bootstrap_replicates,
        args.seed + 10,
    )

    if external_df is not None:
        external_dataset = ImageDataset(external_df, eval_transform)
        external_loader = make_loader(
            external_dataset,
            args.batch_size,
            False,
            args.num_workers,
            args.seed + 2,
            pin_memory,
        )
        external_predictions, _ = predict(
            model,
            external_loader,
            external_df,
            criterion,
            device,
            amp_enabled,
        )
        results["external"] = evaluate_and_save(
            "external",
            external_predictions,
            image_threshold,
            patient_threshold,
            args.patient_aggregation,
            args.top_fraction,
            args.output_dir,
            args.bootstrap_replicates,
            args.seed + 20,
        )

    elapsed_seconds = time.time() - start_time
    results["training"] = {
        "best_epoch": best_epoch,
        "best_internal_validation_auc": best_auc,
        "elapsed_seconds": elapsed_seconds,
        "image_threshold": image_threshold,
        "patient_threshold": patient_threshold,
    }
    (args.output_dir / "results_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manuscript_text = build_manuscript_text(
        summaries,
        best_epoch,
        image_threshold,
        patient_threshold,
        args.patient_aggregation,
        args.top_fraction,
    )
    (args.output_dir / "manuscript_methods_text.txt").write_text(
        manuscript_text,
        encoding="utf-8",
    )

    print("\nRun completed successfully.")
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Best epoch: {best_epoch}")
    print(f"Locked image threshold: {image_threshold:.6f}")
    print(f"Locked patient threshold: {patient_threshold:.6f}")
    print(f"Output directory: {args.output_dir.resolve()}")
    print("\nManuscript-ready Methods text:\n")
    print(manuscript_text)


if __name__ == "__main__":
    main()
