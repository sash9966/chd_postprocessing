"""Per-class Dice computation and folder-level evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import FOREGROUND_CLASSES, LABEL_NAMES
from .io_utils import load_nifti, resolve_case_id


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Binary Dice between two boolean/integer arrays.

    Returns ``float('nan')`` when both masks are empty (class absent in both
    prediction and ground truth — not a mistake, just not evaluable).
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = int(np.sum(pred & gt))
    denom = int(np.sum(pred)) + int(np.sum(gt))
    if denom == 0:
        return float("nan")
    return 2.0 * intersection / denom


def dice_per_class(
    pred: np.ndarray,
    gt: np.ndarray,
    class_indices: Optional[List[int]] = None,
) -> Dict[int, float]:
    """Compute Dice for each class in *class_indices*.

    Parameters
    ----------
    pred : predicted integer label volume
    gt : ground-truth integer label volume (same shape)
    class_indices : class IDs to evaluate.  Default: all foreground (1–7).

    Returns
    -------
    dict mapping class_id → Dice score (float or nan)
    """
    if class_indices is None:
        class_indices = FOREGROUND_CLASSES
    return {cls: dice_score(pred == cls, gt == cls) for cls in class_indices}


# ---------------------------------------------------------------------------
# File matching helper
# ---------------------------------------------------------------------------

def _find_gt_file(gt_folder: Path, case_id: str) -> Optional[Path]:
    """Find a GT file for *case_id*, trying several naming conventions.

    nnU-Net predictions are typically named ``ct_1001_image.nii.gz``.
    GT files may be ``ct_1001_image.nii.gz`` or ``ct_1001.nii.gz``
    depending on the dataset organisation.
    """
    base_id = case_id.replace("_image", "")
    candidates = [
        gt_folder / f"{case_id}.nii.gz",
        gt_folder / f"{case_id}.nii",
        gt_folder / f"{base_id}.nii.gz",
        gt_folder / f"{base_id}.nii",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Fuzzy fallback: any file whose stem contains the base ID
    for f in sorted(gt_folder.glob("*.nii*")):
        if base_id in f.name:
            return f
    return None


# ---------------------------------------------------------------------------
# Folder-level evaluation
# ---------------------------------------------------------------------------

def evaluate_folder(
    pred_folder: str | Path,
    gt_folder: str | Path,
    class_indices: Optional[List[int]] = None,
    file_pattern: str = "*.nii.gz",
) -> pd.DataFrame:
    """Evaluate all predictions in *pred_folder* against ground truth.

    Parameters
    ----------
    pred_folder : folder containing predicted NIfTI files
    gt_folder : folder containing ground-truth NIfTI files
    class_indices : class IDs to evaluate (default: 1–7)
    file_pattern : glob pattern for prediction files

    Returns
    -------
    DataFrame indexed by ``case_id`` with one column per class plus
    ``mean_fg`` (mean over non-NaN foreground Dice scores).
    Columns are named by label (e.g. ``"AO"``, ``"PA"``, …).
    """
    pred_folder = Path(pred_folder)
    gt_folder = Path(gt_folder)
    if class_indices is None:
        class_indices = FOREGROUND_CLASSES

    rows = []
    for pred_path in sorted(pred_folder.glob(file_pattern)):
        case_id = resolve_case_id(pred_path.name)
        gt_path = _find_gt_file(gt_folder, case_id)
        if gt_path is None:
            print(f"  [WARNING] No GT found for {case_id} — skipping")
            continue

        pred_data, _, _ = load_nifti(pred_path)
        gt_data, _, _ = load_nifti(gt_path)
        pred_data = pred_data.astype(np.int32)
        gt_data   = gt_data.astype(np.int32)

        scores = dice_per_class(pred_data, gt_data, class_indices)
        row: Dict = {"case_id": case_id}
        row.update({LABEL_NAMES.get(cls, str(cls)): v for cls, v in scores.items()})
        fg_vals = [v for v in scores.values() if not np.isnan(v)]
        row["mean_fg"] = float(np.mean(fg_vals)) if fg_vals else float("nan")
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("case_id")
    return df


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean ± std across cases for each column in *df*."""
    summary = pd.DataFrame({
        "mean": df.mean(numeric_only=True),
        "std":  df.std(numeric_only=True),
    })
    return summary
