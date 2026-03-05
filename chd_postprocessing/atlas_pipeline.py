"""Atlas-based post-processing pipeline (Part 1: baseline, Part 2: disease-specific).

Part 1 — Baseline
-----------------
For each prediction:
  1. Randomly select one atlas from the training GT library.
  2. Apply a small random perturbation (so registration is non-trivial).
  3. Register the perturbed atlas into the prediction's coordinate frame.
  4. Correct labels via Dice-overlap + Hungarian matching.
  5. Apply structural cleanup (CC filter + morphological closing).

Part 2 — Disease-specific
--------------------------
Identical to Part 1 except that the atlas is selected to *best match* the
disease profile of the test case (minimum Hamming distance on the binary
disease-flag vector).  For pulmonary atresia (PuA flag = 1) the AO/PA swap
correction from :mod:`anatomy_priors` is skipped — but the atlas-based
relabelling still runs, because the atlas for a PuA case will itself have the
expected fused AO/PA morphology.

Usage
-----
Single file::

    from chd_postprocessing.atlas_pipeline import run_atlas_pipeline
    result = run_atlas_pipeline(
        pred_path    = "pred.nii.gz",
        gt_folder    = "labelsTr/",
        output_path  = "corrected.nii.gz",
        mode         = "disease_specific",
        disease_map_path = "disease_map.json",
        gt_path      = "gt.nii.gz",      # optional, enables Dice evaluation
    )

Whole folder::

    from chd_postprocessing.atlas_pipeline import run_atlas_folder_pipeline
    df = run_atlas_folder_pipeline("preds/", "labelsTr/", "corrected/", mode="baseline")
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .atlas import AtlasLibrary
from .config import FOREGROUND_CLASSES, LABEL_NAMES
from .evaluate import dice_per_class
from .io_utils import get_disease_vec, get_voxel_spacing, load_disease_map, load_nifti, save_nifti, resolve_case_id
from .label_correction import correct_labels_with_atlas
from .registration import register_atlas_to_pred


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AtlasPipelineResult:
    """Complete output of one atlas-based post-processing run.

    Attributes
    ----------
    corrected_labels : post-processed integer label volume.
    affine, header : geometry metadata from the input prediction NIfTI.
    atlas_case_id : case ID of the atlas that was used.
    atlas_disease_name : human-readable disease profile of the atlas.
    mode : ``"baseline"`` or ``"disease_specific"``.
    was_relabeled : True if any label assignment was changed.
    mapping_applied : dict ``{pred_label: corrected_label}``.
    reassignment_summary : human-readable description of changes.
    overlap_matrix : (N, N) Dice overlap matrix (pred rows, atlas cols).
    component_info : per-label CC statistics.
    dice_before : per-class Dice scores before correction (None if GT absent).
    dice_after : per-class Dice scores after correction (None if GT absent).
    """
    corrected_labels:     np.ndarray
    affine:               np.ndarray
    header:               object

    atlas_case_id:        str
    atlas_disease_name:   str
    mode:                 str

    was_relabeled:        bool
    mapping_applied:      Dict[int, int]
    reassignment_summary: str
    overlap_matrix:       np.ndarray
    label_ids:            List[int]
    component_info:       Dict

    dice_before: Optional[Dict[str, float]] = None
    dice_after:  Optional[Dict[str, float]] = None

    def dice_delta(self) -> Optional[Dict[str, float]]:
        """Per-class Dice improvement (after − before).  None if GT unavailable."""
        if self.dice_before is None or self.dice_after is None:
            return None
        return {
            cls: (self.dice_after[cls] - self.dice_before[cls])
            for cls in self.dice_before
        }

    def summary_dict(self) -> Dict:
        """Flat dict suitable for building a results DataFrame row."""
        d: Dict = {
            "atlas_case_id":      self.atlas_case_id,
            "atlas_disease_name": self.atlas_disease_name,
            "mode":               self.mode,
            "was_relabeled":      self.was_relabeled,
        }
        if self.dice_before is not None:
            for cls, v in self.dice_before.items():
                d[f"dice_before_{cls}"] = round(v, 4) if not np.isnan(v) else float("nan")
        if self.dice_after is not None:
            for cls, v in self.dice_after.items():
                d[f"dice_after_{cls}"] = round(v, 4) if not np.isnan(v) else float("nan")
        delta = self.dice_delta()
        if delta is not None:
            for cls, v in delta.items():
                d[f"dice_delta_{cls}"] = round(v, 4) if not np.isnan(v) else float("nan")
            non_nan = [v for v in delta.values() if not np.isnan(v)]
            d["mean_dice_delta"] = round(float(np.mean(non_nan)), 4) if non_nan else float("nan")
        return d


# ---------------------------------------------------------------------------
# Single-case pipeline
# ---------------------------------------------------------------------------

def run_atlas_pipeline(
    pred_path:          str | Path,
    gt_folder:          str | Path,
    output_path:        Optional[str | Path] = None,
    disease_map_path:   Optional[str | Path] = None,
    gt_path:            Optional[str | Path] = None,
    mode:               str = "baseline",
    seed:               int = 42,
    registration_mode:  str = "centroid",
    min_overlap:        float = 0.01,
    min_component_fraction: float = 0.01,
    do_morphological_cleanup: bool = True,
    label_ids:          Optional[List[int]] = None,
) -> AtlasPipelineResult:
    """Run the atlas-based post-processing pipeline on a single prediction.

    Parameters
    ----------
    pred_path : path to the nnU-Net prediction NIfTI.
    gt_folder : folder containing training GT label NIfTI files (atlas library).
    output_path : where to save the corrected NIfTI.  Skipped if ``None``.
    disease_map_path : path to ``disease_map.json``.
                       Required for ``mode="disease_specific"``.
    gt_path : ground-truth segmentation for the current case.
              When supplied, Dice scores before/after correction are computed.
    mode : ``"baseline"`` — random atlas selection;
           ``"disease_specific"`` — atlas best-matching the case's disease profile.
    seed : random seed for reproducibility.
    registration_mode : ``"centroid"`` (default) or ``"pca"``
                        (see :mod:`registration`).
    min_overlap : minimum Dice for a component-atlas match to be accepted.
    min_component_fraction : small-fragment reassignment threshold.
    do_morphological_cleanup : apply closing + hole-fill to AO and PA.
    label_ids : foreground labels to consider.  Default: 1–7.

    Returns
    -------
    :class:`AtlasPipelineResult`
    """
    if mode not in {"baseline", "disease_specific"}:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'baseline' or 'disease_specific'.")
    if label_ids is None:
        label_ids = list(FOREGROUND_CLASSES)

    rng = random.Random(seed)   # used only for atlas selection tie-breaking

    # ------------------------------------------------------------------
    # 1. Load prediction
    # ------------------------------------------------------------------
    pred_labels, affine, header = load_nifti(pred_path)
    pred_labels  = pred_labels.astype(np.int32)
    pred_spacing = get_voxel_spacing(header)

    case_id = resolve_case_id(Path(pred_path).name)

    # ------------------------------------------------------------------
    # 2. Build atlas library
    # ------------------------------------------------------------------
    library = AtlasLibrary.load_all(
        gt_folder,
        disease_map_path=disease_map_path,
    )
    if len(library) == 0:
        raise FileNotFoundError(f"No GT files found in {gt_folder}")

    # ------------------------------------------------------------------
    # 3. Determine disease vector for this case
    # ------------------------------------------------------------------
    disease_vec = [0] * 8
    if disease_map_path is not None:
        dm = load_disease_map(disease_map_path)
        vec = get_disease_vec(dm, case_id)
        if vec is not None:
            disease_vec = vec

    # ------------------------------------------------------------------
    # 4. Select atlas
    # ------------------------------------------------------------------
    selection_mode = "best_match" if mode == "disease_specific" else "random"
    atlas_entry = library.select_for_case(
        disease_vec=disease_vec,
        rng=rng,
        mode=selection_mode,
        exclude_case_id=case_id,
    )
    atlas_entry.load()
    atlas_labels = atlas_entry.labels

    # ------------------------------------------------------------------
    # 5. Compute Dice *before* correction (optional)
    # ------------------------------------------------------------------
    dice_before = None
    if gt_path is not None:
        gt_labels, _, _ = load_nifti(gt_path)
        gt_labels = gt_labels.astype(np.int32)
        raw_scores = dice_per_class(pred_labels, gt_labels, label_ids)
        dice_before = {LABEL_NAMES.get(k, str(k)): v for k, v in raw_scores.items()}

    # ------------------------------------------------------------------
    # 6. Register atlas → prediction space (no synthetic perturbation;
    #    the atlas is already a different patient so perturbation only
    #    degrades registration accuracy)
    # ------------------------------------------------------------------
    registered_atlas = register_atlas_to_pred(
        atlas_labels, pred_labels, pred_spacing, mode=registration_mode
    )

    # ------------------------------------------------------------------
    # 7. Component-level label correction
    # ------------------------------------------------------------------
    correction = correct_labels_with_atlas(
        pred_labels, registered_atlas,
        label_ids=label_ids,
        min_overlap=min_overlap,
        min_component_fraction=min_component_fraction,
        do_morphological_cleanup=do_morphological_cleanup,
    )

    # ------------------------------------------------------------------
    # 8. Compute Dice *after* correction (optional)
    # ------------------------------------------------------------------
    dice_after = None
    if gt_path is not None:
        raw_after = dice_per_class(correction.corrected_labels, gt_labels, label_ids)
        dice_after = {LABEL_NAMES.get(k, str(k)): v for k, v in raw_after.items()}

    # ------------------------------------------------------------------
    # 9. Save output (optional)
    # ------------------------------------------------------------------
    if output_path is not None:
        save_nifti(correction.corrected_labels, affine, header, output_path)

    return AtlasPipelineResult(
        corrected_labels=correction.corrected_labels,
        affine=affine,
        header=header,
        atlas_case_id=atlas_entry.case_id,
        atlas_disease_name=atlas_entry.disease_name,
        mode=mode,
        was_relabeled=correction.was_relabeled,
        mapping_applied=correction.mapping_applied,
        reassignment_summary=correction.reassignment_summary,
        overlap_matrix=correction.overlap_matrix,
        label_ids=correction.label_ids,
        component_info=correction.component_info,
        dice_before=dice_before,
        dice_after=dice_after,
    )


# ---------------------------------------------------------------------------
# Folder pipeline
# ---------------------------------------------------------------------------

def run_atlas_folder_pipeline(
    pred_folder:        str | Path,
    gt_folder:          str | Path,
    output_folder:      str | Path,
    disease_map_path:   Optional[str | Path] = None,
    gt_folder_eval:     Optional[str | Path] = None,
    mode:               str = "baseline",
    seed:               int = 42,
    registration_mode:  str = "centroid",
    file_pattern:       str = "*.nii.gz",
    **kwargs,
) -> pd.DataFrame:
    """Run the atlas pipeline on every prediction in *pred_folder*.

    Parameters
    ----------
    pred_folder : folder of nnU-Net prediction NIfTI files.
    gt_folder : training GT folder (atlas library).
    output_folder : destination for corrected NIfTI files.
    disease_map_path : path to ``disease_map.json``.
    gt_folder_eval : if provided, per-case GT is looked up here for
                     Dice evaluation.
    mode : ``"baseline"`` or ``"disease_specific"``.
    seed : base random seed (each case gets a derived seed for reproducibility).
    **kwargs : forwarded to :func:`run_atlas_pipeline`.

    Returns
    -------
    DataFrame with one row per case containing Dice before/after and a
    summary of label changes.
    """
    pred_folder   = Path(pred_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_folder.glob(file_pattern))
    if not pred_files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' in {pred_folder}")

    rows = []
    for idx, pred_path in enumerate(pred_files):
        case_id = resolve_case_id(pred_path.name)
        out_path = output_folder / pred_path.name

        # Optional GT lookup for evaluation
        gt_path = None
        if gt_folder_eval is not None:
            gt_folder_eval = Path(gt_folder_eval)
            for suffix in [".nii.gz", ".nii"]:
                cand = gt_folder_eval / f"{case_id}{suffix}"
                if cand.exists():
                    gt_path = cand
                    break
                cand_img = gt_folder_eval / f"{case_id}_image{suffix}"
                if cand_img.exists():
                    gt_path = cand_img
                    break

        try:
            result = run_atlas_pipeline(
                pred_path=pred_path,
                gt_folder=gt_folder,
                output_path=out_path,
                disease_map_path=disease_map_path,
                gt_path=gt_path,
                mode=mode,
                seed=seed + idx,   # deterministic per-case seed
                registration_mode=registration_mode,
                **kwargs,
            )
            row = {"case_id": case_id, "status": "ok"}
            row.update(result.summary_dict())
        except Exception as exc:
            row = {"case_id": case_id, "status": "error", "error": str(exc)}
            print(f"  [ERROR] {case_id}: {exc}")

        rows.append(row)
        print(
            f"  [{idx + 1}/{len(pred_files)}] {case_id}: "
            f"atlas={row.get('atlas_case_id', '?')}  "
            f"relabeled={row.get('was_relabeled', '?')}"
        )

    df = pd.DataFrame(rows)
    return df
