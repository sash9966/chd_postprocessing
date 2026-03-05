"""AO/PA label correction via ventricle adjacency priors.

Core idea
---------
The aorta (AO, label 6) exits from the **left ventricle** (LV, label 1).
The pulmonary artery (PA, label 7) exits from the **right ventricle** (RV, label 2).

When the network swaps the two vessel labels the predicted "AO" region will
be spatially adjacent to the RV and the predicted "PA" region will be adjacent
to the LV — the reverse of the expected anatomy.  This module detects that
pattern and swaps the labels back.

Two correction levels are available:

``correct_ao_pa_labels`` — global correction
    Checks whether the *entire* AO/PA prediction is on the wrong side.
    Swaps all AO↔PA when both vessels are globally misassigned.

``correct_ao_pa_fragments`` — fragment-level correction (NEW)
    Operates on each connected component independently.  An AO fragment
    that is adjacent to RV (but not LV) is relabeled as PA, and vice versa.
    This handles the common case where the *main* AO/PA bodies are correct
    but a few spatially disconnected fragments have been given the wrong
    vessel label.  Requires no atlas registration.

Disease exception
-----------------
In **pulmonary atresia** (PuA, flag index 5 in the disease vector) the AO and PA
genuinely fuse or one vessel is absent, so the anatomical prior does not apply.
Cases with PuA=1 are left untouched by both functions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation

from .config import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_DILATION_RADIUS_MM,
    LABELS,
    PUA_FLAG_INDEX,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CorrectionResult:
    """Output of :func:`correct_ao_pa_labels`.

    Attributes
    ----------
    corrected_labels : modified (or unmodified) segmentation volume
    was_swapped : True if AO and PA labels were exchanged
    skipped_reason : human-readable reason when no correction was applied
                     (``None`` means correction ran without being skipped)
    confidence_score : 0 = completely ambiguous, 1 = highly confident.
                       Computed as the mean absolute deviation of each vessel's
                       adjacency score from 0.5 (chance level), scaled to [0, 1].
    needs_manual_review : True if confidence_score < threshold or the signal
                          was contradictory
    adjacency_details : raw adjacency counts / scores for diagnostics
    """
    corrected_labels: np.ndarray
    was_swapped: bool
    skipped_reason: Optional[str]
    confidence_score: float
    needs_manual_review: bool
    adjacency_details: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_ellipsoid_se(
    radius_mm: float,
    spacing_mm: Tuple[float, ...],
) -> np.ndarray:
    """Build an ellipsoidal binary structuring element.

    The element accounts for anisotropic voxel spacing so that the dilation
    corresponds to a sphere of *radius_mm* millimetres in physical space.

    Parameters
    ----------
    radius_mm : desired dilation radius in mm
    spacing_mm : voxel spacing along each axis (must match ndim of labels)
    """
    radii_vox = [radius_mm / s for s in spacing_mm]
    half = [math.ceil(r) for r in radii_vox]
    # Build using open grid for efficiency (no Python-level loop over voxels)
    grids = np.ogrid[tuple(slice(-h, h + 1) for h in half)]
    dist_sq = sum((g / r) ** 2 for g, r in zip(grids, radii_vox))
    return dist_sq <= 1.0


def _adjacency_scores(
    vessel_mask: np.ndarray,
    se: np.ndarray,
    lv_mask: np.ndarray,
    rv_mask: np.ndarray,
) -> Tuple[float, float, int, int]:
    """Compute how much a dilated vessel mask overlaps with LV vs RV.

    Returns
    -------
    lv_score : fraction of LV+RV overlap that falls on LV side
    rv_score : 1 - lv_score
    lv_count : raw voxel overlap with LV
    rv_count : raw voxel overlap with RV
    """
    dilated = binary_dilation(vessel_mask, structure=se)
    lv_count = int(np.sum(dilated & lv_mask))
    rv_count = int(np.sum(dilated & rv_mask))
    total = lv_count + rv_count
    if total == 0:
        # No overlap at all — return chance level
        return 0.5, 0.5, 0, 0
    lv_score = lv_count / total
    return lv_score, 1.0 - lv_score, lv_count, rv_count


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def correct_ao_pa_labels(
    labels: np.ndarray,
    disease_vec: Optional[List[int]],
    spacing_mm: Tuple[float, float, float],
    dilation_radius_mm: float = DEFAULT_DILATION_RADIUS_MM,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> CorrectionResult:
    """Correct AO/PA label swaps using the ventricle-adjacency anatomical prior.

    Parameters
    ----------
    labels : integer segmentation volume (shape H × W × D, dtype int)
    disease_vec : binary disease flags ``[HLHS, ASD, VSD, AVSD, DORV, PuA, ToF, TGA]``,
                  or ``None`` (treated as all-zeros = no known disease).
    spacing_mm : voxel spacing along each axis in millimetres.
    dilation_radius_mm : radius of the sphere used to test adjacency.
                         Default 3 mm — enough to bridge 2–3 voxel gaps at
                         typical cardiac CT spacing.
    confidence_threshold : cases with ``confidence_score`` below this value
                           are flagged for manual review.

    Returns
    -------
    :class:`CorrectionResult`
    """
    labels = labels.copy()

    # ------------------------------------------------------------------
    # 1. Skip pulmonary atresia cases — AO/PA fusion is expected.
    # ------------------------------------------------------------------
    if disease_vec is not None and disease_vec[PUA_FLAG_INDEX] == 1:
        return CorrectionResult(
            corrected_labels=labels,
            was_swapped=False,
            skipped_reason="PuA=1: AO/PA fusion is anatomically expected; correction skipped",
            confidence_score=1.0,
            needs_manual_review=False,
            adjacency_details={},
        )

    ao_lbl, pa_lbl = LABELS["AO"], LABELS["PA"]
    lv_lbl, rv_lbl = LABELS["LV"], LABELS["RV"]

    ao_mask = labels == ao_lbl
    pa_mask = labels == pa_lbl
    lv_mask = labels == lv_lbl
    rv_mask = labels == rv_lbl

    # ------------------------------------------------------------------
    # 2. Edge cases — missing structures.
    # ------------------------------------------------------------------
    if not ao_mask.any() or not pa_mask.any():
        return CorrectionResult(
            corrected_labels=labels,
            was_swapped=False,
            skipped_reason="AO or PA has zero voxels; cannot determine adjacency",
            confidence_score=0.0,
            needs_manual_review=True,
            adjacency_details={
                "ao_voxels": int(ao_mask.sum()),
                "pa_voxels": int(pa_mask.sum()),
            },
        )

    if not lv_mask.any() or not rv_mask.any():
        return CorrectionResult(
            corrected_labels=labels,
            was_swapped=False,
            skipped_reason="LV or RV has zero voxels; cannot determine adjacency",
            confidence_score=0.0,
            needs_manual_review=True,
            adjacency_details={
                "lv_voxels": int(lv_mask.sum()),
                "rv_voxels": int(rv_mask.sum()),
            },
        )

    # ------------------------------------------------------------------
    # 3. Build structuring element (physical-space sphere).
    # ------------------------------------------------------------------
    se = _make_ellipsoid_se(dilation_radius_mm, spacing_mm)

    # ------------------------------------------------------------------
    # 4. Compute adjacency scores.
    # ------------------------------------------------------------------
    ao_lv_score, ao_rv_score, ao_lv_cnt, ao_rv_cnt = _adjacency_scores(
        ao_mask, se, lv_mask, rv_mask
    )
    pa_lv_score, pa_rv_score, pa_lv_cnt, pa_rv_cnt = _adjacency_scores(
        pa_mask, se, lv_mask, rv_mask
    )

    # "Correct" means:  AO is more adjacent to LV  (ao_lv_score > 0.5)
    #                   PA is more adjacent to RV  (pa_rv_score > 0.5)
    ao_correct = ao_lv_score >= ao_rv_score
    pa_correct = pa_rv_score >= pa_lv_score

    # Confidence: mean distance from the ambiguous midpoint (0.5), scaled to [0,1]
    ao_confidence = abs(ao_lv_score - 0.5) * 2   # 0 = chance, 1 = certain
    pa_confidence = abs(pa_rv_score - 0.5) * 2
    confidence_score = float((ao_confidence + pa_confidence) / 2)
    needs_review = confidence_score < confidence_threshold

    adjacency_details: Dict = {
        "ao_lv_count":    ao_lv_cnt,
        "ao_rv_count":    ao_rv_cnt,
        "ao_lv_score":    round(ao_lv_score, 4),   # high → AO near LV (correct)
        "ao_rv_score":    round(ao_rv_score, 4),
        "pa_lv_count":    pa_lv_cnt,
        "pa_rv_count":    pa_rv_cnt,
        "pa_lv_score":    round(pa_lv_score, 4),
        "pa_rv_score":    round(pa_rv_score, 4),   # high → PA near RV (correct)
        "ao_confidence":  round(ao_confidence, 4),
        "pa_confidence":  round(pa_confidence, 4),
    }

    # ------------------------------------------------------------------
    # 5. Decide: swap, keep, or flag.
    # ------------------------------------------------------------------
    if ao_correct and pa_correct:
        # Both vessels are on the anatomically correct side — no action needed.
        return CorrectionResult(
            corrected_labels=labels,
            was_swapped=False,
            skipped_reason="Labels already anatomically consistent",
            confidence_score=confidence_score,
            needs_manual_review=needs_review,
            adjacency_details=adjacency_details,
        )

    if not ao_correct and not pa_correct:
        # Both vessels are on the wrong side — swap the labels.
        labels[ao_mask] = pa_lbl
        labels[pa_mask] = ao_lbl
        return CorrectionResult(
            corrected_labels=labels,
            was_swapped=True,
            skipped_reason=None,
            confidence_score=confidence_score,
            needs_manual_review=needs_review,
            adjacency_details=adjacency_details,
        )

    # Mixed signal: one vessel looks correct, the other doesn't.
    # Safer to leave the case untouched and flag for manual review.
    return CorrectionResult(
        corrected_labels=labels,
        was_swapped=False,
        skipped_reason="Mixed adjacency signal (one vessel consistent, one not); flagged for review",
        confidence_score=confidence_score,
        needs_manual_review=True,
        adjacency_details=adjacency_details,
    )


# ---------------------------------------------------------------------------
# Fragment-level AO/PA correction (new)
# ---------------------------------------------------------------------------

def correct_ao_pa_fragments(
    labels: np.ndarray,
    disease_vec: Optional[List[int]],
    spacing_mm: Tuple[float, float, float],
    dilation_radius_mm: float = DEFAULT_DILATION_RADIUS_MM,
) -> Tuple[np.ndarray, Dict]:
    """Per-component AO/PA correction using ventricle-adjacency anatomy.

    For each connected component of AO (label 6) and PA (label 7):
    - An AO component adjacent primarily to RV (and not LV) is relabeled as PA.
    - A PA component adjacent primarily to LV (and not RV) is relabeled as AO.

    Components with no ventricle adjacency signal (e.g., a stray RA fragment
    that nnU-Net mislabeled as AO) are left as-is; the atlas step handles those.

    Unlike ``correct_ao_pa_labels``, this operates independently on each
    disconnected fragment rather than the global vessel masks.  No registration
    is needed — only binary dilation and voxel counting.

    Parameters
    ----------
    labels : integer segmentation volume.
    disease_vec : binary disease flags.  PuA=1 → function is a no-op.
    spacing_mm : voxel spacing in mm.
    dilation_radius_mm : adjacency test radius.

    Returns
    -------
    (corrected_labels, fragment_log) where fragment_log is a list of dicts
    describing each reassigned fragment.
    """
    labels = labels.copy()
    fragment_log: Dict = {"reassigned": [], "skipped_pua": False}

    if disease_vec is not None and disease_vec[PUA_FLAG_INDEX] == 1:
        fragment_log["skipped_pua"] = True
        return labels, fragment_log

    from scipy.ndimage import label as nd_label

    ao_lbl, pa_lbl = LABELS["AO"], LABELS["PA"]
    lv_lbl, rv_lbl = LABELS["LV"], LABELS["RV"]

    lv_mask = labels == lv_lbl
    rv_mask = labels == rv_lbl

    if not lv_mask.any() or not rv_mask.any():
        return labels, fragment_log

    se = _make_ellipsoid_se(dilation_radius_mm, spacing_mm)

    for vessel_lbl, correct_ventricle, wrong_ventricle in [
        (ao_lbl, lv_mask, rv_mask),   # AO should be near LV
        (pa_lbl, rv_mask, lv_mask),   # PA should be near RV
    ]:
        target_lbl = pa_lbl if vessel_lbl == ao_lbl else ao_lbl
        vessel_mask = labels == vessel_lbl
        if not vessel_mask.any():
            continue

        labeled_vol, n = nd_label(vessel_mask)
        for cid in range(1, n + 1):
            comp_mask = labeled_vol == cid
            comp_size = int(comp_mask.sum())

            lv_score, rv_score, lv_cnt, rv_cnt = _adjacency_scores(
                comp_mask, se, lv_mask, rv_mask
            )

            correct_score = lv_score if vessel_lbl == ao_lbl else rv_score
            wrong_score   = rv_score if vessel_lbl == ao_lbl else lv_score

            # No ventricle signal at all — skip (stray fragment, atlas handles it)
            if lv_cnt == 0 and rv_cnt == 0:
                continue

            # Only reassign if the wrong ventricle dominates clearly
            if wrong_score > correct_score:
                labels[comp_mask] = target_lbl
                fragment_log["reassigned"].append({
                    "original_label":  vessel_lbl,
                    "assigned_label":  target_lbl,
                    "size":            comp_size,
                    "correct_score":   round(correct_score, 4),
                    "wrong_score":     round(wrong_score, 4),
                    "lv_count":        lv_cnt,
                    "rv_count":        rv_cnt,
                })

    return labels, fragment_log
