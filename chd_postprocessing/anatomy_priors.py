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

``correct_ao_pa_fragments`` — fragment-level correction
    Operates on each connected component independently.  An AO fragment
    that is adjacent to RV (but not LV) is relabeled as PA, and vice versa.
    This handles the common case where the *main* AO/PA bodies are correct
    but a few spatially disconnected fragments have been given the wrong
    vessel label.  Requires no atlas registration.

Disease-aware exceptions
------------------------
Each function checks :data:`~chd_postprocessing.config.DISEASE_ANATOMY_RULES`
before applying any correction:

* **HLHS** (flag 0) and **PuA** (flag 5): correction is skipped entirely
  (``skip_ao_pa_correction=True``).
* **TGA** (flag 7): AO expected near RV, PA near LV (vessels transposed).
* **DORV** (flag 4): both AO and PA exit the RV.
* **ToF** (flag 6): AO is unconstrained (overriding aorta straddles VSD);
  only PA is constrained to the RV.

When multiple disease flags are active the **most permissive** rule wins:
a vessel is unconstrained if *any* active disease flags it as unconstrained,
and a vessel's allowed ventricle set is the union across all active rules.
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
    DISEASE_ANATOMY_RULES,
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
    # 1. Check disease-specific skip conditions.
    # ------------------------------------------------------------------
    if disease_vec is not None:
        for flag_idx, is_active in enumerate(disease_vec):
            if is_active and flag_idx in DISEASE_ANATOMY_RULES:
                rule = DISEASE_ANATOMY_RULES[flag_idx]
                if rule.get("skip_ao_pa_correction", False):
                    return CorrectionResult(
                        corrected_labels=labels,
                        was_swapped=False,
                        skipped_reason=(
                            f"{rule['name']}=1: {rule['notes']}; correction skipped"
                        ),
                        confidence_score=1.0,
                        needs_manual_review=False,
                        adjacency_details={},
                    )

    ao_lbl, pa_lbl = LABELS["AO"], LABELS["PA"]
    lv_lbl, rv_lbl = LABELS["LV"], LABELS["RV"]

    # ------------------------------------------------------------------
    # 2. Resolve effective vessel→ventricle map from active disease flags.
    #    ao_expected_ventricle / pa_expected_ventricle:
    #      None  → unconstrained (skip adjacency check for this vessel)
    #      int   → the single ventricle label the vessel should be near
    # ------------------------------------------------------------------
    ao_expected_ventricle: Optional[int] = lv_lbl   # normal: AO → LV
    pa_expected_ventricle: Optional[int] = rv_lbl   # normal: PA → RV

    if disease_vec is not None:
        for flag_idx, is_active in enumerate(disease_vec):
            if is_active and flag_idx in DISEASE_ANATOMY_RULES:
                vv = DISEASE_ANATOMY_RULES[flag_idx].get("vessel_ventricle", {})
                if ao_lbl in vv:
                    val = vv[ao_lbl]
                    if val is None:
                        ao_expected_ventricle = None  # unconstrained
                    elif ao_expected_ventricle is not None:
                        ao_expected_ventricle = val   # override
                if pa_lbl in vv:
                    val = vv[pa_lbl]
                    if val is None:
                        pa_expected_ventricle = None
                    elif pa_expected_ventricle is not None:
                        pa_expected_ventricle = val

    # If both vessels are unconstrained there is nothing to check
    if ao_expected_ventricle is None and pa_expected_ventricle is None:
        return CorrectionResult(
            corrected_labels=labels,
            was_swapped=False,
            skipped_reason="Both vessels are unconstrained for this disease; correction skipped",
            confidence_score=1.0,
            needs_manual_review=False,
            adjacency_details={},
        )

    ao_mask = labels == ao_lbl
    pa_mask = labels == pa_lbl
    lv_mask = labels == lv_lbl
    rv_mask = labels == rv_lbl

    # ------------------------------------------------------------------
    # 3. Edge cases — missing structures.
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
    # 4. Build structuring element (physical-space sphere).
    # ------------------------------------------------------------------
    se = _make_ellipsoid_se(dilation_radius_mm, spacing_mm)

    # ------------------------------------------------------------------
    # 5. Compute adjacency scores.
    # ------------------------------------------------------------------
    ao_lv_score, ao_rv_score, ao_lv_cnt, ao_rv_cnt = _adjacency_scores(
        ao_mask, se, lv_mask, rv_mask
    )
    pa_lv_score, pa_rv_score, pa_lv_cnt, pa_rv_cnt = _adjacency_scores(
        pa_mask, se, lv_mask, rv_mask
    )

    adjacency_details: Dict = {
        "ao_lv_count":    ao_lv_cnt,
        "ao_rv_count":    ao_rv_cnt,
        "ao_lv_score":    round(ao_lv_score, 4),
        "ao_rv_score":    round(ao_rv_score, 4),
        "pa_lv_count":    pa_lv_cnt,
        "pa_rv_count":    pa_rv_cnt,
        "pa_lv_score":    round(pa_lv_score, 4),
        "pa_rv_score":    round(pa_rv_score, 4),
    }

    # ------------------------------------------------------------------
    # 6. Evaluate correctness using the disease-resolved ventricle map.
    # ------------------------------------------------------------------
    # ao_correct: True if AO is near its expected ventricle (or unconstrained)
    # pa_correct: True if PA is near its expected ventricle (or unconstrained)
    if ao_expected_ventricle is None:
        ao_correct = True
        ao_confidence = 1.0
    elif ao_expected_ventricle == lv_lbl:
        ao_correct = ao_lv_score >= ao_rv_score
        ao_confidence = abs(ao_lv_score - 0.5) * 2
    else:  # ao_expected_ventricle == rv_lbl (e.g. TGA, DORV)
        ao_correct = ao_rv_score >= ao_lv_score
        ao_confidence = abs(ao_rv_score - 0.5) * 2

    if pa_expected_ventricle is None:
        pa_correct = True
        pa_confidence = 1.0
    elif pa_expected_ventricle == rv_lbl:
        pa_correct = pa_rv_score >= pa_lv_score
        pa_confidence = abs(pa_rv_score - 0.5) * 2
    else:  # pa_expected_ventricle == lv_lbl (e.g. TGA)
        pa_correct = pa_lv_score >= pa_rv_score
        pa_confidence = abs(pa_lv_score - 0.5) * 2

    adjacency_details["ao_confidence"] = round(ao_confidence, 4)
    adjacency_details["pa_confidence"] = round(pa_confidence, 4)
    confidence_score = float((ao_confidence + pa_confidence) / 2)
    needs_review = confidence_score < confidence_threshold

    # ------------------------------------------------------------------
    # 7. Decide: swap, keep, or flag.
    # ------------------------------------------------------------------
    if ao_correct and pa_correct:
        return CorrectionResult(
            corrected_labels=labels,
            was_swapped=False,
            skipped_reason="Labels already anatomically consistent",
            confidence_score=confidence_score,
            needs_manual_review=needs_review,
            adjacency_details=adjacency_details,
        )

    if not ao_correct and not pa_correct:
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

    # Mixed signal — flag for review.
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
    fragment_log: Dict = {"reassigned": [], "skipped_disease": False}

    # ------------------------------------------------------------------
    # Check disease-specific skip conditions (HLHS, PuA, …)
    # ------------------------------------------------------------------
    if disease_vec is not None:
        for flag_idx, is_active in enumerate(disease_vec):
            if is_active and flag_idx in DISEASE_ANATOMY_RULES:
                rule = DISEASE_ANATOMY_RULES[flag_idx]
                if rule.get("skip_ao_pa_correction", False):
                    fragment_log["skipped_disease"] = True
                    return labels, fragment_log

    from scipy.ndimage import label as nd_label

    ao_lbl, pa_lbl = LABELS["AO"], LABELS["PA"]
    lv_lbl, rv_lbl = LABELS["LV"], LABELS["RV"]

    lv_mask = labels == lv_lbl
    rv_mask = labels == rv_lbl

    if not lv_mask.any() or not rv_mask.any():
        return labels, fragment_log

    se = _make_ellipsoid_se(dilation_radius_mm, spacing_mm)

    # ------------------------------------------------------------------
    # Build disease-resolved vessel→allowed_ventricles map.
    #   None = unconstrained (skip fragment correction for this vessel).
    #   set  = ventricle labels this vessel should be near.
    # ------------------------------------------------------------------
    # Defaults: AO → {LV}, PA → {RV}
    ao_allowed: Optional[set] = {lv_lbl}
    pa_allowed: Optional[set] = {rv_lbl}

    if disease_vec is not None:
        for flag_idx, is_active in enumerate(disease_vec):
            if is_active and flag_idx in DISEASE_ANATOMY_RULES:
                vv = DISEASE_ANATOMY_RULES[flag_idx].get("vessel_ventricle", {})
                if ao_lbl in vv:
                    val = vv[ao_lbl]
                    if val is None:
                        ao_allowed = None  # unconstrained
                    elif ao_allowed is not None:
                        ao_allowed = {val}
                if pa_lbl in vv:
                    val = vv[pa_lbl]
                    if val is None:
                        pa_allowed = None
                    elif pa_allowed is not None:
                        pa_allowed = {val}

    all_ventricles = {lv_lbl, rv_lbl}

    for vessel_lbl, allowed_vents in [
        (ao_lbl, ao_allowed),
        (pa_lbl, pa_allowed),
    ]:
        if allowed_vents is None:
            continue  # unconstrained — skip this vessel

        wrong_vents = all_ventricles - allowed_vents
        if not wrong_vents:
            continue  # no possible "wrong" ventricle

        # Masks for the correct and wrong ventricle side
        correct_vent_mask = np.zeros(labels.shape, dtype=bool)
        wrong_vent_mask   = np.zeros(labels.shape, dtype=bool)
        for v in allowed_vents:
            correct_vent_mask |= (labels == v)
        for v in wrong_vents:
            wrong_vent_mask |= (labels == v)

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

            # Aggregate scores for correct / wrong ventricle sets
            score_map = {lv_lbl: lv_score, rv_lbl: rv_score}
            correct_score = sum(score_map[v] for v in allowed_vents) / len(allowed_vents)
            wrong_score   = sum(score_map[v] for v in wrong_vents)   / len(wrong_vents)

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
