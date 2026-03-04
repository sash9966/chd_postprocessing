"""Atlas-guided label correction via overlap matrix and Hungarian matching.

After the atlas is registered into the prediction's coordinate frame, we
compare the predicted labels with the atlas labels spatially.  A Dice-based
overlap matrix captures how well each predicted label region corresponds to
each atlas label region.  The Hungarian algorithm then finds the globally
optimal bijective relabelling that maximises total overlap.

If the optimal mapping is the identity (each predicted label already matches
the atlas), no changes are made.  Otherwise, only the subset of labels that
are genuinely swapped is relabelled, and the result is passed through a
connected-component cleanup step to remove small spurious fragments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, label as nd_label
from scipy.optimize import linear_sum_assignment

from .config import FOREGROUND_CLASSES, LABEL_NAMES, MIN_COMPONENT_FRACTION


# ---------------------------------------------------------------------------
# Overlap matrix
# ---------------------------------------------------------------------------

def compute_overlap_matrix(
    pred:       np.ndarray,
    atlas_reg:  np.ndarray,
    label_ids:  Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Compute a Dice-based overlap matrix between *pred* and *atlas_reg*.

    Parameters
    ----------
    pred : predicted integer label volume.
    atlas_reg : atlas label volume registered into *pred*'s space.
    label_ids : foreground class IDs to consider.  Default: 1–7.

    Returns
    -------
    M : (N, N) float array where M[i, j] = Dice(pred == label_ids[i],
        atlas_reg == label_ids[j]).  A high value means the region predicted
        as label_ids[i] spatially overlaps with the atlas region label_ids[j].
    label_ids : the label IDs that were used (same list as input, for reference).
    """
    if label_ids is None:
        label_ids = list(FOREGROUND_CLASSES)
    N = len(label_ids)
    M = np.zeros((N, N), dtype=float)

    for i, p_lbl in enumerate(label_ids):
        pred_mask = pred == p_lbl
        pred_sum  = int(pred_mask.sum())
        for j, a_lbl in enumerate(label_ids):
            atlas_mask = atlas_reg == a_lbl
            atlas_sum  = int(atlas_mask.sum())
            denom = pred_sum + atlas_sum
            if denom == 0:
                M[i, j] = float("nan")
            else:
                inter = int((pred_mask & atlas_mask).sum())
                M[i, j] = 2.0 * inter / denom

    return M, label_ids


# ---------------------------------------------------------------------------
# Optimal label mapping
# ---------------------------------------------------------------------------

def optimal_label_mapping(
    M:          np.ndarray,
    label_ids:  List[int],
    min_overlap: float = 0.01,
) -> Dict[int, int]:
    """Find the label reassignment that maximises total Dice overlap.

    Uses the Hungarian algorithm (:func:`scipy.optimize.linear_sum_assignment`)
    on the overlap matrix.

    Only labels that have *some* presence (at least one finite, non-zero entry
    in their row or column of *M*) participate in the assignment.  Labels with
    no overlap anywhere are mapped to themselves (identity).

    Parameters
    ----------
    M : (N, N) Dice overlap matrix from :func:`compute_overlap_matrix`.
    label_ids : ordered list of label IDs corresponding to rows/cols of *M*.
    min_overlap : diagonal Dice value below which we treat a label as absent
                  in the prediction; these labels are excluded from reassignment.

    Returns
    -------
    mapping : dict ``{pred_label: atlas_label}`` for all label_ids.  Labels
              not reassigned map to themselves.
    """
    N = len(label_ids)
    mapping = {lbl: lbl for lbl in label_ids}     # start with identity

    # Identify which labels are meaningfully present
    cost = M.copy()
    cost[np.isnan(cost)] = 0.0

    # Rows (pred labels) with any non-trivial overlap
    active_rows = [i for i in range(N) if cost[i, :].max() > min_overlap]
    # Cols (atlas labels) with any non-trivial overlap
    active_cols = [j for j in range(N) if cost[:, j].max() > min_overlap]

    active = sorted(set(active_rows) | set(active_cols))
    if len(active) < 2:
        return mapping   # nothing to swap

    sub_cost = cost[np.ix_(active, active)]
    row_idx, col_idx = linear_sum_assignment(-sub_cost)   # maximise overlap

    for ri, ci in zip(row_idx, col_idx):
        pred_lbl  = label_ids[active[ri]]
        atlas_lbl = label_ids[active[ci]]
        mapping[pred_lbl] = atlas_lbl

    return mapping


# ---------------------------------------------------------------------------
# Apply mapping
# ---------------------------------------------------------------------------

def apply_label_mapping(
    labels:  np.ndarray,
    mapping: Dict[int, int],
) -> Tuple[np.ndarray, bool]:
    """Relabel *labels* according to *mapping*.

    Parameters
    ----------
    labels : integer label volume.
    mapping : dict ``{old_label: new_label}``.

    Returns
    -------
    relabelled : new label array (copy).
    was_changed : True if any label was actually changed.
    """
    # Check if the mapping differs from identity
    non_identity = {k: v for k, v in mapping.items() if k != v}
    if not non_identity:
        return labels.copy(), False

    out = labels.copy()
    # Work on a separate scratch buffer to avoid overwriting labels we still need
    scratch = labels.copy()
    for old_lbl, new_lbl in non_identity.items():
        out[scratch == old_lbl] = new_lbl

    return out, True


# ---------------------------------------------------------------------------
# Structural / topological cleanup
# ---------------------------------------------------------------------------

def enforce_single_component(
    labels:                np.ndarray,
    label_ids:             Optional[List[int]] = None,
    min_component_fraction: float = MIN_COMPONENT_FRACTION,
) -> Tuple[np.ndarray, Dict]:
    """Remove small disconnected fragments from the specified label classes.

    For each label in *label_ids*, connected-component analysis is run.
    Any component whose voxel count is less than
    *min_component_fraction* × (size of the largest component) is set to 0
    (background).

    Returns
    -------
    cleaned : label array with small fragments removed (copy).
    info : ``{label_id: {"n_components", "removed", "kept", "sizes"}}``
    """
    if label_ids is None:
        label_ids = list(FOREGROUND_CLASSES)

    cleaned = labels.copy()
    info: Dict = {}

    for lbl in label_ids:
        mask = cleaned == lbl
        if not mask.any():
            info[lbl] = {"n_components": 0, "removed": 0, "kept": 0, "sizes": []}
            continue

        labeled_vol, n = nd_label(mask)
        sizes = [int((labeled_vol == c).sum()) for c in range(1, n + 1)]
        largest = max(sizes)
        threshold = min_component_fraction * largest

        removed = 0
        for comp_idx, size in enumerate(sizes, start=1):
            if size < threshold:
                cleaned[labeled_vol == comp_idx] = 0
                removed += 1

        info[lbl] = {
            "n_components": n,
            "removed":      removed,
            "kept":         n - removed,
            "sizes":        sorted(sizes, reverse=True),
        }

    return cleaned, info


def apply_morphological_cleanup(
    labels:    np.ndarray,
    label_ids: Optional[List[int]] = None,
    closing_iters: int = 1,
) -> np.ndarray:
    """Apply binary closing and hole-filling to each specified label class.

    Closing (dilation followed by erosion) bridges small gaps between nearby
    fragments of the same label.  Hole-filling removes enclosed voids.

    Parameters
    ----------
    labels : integer label volume.
    label_ids : labels to clean.  Default: AO (6) and PA (7) only, as these
                are the structures most prone to fragmentation.
    closing_iters : number of binary closing iterations.

    Returns
    -------
    Cleaned label array (copy).
    """
    if label_ids is None:
        # Default: only the two vessel labels most prone to fragmentation
        from .config import LABELS
        label_ids = [LABELS["AO"], LABELS["PA"]]

    out = labels.copy()
    for lbl in label_ids:
        mask = out == lbl
        if not mask.any():
            continue
        closed = binary_closing(mask, iterations=closing_iters)
        filled = binary_fill_holes(closed)
        # Add back voxels recovered by closing/filling (don't overwrite other labels)
        new_voxels = filled & ~mask
        # Only assign to background voxels to avoid overwriting other structures
        out[new_voxels & (out == 0)] = lbl

    return out


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LabelCorrectionResult:
    """Output of the overlap-based atlas label correction.

    Attributes
    ----------
    corrected_labels : integer label array after relabelling + cleanup.
    overlap_matrix : (N, N) Dice matrix (rows = pred labels, cols = atlas labels).
    label_ids : class IDs corresponding to matrix rows/cols.
    mapping_applied : dict ``{pred_lbl → atlas_lbl}`` (may be identity).
    was_relabeled : True when at least one label was actually changed.
    component_info : per-label CC statistics after cleanup.
    reassignment_summary : human-readable description of what changed.
    """
    corrected_labels:     np.ndarray
    overlap_matrix:       np.ndarray
    label_ids:            List[int]
    mapping_applied:      Dict[int, int]
    was_relabeled:        bool
    component_info:       Dict
    reassignment_summary: str = ""


# ---------------------------------------------------------------------------
# High-level correction entry point
# ---------------------------------------------------------------------------

def correct_labels_with_atlas(
    pred:       np.ndarray,
    atlas_reg:  np.ndarray,
    label_ids:  Optional[List[int]] = None,
    min_overlap: float = 0.01,
    min_component_fraction: float = MIN_COMPONENT_FRACTION,
    do_morphological_cleanup: bool = True,
) -> LabelCorrectionResult:
    """Full atlas-guided label correction pipeline.

    1. Compute Dice overlap matrix between *pred* and registered atlas.
    2. Find optimal bijective label mapping (Hungarian algorithm).
    3. Apply relabelling.
    4. Remove small disconnected fragments.
    5. Optionally apply morphological closing + hole-filling.

    Parameters
    ----------
    pred : predicted integer label volume.
    atlas_reg : atlas registered into *pred*'s space.
    label_ids : foreground labels to consider.  Default: 1–7.
    min_overlap : minimum Dice for a label to participate in reassignment.
    min_component_fraction : CC fragment threshold.
    do_morphological_cleanup : whether to apply closing + hole-fill.

    Returns
    -------
    :class:`LabelCorrectionResult`
    """
    if label_ids is None:
        label_ids = list(FOREGROUND_CLASSES)

    # Step 1: overlap matrix
    M, lbl_ids = compute_overlap_matrix(pred, atlas_reg, label_ids)

    # Step 2: optimal mapping
    mapping = optimal_label_mapping(M, lbl_ids, min_overlap=min_overlap)

    # Step 3: apply relabelling
    corrected, was_relabeled = apply_label_mapping(pred, mapping)

    # Step 4: CC cleanup (all foreground labels)
    corrected, cc_info = enforce_single_component(
        corrected, label_ids, min_component_fraction
    )

    # Step 5: morphological cleanup
    if do_morphological_cleanup:
        corrected = apply_morphological_cleanup(corrected)

    # Build human-readable summary
    swaps = [
        f"  {LABEL_NAMES.get(k, k)} → {LABEL_NAMES.get(v, v)}"
        for k, v in mapping.items() if k != v
    ]
    if swaps:
        summary = "Label reassignments:\n" + "\n".join(swaps)
    else:
        summary = "No label reassignments (mapping is identity)."

    # Append CC cleanup stats
    cc_lines = []
    for lbl, st in cc_info.items():
        if st["n_components"] > 1:
            cc_lines.append(
                f"  {LABEL_NAMES.get(lbl, lbl)}: "
                f"{st['n_components']} components, {st['removed']} removed"
            )
    if cc_lines:
        summary += "\nCC cleanup:\n" + "\n".join(cc_lines)

    return LabelCorrectionResult(
        corrected_labels=corrected,
        overlap_matrix=M,
        label_ids=lbl_ids,
        mapping_applied=mapping,
        was_relabeled=was_relabeled,
        component_info=cc_info,
        reassignment_summary=summary,
    )
