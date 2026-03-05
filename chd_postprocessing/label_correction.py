"""Atlas-guided label correction at the connected-component level.

Design
------
The previous approach treated each predicted label as a monolithic unit and
applied a global bijective permutation via Hungarian matching.  This fails
when a structure has disconnected fragments with wrong class labels — e.g.,
some aorta voxels labelled as RA — because the permutation operates on
whole-label aggregates and cannot fix individual fragments independently.

New algorithm
-------------
Phase 1 — Component-level atlas-guided assignment
  1. For each predicted label class find its connected components independently,
     so every spatially disconnected piece is treated as a separate candidate.
  2. Precompute the atlas label masks once for efficiency.
  3. For each component compute Dice overlap against every atlas label region.
  4. Assign each component to the atlas label with the highest Dice overlap.
     If a component has zero overlap with every atlas label (registration miss),
     it keeps its original predicted label.
  5. Build the output volume by writing each component's assigned label.

Phase 2 — Anatomical constraint enforcement
  6. For each final label that ends up with multiple assigned components,
     keep the largest and reassign smaller fragments (below
     *min_component_fraction* × largest) to their next-best atlas match.
     Iterate until stable (converges in O(n_components) passes).
  7. Apply morphological closing only after the component-level assignment
     is complete.

Backward compatibility
----------------------
``compute_overlap_matrix``, ``optimal_label_mapping``, ``apply_label_mapping``,
``enforce_single_component``, and ``apply_morphological_cleanup`` are preserved
unchanged so existing callers and tests continue to work.  ``LabelCorrectionResult``
gains a new ``component_assignments`` field; all existing fields are kept.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, label as nd_label
from scipy.optimize import linear_sum_assignment

from .config import FOREGROUND_CLASSES, LABEL_NAMES, MIN_COMPONENT_FRACTION


# ---------------------------------------------------------------------------
# Legacy: whole-label overlap matrix (kept for backward compat / tests)
# ---------------------------------------------------------------------------

def compute_overlap_matrix(
    pred:       np.ndarray,
    atlas_reg:  np.ndarray,
    label_ids:  Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Compute a Dice-based overlap matrix between *pred* and *atlas_reg*.

    Returns
    -------
    M : (N, N) float array where M[i, j] = Dice(pred == label_ids[i],
        atlas_reg == label_ids[j]).
    label_ids : the label IDs used (same as input).
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
# Legacy: Hungarian whole-label mapping (kept for backward compat / tests)
# ---------------------------------------------------------------------------

def optimal_label_mapping(
    M:           np.ndarray,
    label_ids:   List[int],
    min_overlap: float = 0.01,
) -> Dict[int, int]:
    """Find the bijective label reassignment that maximises total Dice overlap.

    Uses the Hungarian algorithm on the (N, N) whole-label overlap matrix.
    Labels absent from both pred and atlas (all-NaN or near-zero row/column)
    are mapped to themselves.
    """
    N = len(label_ids)
    mapping = {lbl: lbl for lbl in label_ids}

    cost = M.copy()
    cost[np.isnan(cost)] = 0.0

    active_rows = [i for i in range(N) if cost[i, :].max() > min_overlap]
    active_cols = [j for j in range(N) if cost[:, j].max() > min_overlap]
    active = sorted(set(active_rows) | set(active_cols))
    if len(active) < 2:
        return mapping

    sub_cost = cost[np.ix_(active, active)]
    row_idx, col_idx = linear_sum_assignment(-sub_cost)

    for ri, ci in zip(row_idx, col_idx):
        pred_lbl  = label_ids[active[ri]]
        atlas_lbl = label_ids[active[ci]]
        mapping[pred_lbl] = atlas_lbl

    return mapping


# ---------------------------------------------------------------------------
# Legacy: apply a whole-label mapping (kept for backward compat / tests)
# ---------------------------------------------------------------------------

def apply_label_mapping(
    labels:  np.ndarray,
    mapping: Dict[int, int],
) -> Tuple[np.ndarray, bool]:
    """Relabel *labels* according to *mapping*.

    Returns (relabelled array, was_changed).
    """
    non_identity = {k: v for k, v in mapping.items() if k != v}
    if not non_identity:
        return labels.copy(), False

    out     = labels.copy()
    scratch = labels.copy()
    for old_lbl, new_lbl in non_identity.items():
        out[scratch == old_lbl] = new_lbl

    return out, True


# ---------------------------------------------------------------------------
# Structural cleanup (unchanged)
# ---------------------------------------------------------------------------

def enforce_single_component(
    labels:                 np.ndarray,
    label_ids:              Optional[List[int]] = None,
    min_component_fraction: float = MIN_COMPONENT_FRACTION,
) -> Tuple[np.ndarray, Dict]:
    """Remove small disconnected fragments from each label class.

    Any component whose voxel count is less than
    *min_component_fraction* × (size of the largest component for that label)
    is set to background (0).

    Returns (cleaned array, info dict per label).
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
        largest   = max(sizes)
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
    labels:        np.ndarray,
    label_ids:     Optional[List[int]] = None,
    closing_iters: int = 1,
) -> np.ndarray:
    """Apply binary closing and hole-filling to each specified label class.

    Default target labels: AO (6) and PA (7) — the vessel structures most
    prone to fragmentation in nnU-Net outputs.
    """
    if label_ids is None:
        from .config import LABELS
        label_ids = [LABELS["AO"], LABELS["PA"]]

    out = labels.copy()
    for lbl in label_ids:
        mask = out == lbl
        if not mask.any():
            continue
        closed = binary_closing(mask, iterations=closing_iters)
        filled = binary_fill_holes(closed)
        new_voxels = filled & ~mask
        out[new_voxels & (out == 0)] = lbl

    return out


# ---------------------------------------------------------------------------
# Component-level assignment — new core logic
# ---------------------------------------------------------------------------

@dataclass
class ComponentAssignment:
    """Assignment record for a single connected component.

    Attributes
    ----------
    original_label : label the component had in the raw prediction.
    assigned_label : label assigned by atlas overlap matching.
    size : voxel count of this component.
    best_overlap : Dice score between this component and the assigned atlas region.
    was_reassigned : True when assigned_label != original_label.
    label_component_idx : index of this component within its original label's
                          component list (1-indexed, matches scipy.ndimage.label).
    """
    original_label:      int
    assigned_label:      int
    size:                int
    best_overlap:        float
    was_reassigned:      bool
    label_component_idx: int


def _find_all_components(
    pred:      np.ndarray,
    label_ids: List[int],
) -> List[Dict]:
    """Find every connected component for each predicted label class.

    Returns a list of dicts, one per component, with keys:
        mask            : boolean array (pred.shape) marking this component
        original_label  : integer label it had in *pred*
        size            : voxel count
        label_comp_idx  : component index within its original label (1-indexed)
    """
    components = []
    for lbl in label_ids:
        lbl_mask = pred == lbl
        if not lbl_mask.any():
            continue
        labeled_vol, n = nd_label(lbl_mask)
        for cid in range(1, n + 1):
            mask = labeled_vol == cid
            components.append({
                "mask":           mask,
                "original_label": lbl,
                "size":           int(mask.sum()),
                "label_comp_idx": cid,
            })
    return components


def _compute_component_overlaps(
    components:   List[Dict],
    atlas_masks:  Dict[int, np.ndarray],   # {atlas_lbl: bool mask}
    label_ids:    List[int],
) -> np.ndarray:
    """Compute overlap matrix of shape (n_components, n_labels).

    M[i, j] = |component_i ∩ atlas_j| / |component_i|  (intersection over component).

    Unlike Dice, this metric is not penalised by the atlas region being much
    larger than the component.  A 50-voxel fragment that lies entirely inside
    a 6400-voxel atlas LV-BP region gets a score of 1.0 rather than ~0.015,
    so argmax reliably picks the correct atlas label regardless of region size.
    """
    n_comps  = len(components)
    n_labels = len(label_ids)
    M = np.zeros((n_comps, n_labels), dtype=float)

    for i, comp in enumerate(components):
        comp_mask = comp["mask"]
        comp_size = comp["size"]
        if comp_size == 0:
            continue
        for j, lbl in enumerate(label_ids):
            inter  = int((comp_mask & atlas_masks[lbl]).sum())
            M[i, j] = inter / comp_size

    return M


def _initial_assignments(
    overlap_matrix: np.ndarray,
    components:     List[Dict],
    label_ids:      List[int],
) -> List[int]:
    """Assign each component to the atlas label with the highest Dice overlap.

    If a component has zero overlap with all atlas labels (registration miss),
    it keeps its original predicted label.
    """
    assignments = []
    for i, comp in enumerate(components):
        best_j = int(np.argmax(overlap_matrix[i]))
        if overlap_matrix[i, best_j] > 0.0:
            assignments.append(best_j)
        else:
            orig = comp["original_label"]
            assignments.append(label_ids.index(orig) if orig in label_ids else 0)
    return assignments


def _resolve_multi_component_conflicts(
    assignments:            List[int],
    components:             List[Dict],
    overlap_matrix:         np.ndarray,
    label_ids:              List[int],
    min_component_fraction: float,
) -> List[int]:
    """For each atlas label that has multiple assigned components, keep the
    largest; reassign smaller fragments (below *min_component_fraction* ×
    largest) to their next-best atlas match.  Iterates until stable.

    This enforces the anatomical prior that each structure should ideally
    correspond to one primary connected region.
    """
    from collections import defaultdict

    assignments = list(assignments)   # work on a copy
    n_labels    = len(label_ids)

    max_iters = len(components) + 1   # guaranteed upper bound
    for _ in range(max_iters):
        changed = False

        # Group components by their current assignment
        label_to_comps: Dict[int, List[int]] = defaultdict(list)
        for i, j in enumerate(assignments):
            label_to_comps[j].append(i)

        for j, comp_indices in label_to_comps.items():
            if len(comp_indices) <= 1:
                continue

            # Sort by size — largest first
            sorted_comps = sorted(comp_indices,
                                  key=lambda i: components[i]["size"],
                                  reverse=True)
            largest_size = components[sorted_comps[0]]["size"]
            threshold    = min_component_fraction * largest_size

            for comp_idx in sorted_comps[1:]:
                if components[comp_idx]["size"] >= threshold:
                    continue   # large enough to keep current assignment

                # Find next-best atlas label (excluding the contested one)
                row = overlap_matrix[comp_idx].copy()
                row[j] = -1.0   # exclude current assignment
                next_j = int(np.argmax(row))
                if row[next_j] > 0.0 and next_j != j:
                    assignments[comp_idx] = next_j
                    changed = True
                # If no better option exists, leave the assignment as-is

        if not changed:
            break

    return assignments


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class LabelCorrectionResult:
    """Output of the component-level atlas label correction.

    Attributes
    ----------
    corrected_labels      : integer label array after component reassignment
                            and cleanup.
    overlap_matrix        : (n_components, n_labels) Dice matrix — rows are
                            individual prediction components, columns are atlas
                            label regions.
    label_ids             : atlas label IDs corresponding to columns of
                            *overlap_matrix*.
    mapping_applied       : dominant label mapping per original label — for
                            each original label, what label did the majority
                            of its voxels end up as.  Backward-compatible
                            summary of the component-level assignments.
    was_relabeled         : True when at least one component was assigned a
                            different label from its original.
    component_info        : per-label CC statistics after cleanup (same format
                            as before, for backward compat).
    component_assignments : per-component detail records
                            (:class:`ComponentAssignment`).
    reassignment_summary  : human-readable description of what changed.
    """
    corrected_labels:      np.ndarray
    overlap_matrix:        np.ndarray          # (n_components, n_labels)
    label_ids:             List[int]
    mapping_applied:       Dict[int, int]
    was_relabeled:         bool
    component_info:        Dict
    component_assignments: List[ComponentAssignment]
    reassignment_summary:  str = ""


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
    """Component-level atlas-guided label correction.

    Parameters
    ----------
    pred : predicted integer label volume.
    atlas_reg : atlas registered into *pred*'s coordinate frame (no
                synthetic perturbation — the atlas is already a different
                patient).
    label_ids : foreground labels to consider.  Default: 1–7.
    min_overlap : minimum Dice for the best atlas match to be accepted;
                  components below this threshold keep their original label.
    min_component_fraction : fraction of the largest component below which a
                             fragment is considered a candidate for reassignment
                             during conflict resolution.
    do_morphological_cleanup : apply closing + hole-filling after assignment.

    Returns
    -------
    :class:`LabelCorrectionResult`
    """
    if label_ids is None:
        label_ids = list(FOREGROUND_CLASSES)

    # ------------------------------------------------------------------
    # Phase 1: component-level atlas-guided assignment
    # ------------------------------------------------------------------

    # 1a. Find all connected components across all predicted labels
    components = _find_all_components(pred, label_ids)

    if not components:
        # No foreground at all — return input unchanged
        empty_result = LabelCorrectionResult(
            corrected_labels=pred.copy(),
            overlap_matrix=np.zeros((0, len(label_ids))),
            label_ids=label_ids,
            mapping_applied={lbl: lbl for lbl in label_ids},
            was_relabeled=False,
            component_info={lbl: {"n_components": 0, "removed": 0, "kept": 0, "sizes": []}
                            for lbl in label_ids},
            component_assignments=[],
            reassignment_summary="No foreground voxels found.",
        )
        return empty_result

    # 1b. Precompute atlas label masks (once, reused across all components)
    atlas_masks: Dict[int, np.ndarray] = {lbl: (atlas_reg == lbl) for lbl in label_ids}

    # 1c. Intersection-over-component overlap: (n_components, n_labels)
    M = _compute_component_overlaps(components, atlas_masks, label_ids)

    # 1d. Conservative assignment: lock the dominant (largest) component for each
    #     original label; only apply IoC-based reassignment to extra fragments.
    #
    #     Rationale: the dominant component of each label in the nnU-Net prediction
    #     is almost certainly correct — the network has high overall accuracy.
    #     Reassigning large correct components via an imperfectly registered atlas
    #     introduces more errors than it fixes.  Only the secondary fragments (which
    #     are anatomically implausible as separate structures) need correction.
    largest_per_label: Dict[int, int] = {}  # {original_label: component_index}
    for i, comp in enumerate(components):
        lbl = comp["original_label"]
        if lbl not in largest_per_label or comp["size"] > components[largest_per_label[lbl]]["size"]:
            largest_per_label[lbl] = i

    assignments: List[int] = []
    for i, comp in enumerate(components):
        lbl = comp["original_label"]
        orig_j = label_ids.index(lbl)
        if largest_per_label.get(lbl) == i:
            # Dominant component — keep original label unconditionally
            assignments.append(orig_j)
        else:
            # Extra fragment — assign to best atlas match if signal is present
            best_j = int(np.argmax(M[i]))
            if M[i, best_j] >= min_overlap:
                assignments.append(best_j)
            else:
                assignments.append(orig_j)  # no atlas signal → keep original

    # ------------------------------------------------------------------
    # Phase 2: anatomical constraint enforcement
    # ------------------------------------------------------------------

    # 2a. Resolve conflicts among extra fragments only
    #     (dominant components are already locked and won't be moved)
    assignments = _resolve_multi_component_conflicts(
        assignments, components, M, label_ids, min_component_fraction
    )

    # 2b. Build output volume
    corrected = np.zeros_like(pred)
    for comp, j in zip(components, assignments):
        corrected[comp["mask"]] = label_ids[j]

    # 2c. Record per-component assignment info
    comp_assignments: List[ComponentAssignment] = []
    was_relabeled = False
    for i, (comp, j) in enumerate(zip(components, assignments)):
        assigned_lbl = label_ids[j]
        reassigned   = (assigned_lbl != comp["original_label"])
        if reassigned:
            was_relabeled = True
        comp_assignments.append(ComponentAssignment(
            original_label=     comp["original_label"],
            assigned_label=     assigned_lbl,
            size=               comp["size"],
            best_overlap=       float(M[i, j]),
            was_reassigned=     reassigned,
            label_component_idx=comp["label_comp_idx"],
        ))

    # 2d. Dominant mapping per original label (for backward compat)
    #     = what label did the largest component of each original label end up as
    mapping_applied: Dict[int, int] = {lbl: lbl for lbl in label_ids}
    for lbl in label_ids:
        lbl_comps = [(ca, comp) for ca, comp in zip(comp_assignments, components)
                     if comp["original_label"] == lbl]
        if lbl_comps:
            largest_ca = max(lbl_comps, key=lambda x: x[1]["size"])[0]
            mapping_applied[lbl] = largest_ca.assigned_label

    # 2e. Component info (diagnostic — based on assigned labels, no voxels removed)
    cc_info: Dict = {}
    for lbl in label_ids:
        lbl_cas = [ca for ca in comp_assignments if ca.assigned_label == lbl]
        cc_info[lbl] = {
            "n_components": len(lbl_cas),
            "removed":      0,
            "kept":         len(lbl_cas),
            "sizes":        sorted([ca.size for ca in lbl_cas], reverse=True),
        }

    # 2f. Morphological closing
    if do_morphological_cleanup:
        corrected = apply_morphological_cleanup(corrected)

    # ------------------------------------------------------------------
    # Build human-readable summary
    # ------------------------------------------------------------------
    reassigned_cas = [ca for ca in comp_assignments if ca.was_reassigned]
    if reassigned_cas:
        lines = ["Component reassignments:"]
        for ca in reassigned_cas:
            lines.append(
                f"  {LABEL_NAMES.get(ca.original_label, ca.original_label)}"
                f" component {ca.label_component_idx}"
                f" ({ca.size} vx)"
                f" → {LABEL_NAMES.get(ca.assigned_label, ca.assigned_label)}"
                f" (overlap={ca.best_overlap:.3f})"
            )
        summary = "\n".join(lines)
    else:
        summary = "No component reassignments."

    cc_lines = [
        f"  {LABEL_NAMES.get(lbl, lbl)}: {st['n_components']} components, "
        f"{st['removed']} removed"
        for lbl, st in cc_info.items() if st["n_components"] > 1
    ]
    if cc_lines:
        summary += "\nCC cleanup:\n" + "\n".join(cc_lines)

    return LabelCorrectionResult(
        corrected_labels=corrected,
        overlap_matrix=M,
        label_ids=label_ids,
        mapping_applied=mapping_applied,
        was_relabeled=was_relabeled,
        component_info=cc_info,
        component_assignments=comp_assignments,
        reassignment_summary=summary,
    )
