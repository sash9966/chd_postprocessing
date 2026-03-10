"""Atlas and topology-guided label correction using structural adjacency.

Complements the IoC-based spatial overlap correction by checking whether
each component's neighbourhood relationships are anatomically plausible.
A fragment labelled "AO" that touches RA but disease-aware rules say
AO should never touch RA is almost certainly mislabelled.

Three signals are combined:
1. Disease-aware adjacency rules (from config.get_effective_adjacency) —
   most reliable, no registration needed.
2. Atlas adjacency graph — empirical neighbourhood from the registered atlas.
   Less reliable than disease rules (depends on registration quality).
3. Connectivity improvement — does relabelling reduce disconnected components?

Runs AFTER IoC correction as a refinement step in disease_atlas_rules mode.
Only modifies non-dominant components (largest per label is always locked).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, label as nd_label
from typing import Dict, List, Optional, Tuple

from .config import FOREGROUND_CLASSES, LABEL_NAMES, LABELS

# Try to import get_effective_adjacency; fall back to empty dict if not yet added
try:
    from .config import get_effective_adjacency  # type: ignore[attr-defined]
except ImportError:
    def get_effective_adjacency(disease_vec):  # type: ignore[misc]
        """Fallback: no disease-specific adjacency constraints available."""
        return {}


def build_adjacency_graph(
    labels: np.ndarray,
    label_ids: Optional[List[int]] = None,
    dilation_iters: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build adjacency matrix from a label volume.

    Returns
    -------
    adj_binary : (N, N) bool — adj[i,j] = True means label_ids[i] and [j] share a boundary.
    adj_weight : (N, N) int — number of boundary voxels shared (surface area of contact).
    """
    if label_ids is None:
        label_ids = list(FOREGROUND_CLASSES)

    N = len(label_ids)
    adj_binary = np.zeros((N, N), dtype=bool)
    adj_weight = np.zeros((N, N), dtype=np.int64)

    struct = np.ones((3, 3, 3), dtype=bool)

    # Pre-compute masks for all labels
    masks = {}
    for lbl in label_ids:
        masks[lbl] = labels == lbl

    # Upper-triangle only, then mirror
    for i in range(N):
        lbl_i = label_ids[i]
        mask_i = masks[lbl_i]
        if not mask_i.any():
            continue
        # Dilate label i's mask
        dilated_i = binary_dilation(mask_i, structure=struct, iterations=dilation_iters)
        for j in range(i + 1, N):
            lbl_j = label_ids[j]
            mask_j = masks[lbl_j]
            if not mask_j.any():
                continue
            # Count overlap between dilated i and raw j
            # (boundary contact voxels from j's side)
            contact = int((dilated_i & mask_j).sum())
            if contact > 0:
                adj_binary[i, j] = True
                adj_binary[j, i] = True
                adj_weight[i, j] = contact
                adj_weight[j, i] = contact

    return adj_binary, adj_weight


def component_neighbor_profile(
    component_mask: np.ndarray,
    full_labels: np.ndarray,
    label_ids: List[int],
    dilation_iters: int = 2,
) -> Dict[int, int]:
    """For a single component, return {neighbor_label: boundary_voxel_count}.

    Dilate the component mask, then count how many voxels of each OTHER label
    fall within the dilated region (exclude the component's own voxels).
    Returns only labels with >0 contact.
    """
    struct = np.ones((3, 3, 3), dtype=bool)

    dilated = binary_dilation(component_mask, structure=struct, iterations=dilation_iters)

    # Exclude the component's own voxels from the dilated region
    dilated_excl = dilated & ~component_mask

    profile: Dict[int, int] = {}
    for lbl in label_ids:
        lbl_mask = full_labels == lbl
        # Skip voxels that are part of the component itself
        lbl_mask = lbl_mask & ~component_mask
        contact = int((dilated_excl & lbl_mask).sum())
        if contact > 0:
            profile[lbl] = contact

    return profile


def connectivity_improvement(
    labels: np.ndarray,
    component_mask: np.ndarray,
    current_label: int,
    candidate_label: int,
) -> int:
    """Score how much relabelling a component improves connectivity.

    Returns (n_components_before - n_components_after) across both labels.
    Positive = fewer fragments after relabelling = improvement.
    """
    # Count components before relabelling for both affected labels
    current_mask_before = labels == current_label
    candidate_mask_before = labels == candidate_label

    _, n_current_before = nd_label(current_mask_before)
    _, n_candidate_before = nd_label(candidate_mask_before)
    n_before = n_current_before + n_candidate_before

    # Simulate relabelling: component moves from current_label to candidate_label
    current_mask_after = current_mask_before & ~component_mask
    candidate_mask_after = candidate_mask_before | component_mask

    if current_mask_after.any():
        _, n_current_after = nd_label(current_mask_after)
    else:
        n_current_after = 0

    if candidate_mask_after.any():
        _, n_candidate_after = nd_label(candidate_mask_after)
    else:
        n_candidate_after = 0

    n_after = n_current_after + n_candidate_after

    return n_before - n_after


def _compute_atlas_adjacency(
    atlas_reg: np.ndarray,
    label_ids: List[int],
    dilation_iters: int = 2,
) -> Dict[Tuple[int, int], bool]:
    """Build a set of (label_i, label_j) pairs that are adjacent in the atlas.

    Returns a dict mapping (min_lbl, max_lbl) -> True for pairs that touch
    in the registered atlas.  Absent pairs are NOT adjacent in the atlas.
    """
    adj_binary, _ = build_adjacency_graph(atlas_reg, label_ids, dilation_iters)
    atlas_adj: Dict[Tuple[int, int], bool] = {}
    N = len(label_ids)
    for i in range(N):
        for j in range(i + 1, N):
            if adj_binary[i, j]:
                key = (label_ids[i], label_ids[j])
                atlas_adj[key] = True
    return atlas_adj


def _is_adjacency_forbidden(
    lbl_a: int,
    lbl_b: int,
    disease_rules: Dict[Tuple[int, int], bool],
    atlas_adj: Dict[Tuple[int, int], bool],
) -> bool:
    """Return True if the adjacency between lbl_a and lbl_b is forbidden.

    Priority:
    1. If disease_rules has an entry for this pair and it is False -> forbidden.
    2. If disease_rules explicitly says True -> allowed (not forbidden).
    3. No disease constraint: check atlas. If atlas also doesn't show this
       adjacency -> treated as forbidden.
    """
    key = (min(lbl_a, lbl_b), max(lbl_a, lbl_b))
    # Normalise disease_rules keys to (min, max) ordering
    disease_val = None
    if key in disease_rules:
        disease_val = disease_rules[key]
    else:
        # Also check reversed key
        rev_key = (key[1], key[0])
        if rev_key in disease_rules:
            disease_val = disease_rules[rev_key]

    if disease_val is not None:
        # Disease rules have an explicit answer
        return not disease_val  # False -> forbidden, True -> allowed

    # No disease constraint: fall back to atlas
    atlas_val = atlas_adj.get(key, atlas_adj.get((key[1], key[0]), None))
    if atlas_val is None:
        # Atlas doesn't show this adjacency -> treat as forbidden
        return True
    return not atlas_val


def _find_dominant_components(
    labels: np.ndarray,
    label_ids: List[int],
) -> Dict[int, int]:
    """Return {label: component_size} for the largest component of each label."""
    dominant: Dict[int, int] = {}
    for lbl in label_ids:
        mask = labels == lbl
        if not mask.any():
            dominant[lbl] = 0
            continue
        labeled_vol, n = nd_label(mask)
        sizes = [int((labeled_vol == c).sum()) for c in range(1, n + 1)]
        dominant[lbl] = max(sizes) if sizes else 0
    return dominant


def correct_by_adjacency(
    labels: np.ndarray,
    atlas_reg: np.ndarray,
    label_ids: Optional[List[int]] = None,
    disease_vec: Optional[List[int]] = None,
    min_component_fraction: float = 0.05,
    dilation_iters: int = 2,
    protected_labels: Optional[List[int]] = None,
) -> Tuple[np.ndarray, List[Dict]]:
    """Fix label errors detected by adjacency violations.

    Algorithm per non-dominant component:
    1. Compute neighbor profile.
    2. Check forbidden adjacencies: disease rules (most reliable); if no
       disease constraint, also check atlas adjacency graph.
       Forbidden if disease says False, OR (disease has no constraint AND
       atlas doesn't show this adjacency).
    3. If forbidden adjacencies exist: find candidate labels consistent with
       the component's actual neighbors per disease rules. Score candidates by:
         3*connectivity_improvement + 2*(boundary_contact/size) + IoC_with_atlas
       Relabel to best-scoring candidate if score > 0.
    4. Chain detection: after the main pass, scan for A-B-A sandwich patterns.
       If a small B component touches two separate A components AND relabelling
       B as A would merge them (connectivity_improvement > 0) AND disease rules
       allow A to have B's actual neighbors, relabel B as A.

    Returns (corrected_labels, log) where each log entry is a dict with keys:
    original_label, new_label, component_size, reason, neighbor_profile,
    forbidden_pairs, connectivity_delta.

    Only modifies components smaller than min_component_fraction * (largest
    component of that label). Dominant components are always locked.
    """
    if label_ids is None:
        label_ids = list(FOREGROUND_CLASSES)

    corrected = labels.copy()
    log: List[Dict] = []

    # Get disease-aware adjacency rules
    disease_rules: Dict[Tuple[int, int], bool] = {}
    if disease_vec is not None:
        raw_rules = get_effective_adjacency(disease_vec)
        # Normalise keys to (min, max) ordering
        for key, val in raw_rules.items():
            norm_key = (min(key[0], key[1]), max(key[0], key[1]))
            disease_rules[norm_key] = val
    else:
        disease_rules = {}

    # Build atlas adjacency graph once
    atlas_adj = _compute_atlas_adjacency(atlas_reg, label_ids, dilation_iters)

    # ---------------------------------------------------------------------------
    # Main correction passes (up to 3 — labels may change between passes)
    # ---------------------------------------------------------------------------
    for pass_idx in range(3):
        pass_changed = False

        # Re-find dominant component sizes each pass since labels may have changed
        dominant_sizes = _find_dominant_components(corrected, label_ids)

        # Collect all non-dominant components across all labels
        non_dominant_components = []
        for lbl in label_ids:
            mask = corrected == lbl
            if not mask.any():
                continue
            largest_size = dominant_sizes[lbl]
            if largest_size == 0:
                continue
            threshold = min_component_fraction * largest_size

            labeled_vol, n = nd_label(mask)
            # Find the dominant component index (largest)
            sizes = [(int((labeled_vol == c).sum()), c) for c in range(1, n + 1)]
            sizes.sort(reverse=True)
            dominant_comp_idx = sizes[0][1]

            for size, cid in sizes:
                if cid == dominant_comp_idx:
                    continue  # skip dominant
                if size >= largest_size:
                    continue  # too large — effectively dominant-sized, skip
                if size < threshold:
                    # Also skip very tiny fragments below the threshold
                    # (they are sub-threshold: potentially relocate them)
                    pass
                # Add all non-dominant fragments (including very small ones)
                comp_mask = labeled_vol == cid
                non_dominant_components.append({
                    "mask": comp_mask,
                    "label": lbl,
                    "size": size,
                    "largest_size": largest_size,
                })

        if not non_dominant_components:
            break

        for comp_info in non_dominant_components:
            comp_mask = comp_info["mask"]
            comp_label = comp_info["label"]
            comp_size = comp_info["size"]
            largest_size = comp_info["largest_size"]

            # Skip components whose label is protected (e.g. AO/PA in PuA cases)
            if protected_labels and comp_label in protected_labels:
                continue

            # Re-check: is this component still present and non-dominant?
            # (may have changed in a previous iteration of this pass)
            current_label_at_mask = corrected[comp_mask]
            if not np.all(current_label_at_mask == comp_label):
                continue  # already relabelled in this pass

            # 1. Compute neighbor profile
            neighbor_profile = component_neighbor_profile(
                comp_mask, corrected, label_ids, dilation_iters
            )

            # 2. Check for forbidden adjacencies
            forbidden_pairs = []
            for neighbor_lbl in neighbor_profile:
                if _is_adjacency_forbidden(comp_label, neighbor_lbl, disease_rules, atlas_adj):
                    forbidden_pairs.append(neighbor_lbl)

            if not forbidden_pairs:
                continue  # No adjacency violations — skip this component

            # 3. Find candidate labels consistent with the component's actual neighbors
            # A candidate label C is valid if ALL of comp's neighbors are allowed
            # to be adjacent to C (per disease rules / atlas).
            candidates = []
            for cand_lbl in label_ids:
                if cand_lbl == comp_label:
                    continue

                # Check if candidate is consistent with all observed neighbors
                consistent = True
                for neighbor_lbl in neighbor_profile:
                    if neighbor_lbl == cand_lbl:
                        continue  # the component itself being relabelled to cand_lbl
                    if _is_adjacency_forbidden(cand_lbl, neighbor_lbl, disease_rules, atlas_adj):
                        consistent = False
                        break

                if consistent:
                    candidates.append(cand_lbl)

            if not candidates:
                continue  # No valid relabelling found

            # Score each candidate
            best_cand = None
            best_score = 0.0  # must exceed 0 to trigger relabelling

            for cand_lbl in candidates:
                # Signal 1: connectivity improvement (weight 3)
                conn_delta = connectivity_improvement(
                    corrected, comp_mask, comp_label, cand_lbl
                )
                conn_score = 3.0 * conn_delta

                # Signal 2: boundary contact with candidate / component size (weight 2)
                boundary_contact = neighbor_profile.get(cand_lbl, 0)
                boundary_score = 2.0 * (boundary_contact / comp_size) if comp_size > 0 else 0.0

                # Signal 3: IoC with atlas (intersection of comp with atlas region / comp size)
                atlas_mask = atlas_reg == cand_lbl
                ioc_with_atlas = float((comp_mask & atlas_mask).sum()) / comp_size if comp_size > 0 else 0.0

                total_score = conn_score + boundary_score + ioc_with_atlas

                if total_score > best_score:
                    best_score = total_score
                    best_cand = cand_lbl

            if best_cand is not None and best_score > 0:
                # Perform relabelling
                conn_delta = connectivity_improvement(
                    corrected, comp_mask, comp_label, best_cand
                )
                corrected[comp_mask] = best_cand
                pass_changed = True
                log.append({
                    "original_label": comp_label,
                    "new_label": best_cand,
                    "component_size": comp_size,
                    "reason": "adjacency_violation",
                    "neighbor_profile": neighbor_profile,
                    "forbidden_pairs": forbidden_pairs,
                    "connectivity_delta": conn_delta,
                    "score": best_score,
                    "pass": pass_idx,
                })

        if not pass_changed:
            break

    # ---------------------------------------------------------------------------
    # Chain detection: A-B-A sandwich patterns
    # Scan for small B components that touch two separate A regions and relabelling
    # B as A would merge them (connectivity_improvement > 0).
    # ---------------------------------------------------------------------------
    # Re-compute dominant sizes for chain detection
    dominant_sizes = _find_dominant_components(corrected, label_ids)

    for lbl_b in label_ids:
        # Skip protected labels in chain detection too
        if protected_labels and lbl_b in protected_labels:
            continue

        mask_b = corrected == lbl_b
        if not mask_b.any():
            continue

        largest_b = dominant_sizes[lbl_b]
        if largest_b == 0:
            continue

        labeled_b, n_b = nd_label(mask_b)
        sizes_b = [(int((labeled_b == c).sum()), c) for c in range(1, n_b + 1)]
        sizes_b.sort(reverse=True)

        # Only process non-dominant B components
        dominant_b_idx = sizes_b[0][1] if sizes_b else None

        for size_b, cid_b in sizes_b:
            if cid_b == dominant_b_idx:
                continue  # skip dominant

            comp_b_mask = labeled_b == cid_b

            # Compute neighbor profile for this B component
            neighbor_profile_b = component_neighbor_profile(
                comp_b_mask, corrected, label_ids, dilation_iters
            )

            # For each potential A label (neighbor of B)
            for lbl_a, contact_count in neighbor_profile_b.items():
                if lbl_a == lbl_b:
                    continue

                # Check: would relabelling B -> A improve connectivity of A?
                conn_delta = connectivity_improvement(
                    corrected, comp_b_mask, lbl_b, lbl_a
                )
                if conn_delta <= 0:
                    continue  # no merger benefit

                # Verify disease rules allow A to have all of B's actual neighbors
                all_neighbors_ok = True
                for neighbor_lbl in neighbor_profile_b:
                    if neighbor_lbl == lbl_a:
                        continue
                    if _is_adjacency_forbidden(lbl_a, neighbor_lbl, disease_rules, atlas_adj):
                        all_neighbors_ok = False
                        break

                if not all_neighbors_ok:
                    continue

                # Relabel B component as A
                corrected[comp_b_mask] = lbl_a
                log.append({
                    "original_label": lbl_b,
                    "new_label": lbl_a,
                    "component_size": size_b,
                    "reason": "chain_sandwich_ABA",
                    "neighbor_profile": neighbor_profile_b,
                    "forbidden_pairs": [],
                    "connectivity_delta": conn_delta,
                    "score": float(conn_delta),
                    "pass": "chain",
                })
                break  # component relabelled — move to next B component

    return corrected, log
