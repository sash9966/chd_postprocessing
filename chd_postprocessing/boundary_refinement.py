"""Per-voxel boundary refinement for CHD label correction.

Addresses boundary-contiguous misassignment — mislabeled voxels that are
physically touching the dominant component of the wrong label and cannot be
corrected by component-level adjacency correction.

Algorithm overview
------------------
For each pair of labels (A, B):
1. Find the boundary zone: voxels of A near B and voxels of B near A.
2. Score each boundary voxel on four signals:
   - Local majority label (neighborhood density)
   - Centroid distance (closer to which label's centroid)
   - Atlas agreement (does the registered atlas say A or B here?)
   - Adjacency improvement (switching fixes forbidden adjacency pairs)
3. Reassign voxels where the weighted score exceeds min_confidence, capping
   at max_fraction of the zone to prevent runaway correction.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter

from .config import FOREGROUND_CLASSES, LABELS, get_effective_adjacency


# ---------------------------------------------------------------------------
# Priority pairs: checked first because they are the most common confusion
# ---------------------------------------------------------------------------
_PRIORITY_PAIRS: List[Tuple[int, int]] = [
    (LABELS["AO"], LABELS["PA"]),  # most common vessel swap
    (LABELS["AO"], LABELS["RA"]),
    (LABELS["PA"], LABELS["RA"]),
    (LABELS["AO"], LABELS["LA"]),
    (LABELS["PA"], LABELS["LV"]),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_boundary_zone(
    labels: np.ndarray,
    label_a: int,
    label_b: int,
    width_voxels: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Find the mutual boundary zone between label_a and label_b.

    Parameters
    ----------
    labels : integer label volume.
    label_a, label_b : the two labels to examine.
    width_voxels : dilation radius (in voxels) defining "near".

    Returns
    -------
    zone_a : bool mask — voxels of A within *width_voxels* of B.
    zone_b : bool mask — voxels of B within *width_voxels* of A.
    """
    mask_a = labels == label_a
    mask_b = labels == label_b

    struct = np.ones((3, 3, 3), dtype=bool)

    dilated_b = binary_dilation(mask_b, structure=struct, iterations=width_voxels)
    zone_a = mask_a & dilated_b

    dilated_a = binary_dilation(mask_a, structure=struct, iterations=width_voxels)
    zone_b = mask_b & dilated_a

    return zone_a, zone_b


def local_majority_label(
    labels: np.ndarray,
    positions: np.ndarray,
    label_ids: List[int],
    kernel_size: int = 5,
) -> np.ndarray:
    """Compute the dominant label in a local window around each position.

    Fully vectorized using ``scipy.ndimage.uniform_filter`` for density maps.

    Parameters
    ----------
    labels : integer label volume, shape (X, Y, Z).
    positions : (N, 3) int array of voxel coordinates.
    label_ids : which labels to consider.
    kernel_size : local window side length (isotropic).

    Returns
    -------
    majority : (N,) int array — label with highest local density at each position.
    """
    N = len(positions)
    if N == 0:
        return np.empty(0, dtype=np.int32)

    # Build smoothed density volume for each label
    density_vols = {}
    for lbl in label_ids:
        binary = (labels == lbl).astype(np.float32)
        density_vols[lbl] = uniform_filter(binary, size=kernel_size)

    # Stack densities at positions: shape (N, n_labels)
    densities = np.stack(
        [density_vols[lbl][positions[:, 0], positions[:, 1], positions[:, 2]]
         for lbl in label_ids],
        axis=1,
    )  # (N, n_labels)

    best_idx = np.argmax(densities, axis=1)  # (N,)
    lbl_array = np.array(label_ids, dtype=np.int32)
    return lbl_array[best_idx]


def centroid_distance_score(
    positions: np.ndarray,
    labels: np.ndarray,
    label_ids: List[int],
) -> np.ndarray:
    """Score each position by proximity to each label's centroid.

    Parameters
    ----------
    positions : (N, 3) float/int coordinates.
    labels : integer label volume (used to compute centroids).
    label_ids : labels to score against.

    Returns
    -------
    scores : (N, n_labels) float in [0, 1]; higher = closer to that label's centroid.
    """
    N = len(positions)
    n_labels = len(label_ids)
    if N == 0:
        return np.empty((0, n_labels), dtype=np.float32)

    centroids = np.full((n_labels, 3), np.nan, dtype=np.float64)
    for k, lbl in enumerate(label_ids):
        mask = labels == lbl
        if mask.any():
            idx = np.argwhere(mask)
            centroids[k] = idx.mean(axis=0)

    pos_f = positions.astype(np.float64)  # (N, 3)

    # Compute distance from each position to each centroid: (N, n_labels)
    dists = np.full((N, n_labels), np.inf, dtype=np.float64)
    for k in range(n_labels):
        if not np.isnan(centroids[k]).any():
            dists[:, k] = np.linalg.norm(pos_f - centroids[k], axis=1)

    # Invert: closer = higher score; normalise to [0, 1]
    # Use max_dist per row for normalisation
    max_dist = dists.max(axis=1, keepdims=True)
    max_dist = np.where(max_dist == 0, 1.0, max_dist)
    scores = 1.0 - dists / max_dist  # (N, n_labels)
    scores = np.clip(scores, 0.0, 1.0).astype(np.float32)
    return scores


def _adjacency_forbidden_count(
    labels: np.ndarray,
    positions: np.ndarray,
    current_lbl: int,
    candidate_lbl: int,
    adj_rules: Dict[Tuple[int, int], bool],
) -> np.ndarray:
    """Count adjacency violations that switching current→candidate would fix.

    For each position in *positions* (currently labeled *current_lbl*):
    - Count neighbors whose pairing with *current_lbl* is forbidden but
      whose pairing with *candidate_lbl* is NOT forbidden.
    This is the "switching would fix violations" improvement score.

    Parameters
    ----------
    labels : integer label volume.
    positions : (N, 3) int coordinates, all currently = current_lbl.
    current_lbl, candidate_lbl : the relabeling being evaluated.
    adj_rules : output of ``get_effective_adjacency``.

    Returns
    -------
    improvement : (N,) float — number of fixed violations per position.
    """
    N = len(positions)
    if N == 0:
        return np.empty(0, dtype=np.float32)

    shape = labels.shape
    unique_labels = np.unique(labels)

    # For each unique neighbor label, determine if switching fixes a violation
    fix_count_per_label: Dict[int, float] = {}
    for nb_lbl in unique_labels:
        if nb_lbl == current_lbl or nb_lbl == candidate_lbl:
            fix_count_per_label[nb_lbl] = 0.0
            continue
        key = (min(current_lbl, nb_lbl), max(current_lbl, nb_lbl))
        key_cand = (min(candidate_lbl, nb_lbl), max(candidate_lbl, nb_lbl))
        cur_forbidden = adj_rules.get(key, False) is False  # True if forbidden or absent from allowed
        cand_ok = adj_rules.get(key_cand, True) is not False  # True if not forbidden

        # Actually: forbidden = the value is False; allowed = True; absent = treated as allowed
        cur_forbidden_v = adj_rules.get(key)
        cand_forbidden_v = adj_rules.get(key_cand)

        cur_is_forbidden = (cur_forbidden_v is not None and cur_forbidden_v is False)
        cand_is_ok = (cand_forbidden_v is None or cand_forbidden_v is True)

        fix_count_per_label[int(nb_lbl)] = 1.0 if (cur_is_forbidden and cand_is_ok) else 0.0

    # Build binary volume for each "fixable" label and count via dilation
    improvement = np.zeros(N, dtype=np.float32)

    struct = np.ones((3, 3, 3), dtype=bool)

    for nb_lbl, fix_val in fix_count_per_label.items():
        if fix_val == 0.0:
            continue
        nb_mask = (labels == nb_lbl).astype(np.uint8)
        # Dilate to find "who is adjacent to this neighbor label"
        dilated = binary_dilation(nb_mask, structure=struct, iterations=1)
        # For each position, does it have a neighbor of nb_lbl?
        has_nb = dilated[positions[:, 0], positions[:, 1], positions[:, 2]]
        improvement += fix_val * has_nb.astype(np.float32)

    return improvement


def refine_label_boundary(
    labels: np.ndarray,
    label_a: int,
    label_b: int,
    atlas_reg: Optional[np.ndarray] = None,
    disease_vec: Optional[List[int]] = None,
    width_voxels: int = 3,
    kernel_size: int = 5,
    min_confidence: float = 0.6,
    max_fraction: float = 0.15,
) -> Tuple[np.ndarray, Dict]:
    """Refine the boundary between label_a and label_b by per-voxel scoring.

    Parameters
    ----------
    labels : integer label volume (will not be modified — a copy is returned).
    label_a, label_b : the two labels whose shared boundary is refined.
    atlas_reg : registered atlas volume for atlas-agreement signal.
    disease_vec : 8-element disease flag vector (for adjacency signal).
    width_voxels : boundary zone dilation radius.
    kernel_size : local majority window size.
    min_confidence : normalized score threshold in (-1, +1) to trigger
                     reassignment (positive means reassign).
    max_fraction : maximum fraction of the zone to reassign per call.

    Returns
    -------
    corrected : corrected label volume (copy).
    log : dict with keys ``a_to_b`` and ``b_to_a`` (voxel counts).
    """
    corrected = labels.copy()
    log = {"a_to_b": 0, "b_to_a": 0, "label_a": label_a, "label_b": label_b}

    zone_a, zone_b = find_boundary_zone(corrected, label_a, label_b, width_voxels)

    if not zone_a.any() and not zone_b.any():
        return corrected, log

    # Adjacency rules
    adj_rules: Dict[Tuple[int, int], bool] = {}
    if disease_vec is not None:
        adj_rules = get_effective_adjacency(disease_vec)

    # Precompute centroid-score vols for both labels
    all_label_ids = [label_a, label_b]

    # ----------------------------------------------------------------
    # Helper: score zone voxels that are currently *current_lbl*
    # and may be reassigned to *candidate_lbl*.
    # Positive confidence → reassign.
    # ----------------------------------------------------------------
    def _score_zone(zone_mask: np.ndarray, current_lbl: int, candidate_lbl: int) -> np.ndarray:
        positions = np.argwhere(zone_mask)  # (N, 3)
        N = len(positions)
        if N == 0:
            return np.empty(0, dtype=np.float32)

        # Signal 1: local majority (weight 2.0)
        majority = local_majority_label(corrected, positions, all_label_ids, kernel_size)
        s_majority = np.where(
            majority == candidate_lbl, 1.0,
            np.where(majority == current_lbl, -1.0, 0.0)
        ).astype(np.float32) * 2.0

        # Signal 2: centroid distance (weight 1.5)
        cent_scores = centroid_distance_score(positions, corrected, all_label_ids)
        # cent_scores[:,0] = score for label_a, [:,1] = score for label_b
        idx_cur  = all_label_ids.index(current_lbl)
        idx_cand = all_label_ids.index(candidate_lbl)
        dist_cur  = cent_scores[:, idx_cur]
        dist_cand = cent_scores[:, idx_cand]
        eps = 1e-6
        # Positive when candidate centroid is closer (higher closeness) than current centroid
        s_centroid = ((dist_cand - dist_cur) / (dist_cur + dist_cand + eps)) * 1.5

        # Signal 3: atlas agreement (weight 1.0)
        if atlas_reg is not None:
            atlas_vals = atlas_reg[positions[:, 0], positions[:, 1], positions[:, 2]]
            s_atlas = np.where(
                atlas_vals == candidate_lbl, 1.0,
                np.where(atlas_vals == current_lbl, -1.0, 0.0)
            ).astype(np.float32) * 1.0
        else:
            s_atlas = np.zeros(N, dtype=np.float32)

        # Signal 4: adjacency improvement (weight 3.0)
        if adj_rules:
            adj_improve = _adjacency_forbidden_count(
                corrected, positions, current_lbl, candidate_lbl, adj_rules
            )
            # Normalize: max possible = number of unique neighbor labels
            max_improve = max(float(adj_improve.max()), 1.0)
            s_adj = (adj_improve / max_improve) * 3.0
        else:
            s_adj = np.zeros(N, dtype=np.float32)

        raw_score = s_majority + s_centroid + s_atlas + s_adj
        max_total = 2.0 + 1.5 + 1.0 + 3.0  # 7.5
        conf = raw_score / max_total  # in [-1, +1]
        return conf  # (N,)

    def _apply_zone(zone_mask: np.ndarray, current_lbl: int, candidate_lbl: int) -> int:
        positions = np.argwhere(zone_mask)
        N = len(positions)
        if N == 0:
            return 0

        conf = _score_zone(zone_mask, current_lbl, candidate_lbl)

        # Only candidates above threshold
        above = conf > min_confidence
        n_above = int(above.sum())
        if n_above == 0:
            return 0

        # Cap at max_fraction of zone
        cap = max(1, int(np.ceil(max_fraction * N)))
        if n_above > cap:
            # Take top cap voxels by confidence
            order = np.argsort(conf)[::-1]
            above = np.zeros(N, dtype=bool)
            above[order[:cap]] = True
            n_above = cap

        # Reassign
        reassign_pos = positions[above]
        corrected[reassign_pos[:, 0], reassign_pos[:, 1], reassign_pos[:, 2]] = candidate_lbl
        return int(above.sum())

    # Apply zone_a: voxels currently A, candidate B
    n_a_to_b = _apply_zone(zone_a, label_a, label_b)
    log["a_to_b"] = n_a_to_b

    # Recompute zone_b after zone_a changes
    zone_a2, zone_b2 = find_boundary_zone(corrected, label_a, label_b, width_voxels)

    # Apply zone_b: voxels currently B, candidate A
    n_b_to_a = _apply_zone(zone_b2, label_b, label_a)
    log["b_to_a"] = n_b_to_a

    return corrected, log


def refine_all_boundaries(
    labels: np.ndarray,
    atlas_reg: Optional[np.ndarray] = None,
    label_ids: Optional[List[int]] = None,
    disease_vec: Optional[List[int]] = None,
    width_voxels: int = 3,
    kernel_size: int = 5,
    min_confidence: float = 0.6,
    max_passes: int = 3,
) -> Tuple[np.ndarray, List[Dict]]:
    """Run boundary refinement for all adjacent label pairs.

    Priority pairs are processed first (most common confusion pairs), then
    all remaining pairs detected as adjacent in the volume.  Repeats for up
    to *max_passes* passes, stopping early if no voxels changed.

    Parameters
    ----------
    labels : integer label volume.
    atlas_reg : registered atlas volume (optional, for atlas-agreement signal).
    label_ids : foreground labels to consider.  Default: 1–7.
    disease_vec : 8-element disease flag vector.
    width_voxels : boundary zone dilation radius.
    kernel_size : local majority window size.
    min_confidence : confidence threshold for reassignment.
    max_passes : maximum correction passes.

    Returns
    -------
    corrected : corrected label volume.
    log_list : list of per-pair log dicts from each pass.
    """
    if label_ids is None:
        label_ids = list(FOREGROUND_CLASSES)

    corrected = labels.copy()
    log_list: List[Dict] = []

    for pass_idx in range(max_passes):
        pass_total = 0

        # Build ordered pair list: priority pairs first, then detected pairs
        present = set(int(v) for v in np.unique(corrected) if v in label_ids)
        seen: set = set()
        pair_list: List[Tuple[int, int]] = []

        for a, b in _PRIORITY_PAIRS:
            if a in present and b in present:
                key = (min(a, b), max(a, b))
                if key not in seen:
                    pair_list.append((a, b))
                    seen.add(key)

        # Detect remaining adjacent pairs from the volume
        struct = np.ones((3, 3, 3), dtype=bool)
        for lbl in sorted(present):
            mask = corrected == lbl
            if not mask.any():
                continue
            dilated = binary_dilation(mask, structure=struct, iterations=width_voxels)
            for other in sorted(present):
                if other <= lbl:
                    continue
                key = (min(lbl, other), max(lbl, other))
                if key in seen:
                    continue
                other_mask = corrected == other
                if other_mask.any() and (dilated & other_mask).any():
                    pair_list.append((lbl, other))
                    seen.add(key)

        for label_a, label_b in pair_list:
            corrected, entry = refine_label_boundary(
                corrected,
                label_a=label_a,
                label_b=label_b,
                atlas_reg=atlas_reg,
                disease_vec=disease_vec,
                width_voxels=width_voxels,
                kernel_size=kernel_size,
                min_confidence=min_confidence,
            )
            entry["pass"] = pass_idx
            changed = entry["a_to_b"] + entry["b_to_a"]
            if changed > 0:
                log_list.append(entry)
                pass_total += changed

        if pass_total == 0:
            break  # converged

    return corrected, log_list
