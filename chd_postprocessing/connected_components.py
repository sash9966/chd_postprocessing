"""Connected-component analysis and small-fragment cleanup.

nnU-Net occasionally produces small floating islands of AO or PA label that
are spatially disconnected from the main vessel body.  These orphan fragments
confound the adjacency-based label correction (they can pull adjacency scores
toward the wrong ventricle).  This module removes them before the adjacency
check runs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import label as nd_label

from .config import LABELS, MIN_COMPONENT_FRACTION


def cleanup_vessel_fragments(
    labels: np.ndarray,
    vessel_label_ids: Optional[List[int]] = None,
    min_component_fraction: float = MIN_COMPONENT_FRACTION,
) -> Tuple[np.ndarray, Dict]:
    """Remove small disconnected fragments from vessel labels.

    For each vessel label, connected-component analysis is performed.  Any
    component whose voxel count is less than *min_component_fraction* × (size
    of the largest component) is reassigned to background (label 0).

    Parameters
    ----------
    labels : integer segmentation volume
    vessel_label_ids : label IDs to clean up.
                       Default: ``[LABELS["AO"], LABELS["PA"]]`` = ``[6, 7]``.
    min_component_fraction : fragments below this fraction of the largest
                             component are removed.  Default 1 % (0.01).

    Returns
    -------
    cleaned_labels : corrected volume (copy)
    info : per-label dict with keys
           ``{"n_components", "removed", "kept", "sizes"}``
    """
    if vessel_label_ids is None:
        vessel_label_ids = [LABELS["AO"], LABELS["PA"]]

    labels = labels.copy()
    info: Dict = {}

    for lbl in vessel_label_ids:
        mask = labels == lbl
        if not mask.any():
            info[lbl] = {"n_components": 0, "removed": 0, "kept": 0, "sizes": []}
            continue

        labeled_vol, n_components = nd_label(mask)
        if n_components == 0:
            info[lbl] = {"n_components": 0, "removed": 0, "kept": 0, "sizes": []}
            continue

        sizes = [int(np.sum(labeled_vol == i)) for i in range(1, n_components + 1)]
        largest_size = max(sizes)
        threshold = min_component_fraction * largest_size

        removed = 0
        for comp_idx, size in enumerate(sizes, start=1):
            if size < threshold:
                labels[labeled_vol == comp_idx] = 0
                removed += 1

        info[lbl] = {
            "n_components": n_components,
            "removed":      removed,
            "kept":         n_components - removed,
            "sizes":        sorted(sizes, reverse=True),
        }

    return labels, info


def component_summary(labels: np.ndarray, label_id: int) -> Dict:
    """Return connected-component statistics for a single label.

    Useful for exploratory analysis before deciding on cleanup parameters.

    Returns
    -------
    dict with keys: ``n_components``, ``sizes`` (sorted descending),
    ``largest_fraction`` (largest / total voxels of this label)
    """
    mask = labels == label_id
    if not mask.any():
        return {"n_components": 0, "sizes": [], "largest_fraction": float("nan")}

    labeled_vol, n_components = nd_label(mask)
    sizes = sorted(
        [int(np.sum(labeled_vol == i)) for i in range(1, n_components + 1)],
        reverse=True,
    )
    total = sum(sizes)
    largest_fraction = sizes[0] / total if total > 0 else float("nan")

    return {
        "n_components":    n_components,
        "sizes":           sizes,
        "largest_fraction": round(largest_fraction, 4),
    }
