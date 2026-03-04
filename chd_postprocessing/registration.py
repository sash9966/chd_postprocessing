"""Rigid label-based registration of atlas → predicted segmentation.

Strategy
--------
Cardiac CT scans share a common rough orientation, so a full deformable
registration is unnecessary.  A centroid-based rigid alignment brings the
atlas into approximate correspondence with the prediction, which is
sufficient for the subsequent overlap-based label correction.

Two modes are available:

``"centroid"`` (default, fast)
    Align the centre of mass of all foreground voxels in the atlas to the
    centre of mass of all foreground voxels in the prediction.  Pure
    translation, no rotation.

``"pca"``
    Align centroids *and* principal axes of the foreground point clouds.
    Adds a rotation component derived from the eigenvectors of each cloud's
    covariance matrix.  More accurate when patient orientation varies
    substantially, but slightly slower.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import affine_transform, center_of_mass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _foreground_coords(labels: np.ndarray) -> np.ndarray:
    """Return (N, 3) array of voxel coordinates where labels > 0."""
    return np.column_stack(np.where(labels > 0)).astype(float)


def _centroid(labels: np.ndarray) -> np.ndarray:
    """Centre of mass of foreground voxels (voxel coordinates)."""
    fg = labels > 0
    if not fg.any():
        return np.array([(s - 1) / 2.0 for s in labels.shape])
    # scipy.ndimage.center_of_mass returns (z, y, x) by default — same axis order
    return np.array(center_of_mass(fg.astype(np.uint8)))


def _pca_axes(coords: np.ndarray) -> np.ndarray:
    """Return the 3×3 rotation matrix whose rows are PCA axes (descending var)."""
    centred = coords - coords.mean(axis=0)
    cov = (centred.T @ centred) / max(len(centred) - 1, 1)
    _, vecs = np.linalg.eigh(cov)          # columns are eigenvectors, ascending order
    return vecs[:, ::-1].T                 # rows = axes, descending variance


def _apply_transform(
    atlas: np.ndarray,
    R: np.ndarray,
    t_vox: np.ndarray,
    center: np.ndarray,
    output_shape: Tuple[int, ...],
) -> np.ndarray:
    """Apply rigid transform (rotation R around *center*, then translate *t_vox*).

    Uses the *pull* convention of ``scipy.ndimage.affine_transform``:

        input_coord = R.T @ output_coord + offset
        offset = center - R.T @ (center + t_vox)
    """
    offset = center - R.T @ (center + t_vox)
    return affine_transform(
        atlas, R.T, offset=offset,
        output_shape=output_shape,
        order=0, mode="constant", cval=0,
    ).astype(atlas.dtype)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_atlas_to_pred(
    atlas:   np.ndarray,
    pred:    np.ndarray,
    spacing: Tuple[float, float, float],
    mode:    str = "centroid",
) -> np.ndarray:
    """Rigidly align *atlas* into the coordinate frame of *pred*.

    Parameters
    ----------
    atlas : integer label volume (H_a × W_a × D_a).
    pred : integer label volume (H_p × W_p × D_p).
           The output will have this shape.
    spacing : voxel spacing (mm) — used by ``"pca"`` mode to convert
              physical-space distances; currently informational.
    mode : ``"centroid"`` or ``"pca"`` (see module docstring).

    Returns
    -------
    Registered atlas as an integer array with the same shape as *pred*.
    """
    if mode not in {"centroid", "pca"}:
        raise ValueError(f"Unknown registration mode: {mode!r}. Use 'centroid' or 'pca'.")

    output_shape = pred.shape

    # ------------------------------------------------------------------
    # Edge case: empty atlas or empty prediction
    # ------------------------------------------------------------------
    atlas_fg = atlas > 0
    pred_fg  = pred  > 0
    if not atlas_fg.any() or not pred_fg.any():
        # Cannot align — resample atlas into pred's grid and return
        if atlas.shape == pred.shape:
            return atlas.copy()
        return affine_transform(
            atlas, np.eye(3), offset=np.zeros(3),
            output_shape=output_shape, order=0, mode="constant", cval=0,
        ).astype(atlas.dtype)

    # ------------------------------------------------------------------
    # Step 1: centroid translation
    # ------------------------------------------------------------------
    atlas_cm = _centroid(atlas)
    pred_cm  = _centroid(pred)
    t_vox    = pred_cm - atlas_cm          # translation to apply in atlas space

    R_identity = np.eye(3)

    # ------------------------------------------------------------------
    # Step 2 (PCA mode): rotation to align principal axes
    # ------------------------------------------------------------------
    if mode == "pca":
        atlas_coords = _foreground_coords(atlas)
        pred_coords  = _foreground_coords(pred)

        # Only compute PCA when there are enough points for a stable estimate
        if len(atlas_coords) >= 10 and len(pred_coords) >= 10:
            R_atlas = _pca_axes(atlas_coords)    # rows = atlas principal axes
            R_pred  = _pca_axes(pred_coords)     # rows = pred  principal axes
            # R maps atlas axes → pred axes:  R = R_pred.T @ R_atlas
            R = R_pred.T @ R_atlas
        else:
            R = R_identity
    else:
        R = R_identity

    # ------------------------------------------------------------------
    # Apply the combined transform
    # ------------------------------------------------------------------
    registered = _apply_transform(
        atlas, R, t_vox, center=atlas_cm, output_shape=output_shape
    )
    return registered
