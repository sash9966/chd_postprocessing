"""Rigid label-based registration of atlas → predicted segmentation.

Strategy
--------
Cardiac CT scans share a common rough orientation, so a full deformable
registration is unnecessary.  A centroid-based rigid alignment brings the
atlas into approximate correspondence with the prediction, which is
sufficient for the subsequent component-level label correction.

Two modes are available:

``"centroid"`` (default, recommended)
    Align the centre of mass of all foreground voxels in the atlas to the
    centre of mass of all foreground voxels in the prediction.  Pure
    translation, no rotation.  Safe for all inputs.

``"pca"``
    Align centroids *and* principal axes of the foreground point clouds.
    Adds a rotation component derived from the eigenvectors of each cloud's
    covariance matrix.  The sign ambiguity of eigenvectors is resolved by
    enforcing a right-handed coordinate system (det(R) = +1), which prevents
    180° flips that could swap left/right structures.  More accurate when
    patient orientation varies substantially, but centroid mode is preferred
    for typical cardiac CT data where orientation is consistent.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

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
    return np.array(center_of_mass(fg.astype(np.uint8)))


def _pca_axes(coords: np.ndarray) -> np.ndarray:
    """Return the 3×3 rotation matrix whose rows are PCA axes (descending var).

    Sign ambiguity is resolved by enforcing a right-handed coordinate system
    (determinant = +1).  Without this fix, eigenvectors can point in arbitrary
    directions, causing the composed rotation R = R_pred.T @ R_atlas to be a
    reflection (det = -1) that flips the volume and swaps laterally symmetric
    structures such as LV/RV or AO/PA.
    """
    centred = coords - coords.mean(axis=0)
    cov = (centred.T @ centred) / max(len(centred) - 1, 1)
    _, vecs = np.linalg.eigh(cov)       # columns = eigenvectors, ascending order
    axes = vecs[:, ::-1].T              # rows = axes, descending variance
    # Enforce right-handed system: flip last axis if determinant is negative
    if np.linalg.det(axes) < 0:
        axes[-1] *= -1
    return axes


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

def register_atlas_per_structure(
    atlas:     np.ndarray,
    pred:      np.ndarray,
    label_ids: List[int],
) -> Dict[int, np.ndarray]:
    """Per-label centroid registration — returns one aligned boolean mask per label.

    Instead of a single global foreground centroid translation, each label is
    independently shifted so that its own centroid in the atlas aligns with the
    same label's centroid in the prediction.  This provides much better alignment
    for structures that are far from the whole-heart centroid (AO and PA are
    displaced in opposite directions from the centroid, so a global shift moves
    one vessel closer and the other farther).

    Parameters
    ----------
    atlas : integer label volume.
    pred : integer label volume (defines output grid shape).
    label_ids : foreground label IDs to process.

    Returns
    -------
    ``{label_id: bool mask}`` where each mask is the atlas region for that label
    shifted to align with the prediction, cropped/padded to ``pred.shape``.
    Falls back to the global foreground centroid shift for labels that are absent
    in either the atlas or the prediction.
    """
    output_shape = pred.shape
    R_identity   = np.eye(3)
    zero_center  = np.zeros(3)

    # Global fallback translation (whole-foreground centroid alignment)
    atlas_fg_cm = _centroid(atlas)
    pred_fg_cm  = _centroid(pred)
    t_global    = pred_fg_cm - atlas_fg_cm

    result: Dict[int, np.ndarray] = {}
    for lbl in label_ids:
        atlas_lbl = (atlas == lbl)
        pred_lbl  = (pred  == lbl)

        if atlas_lbl.any() and pred_lbl.any():
            atlas_cm = np.array(center_of_mass(atlas_lbl.astype(np.uint8)))
            pred_cm  = np.array(center_of_mass(pred_lbl.astype(np.uint8)))
            t_lbl    = pred_cm - atlas_cm
        else:
            t_lbl = t_global   # label absent in one volume — use global shift

        # Pure translation: _apply_transform with R=I and center=0 gives offset = -t_lbl
        shifted = _apply_transform(
            atlas_lbl.astype(np.uint8),
            R_identity, t_lbl,
            center=zero_center,
            output_shape=output_shape,
        )
        result[lbl] = shifted > 0

    return result


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
    spacing : voxel spacing (mm) — used by ``"pca"`` mode.
    mode : ``"centroid"`` (default) or ``"pca"`` (see module docstring).

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
    t_vox    = pred_cm - atlas_cm

    R_identity = np.eye(3)

    # ------------------------------------------------------------------
    # Step 2 (PCA mode): rotation to align principal axes
    # ------------------------------------------------------------------
    if mode == "pca":
        atlas_coords = _foreground_coords(atlas)
        pred_coords  = _foreground_coords(pred)

        if len(atlas_coords) >= 10 and len(pred_coords) >= 10:
            R_atlas = _pca_axes(atlas_coords)   # rows = atlas principal axes
            R_pred  = _pca_axes(pred_coords)    # rows = pred  principal axes
            # R maps atlas axes → pred axes
            R = R_pred.T @ R_atlas
            # Ensure the composed rotation is proper (det = +1)
            if np.linalg.det(R) < 0:
                R_atlas[-1] *= -1
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
