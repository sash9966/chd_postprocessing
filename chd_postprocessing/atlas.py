"""Atlas library management and synthetic atlas creation.

An *atlas* is a labelled ground-truth segmentation from the training set.
For each test case we:
  1. Select one (or more) atlas entries from the library.
  2. Apply a small random perturbation so the registration step is non-trivial.
  3. Register the perturbed atlas to the predicted segmentation space.
  4. Use the registered atlas to guide label correction.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import affine_transform

from .config import DISEASE_FLAGS, FOREGROUND_CLASSES
from .io_utils import get_disease_vec, get_voxel_spacing, load_disease_map, load_nifti


# ---------------------------------------------------------------------------
# Atlas entry
# ---------------------------------------------------------------------------

@dataclass
class AtlasEntry:
    """One labelled atlas (a GT segmentation from the training set).

    Labels and geometry are loaded lazily via :meth:`load`.
    """
    case_id:     str
    path:        Path
    disease_vec: List[int]                               # length-8 binary flags

    # Populated by load()
    labels:  Optional[np.ndarray] = field(default=None, repr=False)
    affine:  Optional[np.ndarray] = field(default=None, repr=False)
    spacing: Optional[Tuple[float, ...]] = field(default=None, repr=False)

    def load(self) -> None:
        """Load NIfTI data into memory (idempotent)."""
        if self.labels is None:
            data, self.affine, header = load_nifti(self.path)
            self.labels  = data.astype(np.int32)
            self.spacing = get_voxel_spacing(header)

    @property
    def disease_name(self) -> str:
        """Human-readable summary of active disease flags."""
        active = [DISEASE_FLAGS[i] for i, f in enumerate(self.disease_vec) if f]
        return "+".join(active) if active else "Normal"

    def hamming_distance(self, other_vec: List[int]) -> int:
        """Hamming distance between this entry's disease vector and *other_vec*."""
        return sum(a != b for a, b in zip(self.disease_vec, other_vec))


# ---------------------------------------------------------------------------
# Atlas library
# ---------------------------------------------------------------------------

class AtlasLibrary:
    """Collection of :class:`AtlasEntry` objects built from a GT folder.

    Parameters
    ----------
    entries : pre-built list (use :meth:`load_all` to construct).
    """

    def __init__(self, entries: List[AtlasEntry]) -> None:
        self.entries = entries

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def load_all(
        cls,
        gt_folder: str | Path,
        disease_map_path: Optional[str | Path] = None,
        file_pattern: str = "*.nii.gz",
    ) -> "AtlasLibrary":
        """Build a library from all NIfTI files in *gt_folder*.

        Parameters
        ----------
        gt_folder : directory containing ground-truth label NIfTI files.
        disease_map_path : path to ``disease_map.json``.  If ``None``, all
                           entries get a zero disease vector.
        file_pattern : glob pattern for label files.
        """
        gt_folder = Path(gt_folder)
        disease_map: Dict = {}
        if disease_map_path is not None:
            disease_map = load_disease_map(disease_map_path)

        entries: List[AtlasEntry] = []
        for path in sorted(gt_folder.glob(file_pattern)):
            # Resolve the case ID from the filename
            stem = path.name
            for ext in (".nii.gz", ".nii"):
                if stem.endswith(ext):
                    stem = stem[: -len(ext)]
                    break

            vec = get_disease_vec(disease_map, stem, n_flags=8)
            if vec is None:
                vec = [0] * 8

            entries.append(AtlasEntry(case_id=stem, path=path, disease_vec=vec))

        return cls(entries)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select_for_case(
        self,
        disease_vec: List[int],
        rng: random.Random,
        mode: str = "best_match",
        exclude_case_id: Optional[str] = None,
    ) -> AtlasEntry:
        """Pick one atlas entry for the given test case.

        Parameters
        ----------
        disease_vec : binary disease flag vector of the test case.
        rng : seeded :class:`random.Random` for reproducibility.
        mode : ``"random"`` — uniform random selection;
               ``"best_match"`` — minimise Hamming distance to *disease_vec*.
        exclude_case_id : if provided, never select this case (avoids
                          selecting the test case itself as its own atlas).
                          The ``_image`` suffix is stripped before comparison
                          so ``ct_1042`` correctly excludes ``ct_1042_image``.

        Returns
        -------
        :class:`AtlasEntry` (not yet loaded into memory).
        """
        def _base(cid: str) -> str:
            return cid[:-6] if cid.endswith("_image") else cid

        exclude_base = _base(exclude_case_id) if exclude_case_id else None
        pool = [
            e for e in self.entries
            if exclude_base is None or _base(e.case_id) != exclude_base
        ]
        if not pool:
            pool = list(self.entries)   # fallback: allow self-match

        if mode == "random":
            return rng.choice(pool)

        if mode == "best_match":
            min_dist  = min(e.hamming_distance(disease_vec) for e in pool)
            best_pool = [e for e in pool if e.hamming_distance(disease_vec) == min_dist]
            return rng.choice(best_pool)

        raise ValueError(f"Unknown selection mode: {mode!r}. Use 'random' or 'best_match'.")


# ---------------------------------------------------------------------------
# Synthetic atlas creation
# ---------------------------------------------------------------------------

def _rotation_matrix_3d(rx: float, ry: float, rz: float) -> np.ndarray:
    """Build a 3×3 rotation matrix from Euler angles (radians, XYZ extrinsic)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx,  cx]], dtype=float)
    Ry = np.array([[cy, 0, sy], [0, 1,  0], [-sy, 0,  cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def create_synthetic_atlas(
    labels: np.ndarray,
    spacing: Tuple[float, float, float],
    rng: random.Random,
    rot_deg:     float = 10.0,
    trans_mm:    float = 5.0,
    scale_range: float = 0.05,
) -> np.ndarray:
    """Apply a small random rigid + scale perturbation to a GT label volume.

    This prevents the atlas from being a trivial identity match to the
    prediction and ensures the registration step has something to solve.

    The transform is applied around the volume centre.  Nearest-neighbour
    interpolation (``order=0``) is used so integer labels are preserved.

    Parameters
    ----------
    labels : integer label volume (H × W × D).
    spacing : voxel spacing (mm) along each axis.
    rng : seeded :class:`random.Random`.
    rot_deg : maximum rotation in degrees (applied independently per axis).
    trans_mm : maximum translation in mm (applied independently per axis).
    scale_range : maximum relative scale change (e.g. 0.05 → ±5 %).

    Returns
    -------
    Perturbed label array with the same shape and dtype as *labels*.
    """
    shape = np.array(labels.shape, dtype=float)
    center = (shape - 1) / 2.0                          # voxel-space centre

    # Random rotation angles
    rx = math.radians(rng.uniform(-rot_deg, rot_deg))
    ry = math.radians(rng.uniform(-rot_deg, rot_deg))
    rz = math.radians(rng.uniform(-rot_deg, rot_deg))
    R  = _rotation_matrix_3d(rx, ry, rz)

    # Random translation (convert mm → voxels)
    t_vox = np.array([
        rng.uniform(-trans_mm, trans_mm) / spacing[i] for i in range(3)
    ])

    # Random isotropic scale (applied as diagonal matrix)
    scale = 1.0 + rng.uniform(-scale_range, scale_range)
    S = np.diag([scale, scale, scale])

    # Combined transform:  new_pos = S @ R @ (old_pos - center) + center + t
    # Pull convention for affine_transform:
    #   input_coord = M @ output_coord + offset
    #   M = (S @ R)^-1, offset = center - M @ (center + t)
    M = np.linalg.inv(S @ R)
    offset = center - M @ (center + t_vox)

    perturbed = affine_transform(
        labels, M, offset=offset,
        output_shape=labels.shape,
        order=0, mode="constant", cval=0,
    )
    return perturbed.astype(labels.dtype)
