"""NIfTI I/O and disease-map loading utilities."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np


def load_nifti(
    path: str | Path,
) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load a NIfTI file.

    Returns
    -------
    data : integer or float array (H, W, D)
    affine : (4, 4) affine matrix
    header : NIfTI header (preserves voxel spacing, orientation, etc.)
    """
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    return data, img.affine, img.header  # type: ignore[return-value]


def save_nifti(
    data: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header,
    path: str | Path,
) -> None:
    """Save an array as a NIfTI file, preserving the supplied affine/header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, str(path))


def get_voxel_spacing(header: nib.Nifti1Header) -> Tuple[float, float, float]:
    """Return voxel spacing (mm) in the same axis order as the data array."""
    zooms = header.get_zooms()[:3]
    return tuple(float(z) for z in zooms)  # type: ignore[return-value]


def load_disease_map(path: str | Path) -> Dict[str, List[int]]:
    """Load disease_map.json → {case_id: [flag0, flag1, ...]}."""
    with open(path) as f:
        return json.load(f)


def get_disease_vec(
    disease_map: Dict[str, List[int]],
    case_id: str,
    n_flags: int = 8,
) -> Optional[List[int]]:
    """Look up the disease vector for *case_id*.

    Tries several key formats to handle naming variations
    (``ct_1001_image``, ``ct_1001``, etc.).
    Zero-pads vectors shorter than *n_flags*.
    Returns ``None`` if the case is not in the map.
    """
    variants = [
        case_id,
        f"{case_id}_image",
        case_id.replace("_image", ""),
    ]
    for key in variants:
        if key in disease_map:
            vec = list(disease_map[key])
            if len(vec) < n_flags:
                vec += [0] * (n_flags - len(vec))
            return vec[:n_flags]
    return None


def resolve_case_id(filename: str) -> str:
    """Extract the case identifier from a NIfTI filename.

    ``"ct_1001_image.nii.gz"`` → ``"ct_1001_image"``
    """
    name = Path(filename).name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name
