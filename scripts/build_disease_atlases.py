"""Build a disease-stratified atlas library from imageCHD training labels.

For each unique disease profile found in the training set, one representative
case is selected, a small random deformation is applied (so the registration
step is non-trivial), and the result is saved to the output directory.

The output directory can be passed directly as ``--gt-folder`` to the atlas
post-processing pipeline.  Because filenames preserve the original case IDs,
``AtlasLibrary.load_all()`` can look them up in disease_map.json as normal.

Usage
-----
::

    python build_disease_atlases.py \\
        --tr-folder /data/imageCHD/labelsTr \\
        --disease-map /data/disease_map.json \\
        --output-dir /data/atlases \\
        --exclude ct_1042 ct_1145

This excludes cases whose *base* IDs (without ``_image``) appear in any
prediction / test set, so that atlases are never built from test data.

The script writes:
- One ``.nii.gz`` per disease group (using the first alphabetically-sorted
  representative case for reproducibility).
- ``manifest.json`` — maps disease name to ``{case_id, file, disease_vec}``.

Default parameters match the ``create_synthetic_atlas`` defaults:
  rotation ±10°, translation ±5 mm, scale ±5%.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

# ---------------------------------------------------------------------------
# Resolve package location even when run as a plain script
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chd_postprocessing.atlas import create_synthetic_atlas
from chd_postprocessing.config import DISEASE_FLAGS
from chd_postprocessing.io_utils import (
    get_disease_vec,
    get_voxel_spacing,
    load_disease_map,
    load_nifti,
    resolve_case_id,
    save_nifti,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_id(case_id: str) -> str:
    """Strip trailing ``_image`` suffix for comparison."""
    return case_id[:-6] if case_id.endswith("_image") else case_id


def _disease_name(vec: List[int]) -> str:
    active = [DISEASE_FLAGS[i] for i, f in enumerate(vec) if f]
    return "+".join(active) if active else "Normal"


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_atlas_library(
    tr_folder: Path,
    disease_map: Dict[str, List[int]],
    output_dir: Path,
    exclude_base_ids: Optional[List[str]] = None,
    seed: int = 42,
    rot_deg: float = 10.0,
    trans_mm: float = 5.0,
    scale_range: float = 0.05,
    file_pattern: str = "*.nii.gz",
) -> Dict:
    """Build and save one deformed atlas per disease group.

    Parameters
    ----------
    tr_folder : folder containing training GT label NIfTI files.
    disease_map : ``{case_id: disease_vec}`` (keys like ``ct_XXXX_image``).
    output_dir : destination directory for atlas NIfTI files.
    exclude_base_ids : base case IDs (without ``_image``) to exclude from
                       the atlas pool (e.g. test cases that appear in
                       the training fold).
    seed : random seed for the synthetic deformation.
    rot_deg, trans_mm, scale_range : deformation magnitude parameters.
    file_pattern : glob pattern to find training NIfTI files.

    Returns
    -------
    manifest dict (also written to ``output_dir/manifest.json``).
    """
    exclude_base_ids = {_base_id(x) for x in (exclude_base_ids or [])}
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 1. Collect training cases and group by disease name
    # ------------------------------------------------------------------
    groups: Dict[str, List[Path]] = defaultdict(list)

    all_files = sorted(tr_folder.glob(file_pattern))
    if not all_files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' in {tr_folder}")

    skipped_exclude = []
    skipped_no_map = []

    for path in all_files:
        case_id = resolve_case_id(path.name)

        # Skip if this case overlaps with the test / prediction set
        if _base_id(case_id) in exclude_base_ids:
            skipped_exclude.append(case_id)
            continue

        vec = get_disease_vec(disease_map, case_id)
        if vec is None:
            skipped_no_map.append(case_id)
            continue

        disease = _disease_name(vec)
        groups[disease].append(path)

    print(f"Training cases found: {len(all_files)}")
    if skipped_exclude:
        print(f"  Excluded (overlap with test set): {skipped_exclude}")
    if skipped_no_map:
        print(f"  Skipped (not in disease_map): {skipped_no_map}")
    print(f"Disease groups: {len(groups)}")
    for disease, paths in sorted(groups.items(), key=lambda x: -len(x[1])):
        print(f"  {disease}: {len(paths)} cases")

    # ------------------------------------------------------------------
    # 2. For each disease group, pick a representative, deform, save
    # ------------------------------------------------------------------
    manifest: Dict = {}

    for disease, paths in sorted(groups.items()):
        # Sort for reproducibility; pick the first (alphabetically)
        paths_sorted = sorted(paths)
        chosen_path = paths_sorted[0]
        case_id = resolve_case_id(chosen_path.name)

        vec = get_disease_vec(disease_map, case_id)

        print(f"\n  [{disease}]  representative: {case_id}")

        # Load
        labels, affine, header = load_nifti(chosen_path)
        labels = labels.astype(np.int32)
        spacing = get_voxel_spacing(header)

        # Deform
        deformed = create_synthetic_atlas(
            labels, spacing, rng,
            rot_deg=rot_deg,
            trans_mm=trans_mm,
            scale_range=scale_range,
        )

        # Save with original case_id filename (so disease_map lookup works)
        out_path = output_dir / f"{case_id}.nii.gz"
        save_nifti(deformed, affine, header, out_path)
        print(f"    Saved → {out_path.name}")

        manifest[disease] = {
            "case_id": case_id,
            "file": out_path.name,
            "disease_vec": vec,
        }

    # ------------------------------------------------------------------
    # 3. Write manifest
    # ------------------------------------------------------------------
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")
    print(f"Total atlas files: {len(manifest)}")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a disease-stratified atlas library from imageCHD training labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--tr-folder", required=True,
        help="Folder containing training GT label NIfTI files (labelsTr/).",
    )
    p.add_argument(
        "--disease-map", required=True,
        help="Path to disease_map.json.",
    )
    p.add_argument(
        "--output-dir", required=True,
        help="Output directory for atlas NIfTI files and manifest.json.",
    )
    p.add_argument(
        "--exclude", nargs="*", default=[],
        metavar="CASE_ID",
        help=(
            "Base case IDs (without _image) to exclude from the atlas pool. "
            "Typically the test / validation case IDs so that atlases are "
            "never built from test data.  Example: --exclude ct_1042 ct_1145"
        ),
    )
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--rot-deg",     type=float, default=10.0,
                   help="Max rotation per axis (degrees).")
    p.add_argument("--trans-mm",    type=float, default=5.0,
                   help="Max translation per axis (mm).")
    p.add_argument("--scale-range", type=float, default=0.05,
                   help="Max relative scale change (e.g. 0.05 → ±5%).")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    disease_map = load_disease_map(args.disease_map)

    build_atlas_library(
        tr_folder=Path(args.tr_folder),
        disease_map=disease_map,
        output_dir=Path(args.output_dir),
        exclude_base_ids=args.exclude,
        seed=args.seed,
        rot_deg=args.rot_deg,
        trans_mm=args.trans_mm,
        scale_range=args.scale_range,
    )


if __name__ == "__main__":
    main()
