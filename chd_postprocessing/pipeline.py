"""Chain post-processing steps into a configurable pipeline.

Usage
-----
Single file::

    from chd_postprocessing.pipeline import run_pipeline
    result = run_pipeline("pred.nii.gz", "corrected.nii.gz", disease_vec=[0,0,0,0,0,0,0,0])

Whole folder::

    from chd_postprocessing.pipeline import run_folder_pipeline
    results = run_folder_pipeline("pred_folder/", "corrected_folder/", disease_map_path="disease_map.json")

Available steps (in default execution order):
    - ``"cc_cleanup"``          — remove small disconnected AO/PA fragments
    - ``"adjacency_correction"``— swap AO/PA labels if anatomically wrong
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .anatomy_priors import CorrectionResult, correct_ao_pa_labels
from .config import DEFAULT_DILATION_RADIUS_MM
from .connected_components import cleanup_vessel_fragments
from .io_utils import (
    get_disease_vec,
    get_voxel_spacing,
    load_disease_map,
    load_nifti,
    resolve_case_id,
    save_nifti,
)

AVAILABLE_STEPS: List[str] = ["cc_cleanup", "adjacency_correction"]
DEFAULT_STEPS:   List[str] = ["cc_cleanup", "adjacency_correction"]


# ---------------------------------------------------------------------------
# Single-case pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    pred_path: str | Path,
    output_path: str | Path,
    disease_vec: Optional[List[int]] = None,
    steps: Optional[List[str]] = None,
    dilation_radius_mm: float = DEFAULT_DILATION_RADIUS_MM,
) -> Dict:
    """Run the post-processing pipeline on a single prediction file.

    Parameters
    ----------
    pred_path : path to input prediction NIfTI
    output_path : path to save the corrected prediction NIfTI
    disease_vec : binary disease flag vector (length 8), or ``None``
    steps : ordered list of steps to execute.
            Default: ``["cc_cleanup", "adjacency_correction"]``.
            Pass ``[]`` for a passthrough (copy only).
    dilation_radius_mm : adjacency dilation radius in mm

    Returns
    -------
    Dict with keys:
        ``input``, ``output``, ``steps_run``, ``cc_info``, ``correction``
    """
    if steps is None:
        steps = list(DEFAULT_STEPS)

    unknown = [s for s in steps if s not in AVAILABLE_STEPS]
    if unknown:
        raise ValueError(f"Unknown pipeline steps: {unknown}. Choose from {AVAILABLE_STEPS}")

    labels, affine, header = load_nifti(pred_path)
    labels = labels.astype(np.int32)
    spacing_mm = get_voxel_spacing(header)

    result: Dict = {
        "input":     str(pred_path),
        "output":    str(output_path),
        "steps_run": steps,
        "cc_info":   None,
        "correction": None,
    }

    if "cc_cleanup" in steps:
        labels, cc_info = cleanup_vessel_fragments(labels)
        result["cc_info"] = cc_info

    if "adjacency_correction" in steps:
        correction: CorrectionResult = correct_ao_pa_labels(
            labels,
            disease_vec=disease_vec,
            spacing_mm=spacing_mm,
            dilation_radius_mm=dilation_radius_mm,
        )
        labels = correction.corrected_labels
        result["correction"] = {
            "was_swapped":        correction.was_swapped,
            "skipped_reason":     correction.skipped_reason,
            "confidence_score":   round(correction.confidence_score, 4),
            "needs_manual_review": correction.needs_manual_review,
            "adjacency_details":  correction.adjacency_details,
        }

    save_nifti(labels, affine, header, output_path)
    return result


# ---------------------------------------------------------------------------
# Folder pipeline
# ---------------------------------------------------------------------------

def run_folder_pipeline(
    input_folder: str | Path,
    output_folder: str | Path,
    disease_map_path: Optional[str | Path] = None,
    steps: Optional[List[str]] = None,
    dilation_radius_mm: float = DEFAULT_DILATION_RADIUS_MM,
    file_pattern: str = "*.nii.gz",
) -> List[Dict]:
    """Run the pipeline on every prediction in *input_folder*.

    Parameters
    ----------
    input_folder : folder of nnU-Net prediction NIfTI files
    output_folder : destination folder for corrected files
    disease_map_path : path to ``disease_map.json``.
                       If omitted, all cases run without disease conditioning
                       (PuA exception is never triggered).
    steps : see :func:`run_pipeline`.  Default: full pipeline.
    dilation_radius_mm : adjacency dilation radius in mm
    file_pattern : glob pattern for input files

    Returns
    -------
    List of per-case result dicts (one per file).  Each dict has a
    ``"status"`` key (``"ok"`` or ``"error"``) and a ``"case_id"`` key.
    """
    input_folder  = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    disease_map = None
    if disease_map_path is not None:
        disease_map = load_disease_map(disease_map_path)

    pred_files = sorted(input_folder.glob(file_pattern))
    if not pred_files:
        raise FileNotFoundError(
            f"No files matching '{file_pattern}' found in {input_folder}"
        )

    all_results: List[Dict] = []
    for pred_path in pred_files:
        case_id = resolve_case_id(pred_path.name)
        disease_vec = None
        if disease_map is not None:
            disease_vec = get_disease_vec(disease_map, case_id)

        output_path = output_folder / pred_path.name
        try:
            res = run_pipeline(
                pred_path=pred_path,
                output_path=output_path,
                disease_vec=disease_vec,
                steps=steps,
                dilation_radius_mm=dilation_radius_mm,
            )
            res["case_id"] = case_id
            res["status"]  = "ok"
        except Exception as exc:
            res = {"case_id": case_id, "status": "error", "error": str(exc)}

        all_results.append(res)

    return all_results
