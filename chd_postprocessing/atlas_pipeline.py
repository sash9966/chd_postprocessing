"""Atlas-based post-processing pipeline.

Three experimental conditions
------------------------------

Condition 1 — Random atlas  (``mode="random_atlas"``, alias ``"baseline"``)
    Simulates having no disease knowledge at inference time.

    * **Atlas selection**: uniform-random from the GT training library.
    * **Perturbation**: ``create_synthetic_atlas`` is applied (±10° rotation,
      ±5 mm translation, ±5 % scale) to simulate a generic, imperfect
      reference atlas and keep the registration step non-trivial.
    * **Anatomy correction**: *skipped*.
    * **Adjacency correction**: *skipped*.
    * **IoC label correction**: component-level assignment for non-dominant
      fragments only.

Condition 2 — Disease atlas  (``mode="disease_atlas"``)
    Adds disease-matched atlas selection; no rule-based corrections.

    * **Atlas selection**: minimum Hamming distance on the 8-element disease
      flag vector.
    * **Perturbation**: *skipped* — the disease-matched atlas is used as-is.
    * **Anatomy correction**: *skipped* — isolates the atlas-matching gain.
    * **Adjacency correction**: *skipped*.
    * **IoC label correction**: component-level assignment for non-dominant
      fragments only.

Condition 3 — Disease atlas + rules  (``mode="disease_atlas_rules"``,
alias ``"disease_specific"``)
    Adds disease-aware anatomy priors and adjacency-graph correction on top of
    condition 2.

    * **Atlas selection**: same as condition 2 (minimum Hamming distance).
    * **Perturbation**: *skipped*.
    * **Anatomy correction**: ``correct_ao_pa_fragments`` runs *first*
      using disease-aware rules (TGA reversed, DORV both-from-RV, ToF
      unconstrained AO, HLHS/PuA skipped).
    * **Adjacency correction**: ``correct_by_adjacency`` runs after IoC
      correction as a final structural refinement.
    * **IoC label correction**: component-level assignment for non-dominant
      fragments only.

Usage
-----
Single file::

    from chd_postprocessing.atlas_pipeline import run_atlas_pipeline
    result = run_atlas_pipeline(
        pred_path        = "pred.nii.gz",
        gt_folder        = "atlases/",          # pre-built disease atlas library
        output_path      = "corrected.nii.gz",
        mode             = "disease_atlas_rules",
        disease_map_path = "disease_map.json",
        gt_path          = "gt.nii.gz",         # optional, enables Dice evaluation
    )

Whole folder::

    from chd_postprocessing.atlas_pipeline import run_atlas_folder_pipeline
    df = run_atlas_folder_pipeline("preds/", "atlases/", "corrected/",
                                   mode="disease_atlas_rules",
                                   disease_map_path="disease_map.json")
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from .adjacency_correction import correct_by_adjacency
from .anatomy_priors import correct_ao_pa_fragments
from .atlas import AtlasLibrary, create_synthetic_atlas
from .config import FOREGROUND_CLASSES, LABEL_NAMES
from .evaluate import dice_per_class
from .io_utils import get_disease_vec, get_voxel_spacing, load_disease_map, load_nifti, save_nifti, resolve_case_id
from .label_correction import correct_labels_with_atlas
from .registration import register_atlas_per_structure, register_atlas_to_pred


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AtlasPipelineResult:
    """Complete output of one atlas-based post-processing run.

    Attributes
    ----------
    corrected_labels : post-processed integer label volume.
    affine, header : geometry metadata from the input prediction NIfTI.
    atlas_case_id : case ID of the atlas that was used.
    atlas_disease_name : human-readable disease profile of the atlas.
    mode : ``"baseline"`` or ``"disease_specific"``.
    was_relabeled : True if any label assignment was changed.
    mapping_applied : dict ``{pred_label: corrected_label}``.
    reassignment_summary : human-readable description of changes.
    overlap_matrix : (N, N) Dice overlap matrix (pred rows, atlas cols).
    component_info : per-label CC statistics.
    dice_before : per-class Dice scores before correction (None if GT absent).
    dice_after : per-class Dice scores after correction (None if GT absent).
    """
    corrected_labels:     np.ndarray
    affine:               np.ndarray
    header:               object

    atlas_case_id:        str
    atlas_disease_name:   str
    mode:                 str

    was_relabeled:        bool
    mapping_applied:      Dict[int, int]
    reassignment_summary: str
    overlap_matrix:       np.ndarray
    label_ids:            List[int]
    component_info:       Dict

    dice_before:    Optional[Dict[str, float]] = None
    dice_after:     Optional[Dict[str, float]] = None
    adjacency_log:  Optional[List[Dict]]       = None
    boundary_log:   Optional[List[Dict]]       = None

    def dice_delta(self) -> Optional[Dict[str, float]]:
        """Per-class Dice improvement (after − before).  None if GT unavailable."""
        if self.dice_before is None or self.dice_after is None:
            return None
        return {
            cls: (self.dice_after[cls] - self.dice_before[cls])
            for cls in self.dice_before
        }

    def summary_dict(self) -> Dict:
        """Flat dict suitable for building a results DataFrame row."""
        d: Dict = {
            "atlas_case_id":      self.atlas_case_id,
            "atlas_disease_name": self.atlas_disease_name,
            "mode":               self.mode,
            "was_relabeled":      self.was_relabeled,
        }
        if self.dice_before is not None:
            for cls, v in self.dice_before.items():
                d[f"dice_before_{cls}"] = round(v, 4) if not np.isnan(v) else float("nan")
        if self.dice_after is not None:
            for cls, v in self.dice_after.items():
                d[f"dice_after_{cls}"] = round(v, 4) if not np.isnan(v) else float("nan")
        delta = self.dice_delta()
        if delta is not None:
            for cls, v in delta.items():
                d[f"dice_delta_{cls}"] = round(v, 4) if not np.isnan(v) else float("nan")
            non_nan = [v for v in delta.values() if not np.isnan(v)]
            d["mean_dice_delta"] = round(float(np.mean(non_nan)), 4) if non_nan else float("nan")
        return d


# ---------------------------------------------------------------------------
# Single-case pipeline
# ---------------------------------------------------------------------------

def run_atlas_pipeline(
    pred_path:          str | Path,
    gt_folder:          str | Path,
    output_path:        Optional[str | Path] = None,
    disease_map_path:   Optional[str | Path] = None,
    gt_path:            Optional[str | Path] = None,
    mode:               str = "random_atlas",
    seed:               int = 42,
    registration_mode:  str = "per_structure",
    min_overlap:        float = 0.10,
    min_component_fraction: float = 0.01,
    max_reassign_fraction: float = 0.15,
    do_perturbation:    Optional[bool] = None,
    do_anatomy_correction: Optional[bool] = None,
    do_adjacency_correction: Optional[bool] = None,
    do_boundary_refinement: Optional[bool] = None,
    do_morphological_cleanup: bool = True,
    label_ids:          Optional[List[int]] = None,
) -> AtlasPipelineResult:
    """Run the atlas-based post-processing pipeline on a single prediction.

    Parameters
    ----------
    pred_path : path to the nnU-Net prediction NIfTI.
    gt_folder : folder containing atlas NIfTI files (pre-built disease library
                or raw training GT folder).
    output_path : where to save the corrected NIfTI.  Skipped if ``None``.
    disease_map_path : path to ``disease_map.json``.
                       Required for disease-matched modes.
    gt_path : ground-truth segmentation for the current case.
              When supplied, Dice scores before/after correction are computed.
    mode : one of ``"random_atlas"``, ``"disease_atlas"``,
           ``"disease_atlas_rules"`` (or the backward-compatible aliases
           ``"baseline"`` → ``"random_atlas"`` and
           ``"disease_specific"`` → ``"disease_atlas_rules"``).
           Controls atlas selection and the default values of the three
           *do_** flags below.

           ================================  ==========  =======  =========
           mode                              perturbation anatomy  adjacency
           ================================  ==========  =======  =========
           random_atlas  (baseline)          True        False    False
           disease_atlas                     False       False    False
           disease_atlas_rules               False       True     True
           (disease_specific)
           ================================  ==========  =======  =========

    seed : random seed for reproducibility (atlas selection + perturbation).
    registration_mode : ``"per_structure"`` (default), ``"centroid"``, or
                        ``"pca"``.
    min_overlap : minimum IoC for the best atlas match to be accepted when
                  reassigning a non-dominant fragment.  Default 0.10 prevents
                  noise assignments under coarse centroid registration.
    min_component_fraction : small-fragment conflict-resolution threshold.
    do_perturbation : override the mode default for atlas perturbation.
    do_anatomy_correction : override the mode default for AO/PA fragment
                            correction (disease-aware rules applied).
    do_adjacency_correction : override the mode default for adjacency-graph
                              correction (runs after IoC step).
    do_morphological_cleanup : apply binary closing + hole-fill to AO and PA.
    label_ids : foreground labels to consider.  Default: 1–7.

    Returns
    -------
    :class:`AtlasPipelineResult`
    """
    # Backward-compatible mode aliases
    _MODE_ALIASES = {
        "baseline":        "random_atlas",
        "disease_specific": "disease_atlas_rules",
    }
    mode = _MODE_ALIASES.get(mode, mode)

    _VALID_MODES = {"random_atlas", "disease_atlas", "disease_atlas_rules"}
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unknown mode: {mode!r}. "
            f"Use one of {sorted(_VALID_MODES)} (or 'baseline'/'disease_specific')."
        )

    if label_ids is None:
        label_ids = list(FOREGROUND_CLASSES)

    # Apply per-mode defaults for unset flags
    if do_perturbation is None:
        do_perturbation = (mode == "random_atlas")
    if do_anatomy_correction is None:
        do_anatomy_correction = (mode == "disease_atlas_rules")
    if do_adjacency_correction is None:
        do_adjacency_correction = (mode == "disease_atlas_rules")
    if do_boundary_refinement is None:
        do_boundary_refinement = True

    rng = random.Random(seed)   # seeded once; used for atlas selection then perturbation

    # ------------------------------------------------------------------
    # 1. Load prediction
    # ------------------------------------------------------------------
    pred_labels, affine, header = load_nifti(pred_path)
    pred_labels  = pred_labels.astype(np.int32)
    pred_spacing = get_voxel_spacing(header)

    case_id = resolve_case_id(Path(pred_path).name)

    # ------------------------------------------------------------------
    # 2. Build atlas library
    # ------------------------------------------------------------------
    library = AtlasLibrary.load_all(
        gt_folder,
        disease_map_path=disease_map_path,
    )
    if len(library) == 0:
        raise FileNotFoundError(f"No GT files found in {gt_folder}")

    # ------------------------------------------------------------------
    # 3. Determine disease vector for this case
    # ------------------------------------------------------------------
    disease_vec = [0] * 8
    if disease_map_path is not None:
        dm = load_disease_map(disease_map_path)
        vec = get_disease_vec(dm, case_id)
        if vec is not None:
            disease_vec = vec

    # ------------------------------------------------------------------
    # 3.5 Determine protected labels (vessel labels for skip_ao_pa diseases)
    #     PuA and HLHS have fused/absent vessels — atlas-based reassignment
    #     of AO/PA is unreliable and causes Dice drops.  Lock those labels.
    # ------------------------------------------------------------------
    from .config import DISEASE_ANATOMY_RULES
    _protected: List[int] = []
    if disease_vec is not None:
        for _flag_idx, _is_active in enumerate(disease_vec):
            if _is_active and _flag_idx in DISEASE_ANATOMY_RULES:
                if DISEASE_ANATOMY_RULES[_flag_idx].get("skip_ao_pa_correction", False):
                    _protected.extend([6, 7])  # AO and PA
    protected_labels: Optional[List[int]] = list(set(_protected)) if _protected else None

    # ------------------------------------------------------------------
    # 4. Select atlas
    # ------------------------------------------------------------------
    selection_mode = "random" if mode == "random_atlas" else "best_match"
    # Exclude both the plain case ID and the _image variant so that the
    # test case is never selected as its own atlas regardless of filename style.
    _base_id = case_id[:-6] if case_id.endswith("_image") else case_id
    atlas_entry = library.select_for_case(
        disease_vec=disease_vec,
        rng=rng,
        mode=selection_mode,
        exclude_case_ids=[case_id, _base_id, _base_id + "_image"],
    )
    atlas_entry.load()
    atlas_labels = atlas_entry.labels

    # ------------------------------------------------------------------
    # 5. Optional atlas perturbation (baseline mode only)
    #    Simulates a generic, imperfect reference atlas and makes the
    #    registration step non-trivial.
    # ------------------------------------------------------------------
    if do_perturbation:
        atlas_labels = create_synthetic_atlas(
            atlas_labels, atlas_entry.spacing, rng
        )

    # ------------------------------------------------------------------
    # 6. Fragment-level AO/PA anatomy correction (no registration needed)
    #    Runs FIRST so atlas step operates on already-corrected vessel labels.
    #    disease_specific mode only; baseline skips this step.
    # ------------------------------------------------------------------
    working_labels = pred_labels.copy()
    if do_anatomy_correction:
        working_labels, frag_log = correct_ao_pa_fragments(
            working_labels, disease_vec, pred_spacing
        )
        n_frag = len(frag_log.get("reassigned", []))
        print(f"  Anatomy priors: {n_frag} fragments reassigned")

    # ------------------------------------------------------------------
    # 7. Compute Dice *before* atlas correction (optional)
    # ------------------------------------------------------------------
    dice_before = None
    if gt_path is not None:
        gt_labels, _, _ = load_nifti(gt_path)
        gt_labels = gt_labels.astype(np.int32)
        raw_scores = dice_per_class(pred_labels, gt_labels, label_ids)
        dice_before = {LABEL_NAMES.get(k, str(k)): v for k, v in raw_scores.items()}

    # ------------------------------------------------------------------
    # 8. Register atlas → prediction space
    #    "per_structure" aligns each label independently for better IoC
    #    matching of structures far from the whole-heart centroid.
    # ------------------------------------------------------------------
    if registration_mode == "per_structure":
        atlas_masks_override = register_atlas_per_structure(
            atlas_labels, working_labels, label_ids
        )
        # Also compute a globally-registered atlas for display purposes
        registered_atlas = register_atlas_to_pred(
            atlas_labels, working_labels, pred_spacing, mode="centroid"
        )
    else:
        registered_atlas = register_atlas_to_pred(
            atlas_labels, working_labels, pred_spacing, mode=registration_mode
        )
        atlas_masks_override = None

    # ------------------------------------------------------------------
    # 9. Component-level label correction (non-dominant fragments only)
    # ------------------------------------------------------------------
    correction = correct_labels_with_atlas(
        working_labels, registered_atlas,
        label_ids=label_ids,
        min_overlap=min_overlap,
        min_component_fraction=min_component_fraction,
        max_reassign_fraction=max_reassign_fraction,
        do_morphological_cleanup=do_morphological_cleanup,
        atlas_masks_override=atlas_masks_override,
        protected_labels=protected_labels,
    )
    print(f"  IoC correction: {'changes made' if correction.was_relabeled else 'no changes'}")

    # ------------------------------------------------------------------
    # 9.5 Adjacency-graph correction (disease_atlas_rules mode only)
    #     Runs after IoC correction as a structural refinement step.
    # ------------------------------------------------------------------
    adjacency_log: Optional[List[Dict]] = None
    final_labels = correction.corrected_labels
    if do_adjacency_correction:
        final_labels, adjacency_log = correct_by_adjacency(
            correction.corrected_labels,
            registered_atlas,
            label_ids=label_ids,
            disease_vec=disease_vec,
            min_component_fraction=min_component_fraction,
            protected_labels=protected_labels,
        )
    print(f"  Adjacency correction: {len(adjacency_log or [])} corrections")

    # ------------------------------------------------------------------
    # 9.7 Per-voxel boundary refinement (all modes)
    # ------------------------------------------------------------------
    from .boundary_refinement import refine_all_boundaries
    boundary_log_list: Optional[List[Dict]] = None
    if do_boundary_refinement:
        final_labels, boundary_log_list = refine_all_boundaries(
            final_labels,
            atlas_reg=registered_atlas,
            label_ids=label_ids,
            disease_vec=disease_vec,
            width_voxels=3,
            min_confidence=0.6,
            protected_labels=protected_labels,
        )
        total_br = sum(e.get("a_to_b", 0) + e.get("b_to_a", 0) for e in boundary_log_list)
        print(f"  Boundary refinement: {total_br} voxels reassigned")

    # ------------------------------------------------------------------
    # 10. Compute Dice *after* correction (optional)
    # ------------------------------------------------------------------
    dice_after = None
    if gt_path is not None:
        raw_after = dice_per_class(final_labels, gt_labels, label_ids)
        dice_after = {LABEL_NAMES.get(k, str(k)): v for k, v in raw_after.items()}

    # ------------------------------------------------------------------
    # 11. Save output (optional)
    # ------------------------------------------------------------------
    if output_path is not None:
        save_nifti(final_labels, affine, header, output_path)

    return AtlasPipelineResult(
        corrected_labels=final_labels,
        affine=affine,
        header=header,
        atlas_case_id=atlas_entry.case_id,
        atlas_disease_name=atlas_entry.disease_name,
        mode=mode,
        was_relabeled=correction.was_relabeled,
        mapping_applied=correction.mapping_applied,
        reassignment_summary=correction.reassignment_summary,
        overlap_matrix=correction.overlap_matrix,
        label_ids=correction.label_ids,
        component_info=correction.component_info,
        dice_before=dice_before,
        dice_after=dice_after,
        adjacency_log=adjacency_log,
        boundary_log=boundary_log_list,
    )


# ---------------------------------------------------------------------------
# Folder pipeline
# ---------------------------------------------------------------------------

def run_atlas_folder_pipeline(
    pred_folder:        str | Path,
    gt_folder:          str | Path,
    output_folder:      str | Path,
    disease_map_path:   Optional[str | Path] = None,
    gt_folder_eval:     Optional[str | Path] = None,
    mode:               str = "baseline",
    seed:               int = 42,
    registration_mode:  str = "centroid",
    file_pattern:       str = "*.nii.gz",
    **kwargs,
) -> pd.DataFrame:
    """Run the atlas pipeline on every prediction in *pred_folder*.

    Parameters
    ----------
    pred_folder : folder of nnU-Net prediction NIfTI files.
    gt_folder : training GT folder (atlas library).
    output_folder : destination for corrected NIfTI files.
    disease_map_path : path to ``disease_map.json``.
    gt_folder_eval : if provided, per-case GT is looked up here for
                     Dice evaluation.
    mode : ``"baseline"`` or ``"disease_specific"``.
    seed : base random seed (each case gets a derived seed for reproducibility).
    **kwargs : forwarded to :func:`run_atlas_pipeline`.

    Returns
    -------
    DataFrame with one row per case containing Dice before/after and a
    summary of label changes.
    """
    pred_folder   = Path(pred_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_folder.glob(file_pattern))
    if not pred_files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' in {pred_folder}")

    rows = []
    for idx, pred_path in enumerate(pred_files):
        case_id = resolve_case_id(pred_path.name)
        out_path = output_folder / pred_path.name

        # Optional GT lookup for evaluation
        gt_path = None
        if gt_folder_eval is not None:
            gt_folder_eval = Path(gt_folder_eval)
            for suffix in [".nii.gz", ".nii"]:
                cand = gt_folder_eval / f"{case_id}{suffix}"
                if cand.exists():
                    gt_path = cand
                    break
                cand_img = gt_folder_eval / f"{case_id}_image{suffix}"
                if cand_img.exists():
                    gt_path = cand_img
                    break

        try:
            result = run_atlas_pipeline(
                pred_path=pred_path,
                gt_folder=gt_folder,
                output_path=out_path,
                disease_map_path=disease_map_path,
                gt_path=gt_path,
                mode=mode,
                seed=seed + idx,   # deterministic per-case seed
                registration_mode=registration_mode,
                **kwargs,
            )
            row = {"case_id": case_id, "status": "ok"}
            row.update(result.summary_dict())
        except Exception as exc:
            row = {"case_id": case_id, "status": "error", "error": str(exc)}
            print(f"  [ERROR] {case_id}: {exc}")

        rows.append(row)
        print(
            f"  [{idx + 1}/{len(pred_files)}] {case_id}: "
            f"atlas={row.get('atlas_case_id', '?')}  "
            f"relabeled={row.get('was_relabeled', '?')}"
        )

    df = pd.DataFrame(rows)
    return df
