"""Tests for the atlas-based post-processing pipeline.

All tests are self-contained — no NIfTI files on disk are required.
Synthetic label volumes are constructed in-memory so the correct
outcomes are known analytically.

Volume layout (40 × 40 × 40 voxels unless noted)
-------------------------------------------------
Labels 1-7 occupy non-overlapping 4-voxel-wide bands along the x-axis:
  label k  →  x = [4*(k-1) : 4*k],  all y, all z
  background (0) fills the rest.
"""
from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np
import pytest

from chd_postprocessing.atlas import AtlasEntry, AtlasLibrary, create_synthetic_atlas
from chd_postprocessing.atlas_pipeline import run_atlas_pipeline, AtlasPipelineResult
from chd_postprocessing.config import FOREGROUND_CLASSES, LABELS
from chd_postprocessing.label_correction import (
    LabelCorrectionResult,
    compute_overlap_matrix,
    correct_labels_with_atlas,
    enforce_single_component,
    optimal_label_mapping,
)
from chd_postprocessing.registration import register_atlas_to_pred


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SHAPE = (40, 40, 40)
SPACING = (1.0, 1.0, 1.0)
LABEL_IDS = list(FOREGROUND_CLASSES)   # [1, 2, 3, 4, 5, 6, 7]


def _make_label_volume(shape=SHAPE) -> np.ndarray:
    """Create a deterministic label volume: label k occupies x=[4(k-1):4k]."""
    vol = np.zeros(shape, dtype=np.int32)
    for k in LABEL_IDS:
        x0 = 4 * (k - 1)
        x1 = 4 * k
        vol[x0:x1, :, :] = k
    return vol


def _save_nifti(data: np.ndarray, path: Path) -> None:
    """Save a numpy array as a NIfTI file with identity affine."""
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, str(path))


# ---------------------------------------------------------------------------
# 1. AtlasEntry and AtlasLibrary
# ---------------------------------------------------------------------------

class TestAtlasLibrary:

    def test_load_all_finds_files(self, tmp_path):
        """load_all() should discover all *.nii.gz files in the folder."""
        vol = _make_label_volume()
        for case_id in ["case_001", "case_002", "case_003"]:
            _save_nifti(vol, tmp_path / f"{case_id}.nii.gz")

        lib = AtlasLibrary.load_all(tmp_path)
        assert len(lib) == 3

    def test_load_all_empty_folder(self, tmp_path):
        """An empty folder should yield an empty library (not a crash)."""
        lib = AtlasLibrary.load_all(tmp_path)
        assert len(lib) == 0

    def test_case_ids_parsed_correctly(self, tmp_path):
        """Case IDs should be the filename stem without .nii.gz extension."""
        vol = _make_label_volume()
        _save_nifti(vol, tmp_path / "patient_042.nii.gz")

        lib = AtlasLibrary.load_all(tmp_path)
        assert lib.entries[0].case_id == "patient_042"

    def test_lazy_load(self, tmp_path):
        """Labels should be None until entry.load() is called."""
        vol = _make_label_volume()
        _save_nifti(vol, tmp_path / "case_001.nii.gz")

        lib = AtlasLibrary.load_all(tmp_path)
        entry = lib.entries[0]
        assert entry.labels is None        # not yet loaded
        entry.load()
        assert entry.labels is not None    # now loaded

    def test_select_random(self, tmp_path):
        """Random mode should return one of the available entries."""
        vol = _make_label_volume()
        for i in range(5):
            _save_nifti(vol, tmp_path / f"case_{i:03d}.nii.gz")

        lib = AtlasLibrary.load_all(tmp_path)
        rng = random.Random(0)
        entry = lib.select_for_case([0]*8, rng, mode="random")
        assert entry in lib.entries

    def test_select_best_match_prefers_exact(self, tmp_path):
        """best_match should prefer the entry whose disease_vec equals the query."""
        vol = _make_label_volume()
        _save_nifti(vol, tmp_path / "normal.nii.gz")
        _save_nifti(vol, tmp_path / "hlhs.nii.gz")

        dm = {"normal": [0]*8, "hlhs": [1, 0, 0, 0, 0, 0, 0, 0]}
        import json
        dm_path = tmp_path / "disease_map.json"
        dm_path.write_text(json.dumps(dm))

        lib = AtlasLibrary.load_all(tmp_path, disease_map_path=dm_path)
        rng = random.Random(0)
        selected = lib.select_for_case([1, 0, 0, 0, 0, 0, 0, 0], rng, mode="best_match")
        assert selected.case_id == "hlhs"

    def test_exclude_self(self, tmp_path):
        """The current case should never be selected as its own atlas."""
        vol = _make_label_volume()
        _save_nifti(vol, tmp_path / "case_001.nii.gz")
        _save_nifti(vol, tmp_path / "case_002.nii.gz")

        lib = AtlasLibrary.load_all(tmp_path)
        rng = random.Random(0)
        for _ in range(20):
            entry = lib.select_for_case([0]*8, rng, mode="random",
                                        exclude_case_id="case_001")
            assert entry.case_id != "case_001"

    def test_hamming_distance(self, tmp_path):
        vol = _make_label_volume()
        _save_nifti(vol, tmp_path / "case_001.nii.gz")
        lib = AtlasLibrary.load_all(tmp_path)
        entry = lib.entries[0]
        entry.disease_vec = [1, 0, 0, 0, 0, 0, 0, 0]
        assert entry.hamming_distance([1, 0, 0, 0, 0, 0, 0, 0]) == 0
        assert entry.hamming_distance([0, 0, 0, 0, 0, 0, 0, 0]) == 1
        assert entry.hamming_distance([1, 1, 0, 0, 0, 0, 0, 0]) == 1


# ---------------------------------------------------------------------------
# 2. Synthetic atlas creation
# ---------------------------------------------------------------------------

class TestCreateSyntheticAtlas:

    def test_same_shape(self):
        """Perturbed atlas must have the same shape as the input."""
        vol = _make_label_volume()
        rng = random.Random(42)
        synth = create_synthetic_atlas(vol, SPACING, rng)
        assert synth.shape == vol.shape

    def test_dtype_preserved(self):
        """Integer dtype must be preserved (nearest-neighbour interpolation)."""
        vol = _make_label_volume().astype(np.int32)
        rng = random.Random(0)
        synth = create_synthetic_atlas(vol, SPACING, rng)
        assert np.issubdtype(synth.dtype, np.integer)

    def test_labels_subset_preserved(self):
        """Perturbed volume should only contain values present in the original."""
        vol = _make_label_volume()
        original_labels = set(np.unique(vol).tolist())
        rng = random.Random(7)
        synth = create_synthetic_atlas(vol, SPACING, rng)
        synth_labels = set(np.unique(synth).tolist())
        assert synth_labels.issubset(original_labels)

    def test_zero_perturbation_is_identity(self):
        """With rot_deg=0, trans_mm=0, scale_range=0 the output must equal input."""
        vol = _make_label_volume()
        rng = random.Random(0)
        synth = create_synthetic_atlas(vol, SPACING, rng,
                                       rot_deg=0.0, trans_mm=0.0, scale_range=0.0)
        assert np.array_equal(synth, vol)

    def test_foreground_survives_small_perturbation(self):
        """A small perturbation should keep most foreground voxels intact."""
        vol = _make_label_volume()
        fg_before = int((vol > 0).sum())
        rng = random.Random(1)
        synth = create_synthetic_atlas(vol, SPACING, rng,
                                       rot_deg=5.0, trans_mm=2.0, scale_range=0.02)
        fg_after = int((synth > 0).sum())
        # At least 70 % of foreground should survive a tiny perturbation
        assert fg_after >= 0.70 * fg_before


# ---------------------------------------------------------------------------
# 3. Registration
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_output_shape_matches_pred(self):
        """Registered atlas must have pred's shape even if atlas has different shape."""
        atlas = _make_label_volume((50, 50, 50))
        pred  = _make_label_volume((40, 40, 40))
        reg   = register_atlas_to_pred(atlas, pred, SPACING, mode="centroid")
        assert reg.shape == pred.shape

    def test_centroid_alignment_reduces_error(self):
        """Centroid registration should reduce centroid distance vs no alignment."""
        pred  = _make_label_volume()
        # Shift atlas by 8 voxels along x
        atlas = np.roll(pred, 8, axis=0)

        def cm_dist(a, b):
            from scipy.ndimage import center_of_mass
            ca = np.array(center_of_mass((a > 0).astype(np.uint8)))
            cb = np.array(center_of_mass((b > 0).astype(np.uint8)))
            return float(np.linalg.norm(ca - cb))

        dist_before = cm_dist(atlas, pred)
        reg = register_atlas_to_pred(atlas, pred, SPACING, mode="centroid")
        dist_after = cm_dist(reg, pred)
        assert dist_after < dist_before, (
            f"Centroid registration should reduce distance "
            f"(before={dist_before:.2f}, after={dist_after:.2f})"
        )

    def test_pca_mode_runs(self):
        """PCA mode should run without error and return the correct shape."""
        pred  = _make_label_volume()
        atlas = np.roll(pred, 4, axis=0)
        reg   = register_atlas_to_pred(atlas, pred, SPACING, mode="pca")
        assert reg.shape == pred.shape

    def test_empty_atlas_handled(self):
        """An all-zero atlas should not raise an exception."""
        pred  = _make_label_volume()
        atlas = np.zeros_like(pred)
        reg   = register_atlas_to_pred(atlas, pred, SPACING)
        assert reg.shape == pred.shape

    def test_empty_pred_handled(self):
        """An all-zero prediction should not raise."""
        atlas = _make_label_volume()
        pred  = np.zeros_like(atlas)
        reg   = register_atlas_to_pred(atlas, pred, SPACING)
        assert reg.shape == pred.shape

    def test_invalid_mode_raises(self):
        vol = _make_label_volume()
        with pytest.raises(ValueError, match="Unknown registration mode"):
            register_atlas_to_pred(vol, vol, SPACING, mode="bad_mode")


# ---------------------------------------------------------------------------
# 4. Label correction
# ---------------------------------------------------------------------------

class TestOverlapMatrix:

    def test_diagonal_high_for_identical_volumes(self):
        """When pred == atlas, every diagonal entry should be 1.0."""
        vol = _make_label_volume()
        M, ids = compute_overlap_matrix(vol, vol, LABEL_IDS)
        for i in range(len(ids)):
            assert abs(M[i, i] - 1.0) < 1e-6, f"M[{i},{i}] = {M[i,i]:.4f}, expected 1.0"

    def test_off_diagonal_zero_for_identical_non_overlapping(self):
        """Non-overlapping labels in the same volume → off-diagonal Dice = 0."""
        vol = _make_label_volume()
        M, _ = compute_overlap_matrix(vol, vol, LABEL_IDS)
        N = len(LABEL_IDS)
        for i in range(N):
            for j in range(N):
                if i != j:
                    assert M[i, j] == 0.0

    def test_shape_is_n_by_n(self):
        vol = _make_label_volume()
        M, ids = compute_overlap_matrix(vol, vol, LABEL_IDS)
        assert M.shape == (len(LABEL_IDS), len(LABEL_IDS))
        assert ids == LABEL_IDS


class TestOptimalLabelMapping:

    def test_identity_mapping_for_identical_volumes(self):
        """Identical pred and atlas should return the identity mapping."""
        vol = _make_label_volume()
        M, ids = compute_overlap_matrix(vol, vol)
        mapping = optimal_label_mapping(M, ids)
        for lbl in ids:
            assert mapping[lbl] == lbl

    def test_detects_swap(self):
        """When two labels are swapped in pred, the mapping should swap them back."""
        vol_correct = _make_label_volume()
        vol_swapped = vol_correct.copy()
        # Swap labels 6 (AO) and 7 (PA)
        ao_mask = vol_correct == 6
        pa_mask = vol_correct == 7
        vol_swapped[ao_mask] = 7
        vol_swapped[pa_mask] = 6

        M, ids = compute_overlap_matrix(vol_swapped, vol_correct)
        mapping = optimal_label_mapping(M, ids)
        assert mapping[6] == 7 or mapping[7] == 6, (
            f"Expected swap to be detected; mapping={mapping}"
        )

    def test_empty_labels_map_to_identity(self):
        """Labels absent from both pred and atlas should map to themselves."""
        vol = np.zeros(SHAPE, dtype=np.int32)
        vol[0:4, :, :] = 1   # only label 1 present
        M, ids = compute_overlap_matrix(vol, vol, LABEL_IDS)
        mapping = optimal_label_mapping(M, ids)
        # Labels 2-7 are absent; they should map to themselves
        for lbl in LABEL_IDS[1:]:
            assert mapping[lbl] == lbl


class TestApplyMapping:

    def test_no_change_for_identity(self):
        from chd_postprocessing.label_correction import apply_label_mapping
        vol = _make_label_volume()
        identity = {lbl: lbl for lbl in LABEL_IDS}
        result, changed = apply_label_mapping(vol, identity)
        assert not changed
        assert np.array_equal(result, vol)

    def test_swap_applied_correctly(self):
        from chd_postprocessing.label_correction import apply_label_mapping
        vol = _make_label_volume()
        mapping = {lbl: lbl for lbl in LABEL_IDS}
        mapping[6] = 7
        mapping[7] = 6
        result, changed = apply_label_mapping(vol, mapping)
        assert changed
        assert (result[vol == 6] == 7).all()
        assert (result[vol == 7] == 6).all()
        # Other labels untouched
        for lbl in [1, 2, 3, 4, 5]:
            assert (result[vol == lbl] == lbl).all()


class TestEnforceSingleComponent:

    def test_single_component_unchanged(self):
        """A label with only one connected component should not lose any voxels."""
        vol = _make_label_volume()
        counts_before = {lbl: int((vol == lbl).sum()) for lbl in LABEL_IDS}
        cleaned, info = enforce_single_component(vol, LABEL_IDS)
        for lbl in LABEL_IDS:
            assert int((cleaned == lbl).sum()) == counts_before[lbl]
            assert info[lbl]["n_components"] == 1
            assert info[lbl]["removed"] == 0

    def test_small_fragment_removed(self):
        """A tiny isolated fragment (< min_fraction * largest) should become 0."""
        vol = _make_label_volume()
        # Add a 1-voxel isolated fragment of label 1 far from its main block
        vol[35, 35, 35] = 1

        cleaned, info = enforce_single_component(vol, [1], min_component_fraction=0.01)
        # The fragment (1 voxel) should be removed
        assert info[1]["removed"] >= 1
        assert vol[35, 35, 35] == 1        # original unchanged
        assert cleaned[35, 35, 35] == 0   # fragment gone in output


class TestCorrectLabelsWithAtlas:

    def test_identity_when_pred_equals_atlas(self):
        """Identical pred and atlas → no relabeling, no CC removal."""
        vol = _make_label_volume()
        result = correct_labels_with_atlas(vol, vol, label_ids=LABEL_IDS)
        assert isinstance(result, LabelCorrectionResult)
        assert not result.was_relabeled
        for lbl in LABEL_IDS:
            assert result.mapping_applied[lbl] == lbl

    def test_swap_corrected(self):
        """Labels swapped in pred should be corrected to match the atlas."""
        vol_correct = _make_label_volume()
        vol_swapped = vol_correct.copy()
        ao_mask = vol_correct == 6
        pa_mask = vol_correct == 7
        vol_swapped[ao_mask] = 7
        vol_swapped[pa_mask] = 6

        result = correct_labels_with_atlas(vol_swapped, vol_correct,
                                           label_ids=LABEL_IDS,
                                           do_morphological_cleanup=False)
        # After correction the corrected volume should match vol_correct
        assert result.was_relabeled
        assert np.array_equal(result.corrected_labels, vol_correct)

    def test_returns_correct_type(self):
        vol = _make_label_volume()
        result = correct_labels_with_atlas(vol, vol)
        assert hasattr(result, "corrected_labels")
        assert hasattr(result, "overlap_matrix")
        assert hasattr(result, "mapping_applied")
        assert hasattr(result, "was_relabeled")
        assert hasattr(result, "reassignment_summary")


# ---------------------------------------------------------------------------
# 5. Full pipeline (run_atlas_pipeline)
# ---------------------------------------------------------------------------

class TestRunAtlasPipeline:

    def _write_atlas_library(self, tmp_path: Path, n: int = 5) -> Path:
        """Write *n* GT NIfTI files to a folder and return the folder path."""
        gt_dir = tmp_path / "labelsTr"
        gt_dir.mkdir()
        vol = _make_label_volume()
        for i in range(n):
            _save_nifti(vol, gt_dir / f"case_{i:03d}.nii.gz")
        return gt_dir

    def test_output_shape_matches_input(self, tmp_path):
        """Corrected output must have the same spatial shape as the prediction."""
        gt_dir  = self._write_atlas_library(tmp_path)
        pred    = _make_label_volume()
        pred_path = tmp_path / "pred_case_999.nii.gz"
        _save_nifti(pred, pred_path)

        result = run_atlas_pipeline(
            pred_path=pred_path,
            gt_folder=gt_dir,
            mode="baseline",
            seed=0,
        )
        assert isinstance(result, AtlasPipelineResult)
        assert result.corrected_labels.shape == pred.shape

    def test_result_has_expected_attributes(self, tmp_path):
        gt_dir    = self._write_atlas_library(tmp_path)
        pred_path = tmp_path / "pred_case_999.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)

        result = run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir, seed=0)
        assert hasattr(result, "atlas_case_id")
        assert hasattr(result, "atlas_disease_name")
        assert hasattr(result, "was_relabeled")
        assert hasattr(result, "overlap_matrix")
        assert result.mode == "baseline"

    def test_output_saved_to_disk(self, tmp_path):
        """When output_path is provided, a NIfTI file should be written."""
        gt_dir    = self._write_atlas_library(tmp_path)
        pred_path = tmp_path / "pred_case_999.nii.gz"
        out_path  = tmp_path / "corrected_case_999.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)

        run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir,
                           output_path=out_path, seed=0)
        assert out_path.exists()

    def test_no_save_when_output_path_none(self, tmp_path):
        """When output_path=None, no file should be written."""
        gt_dir    = self._write_atlas_library(tmp_path)
        pred_path = tmp_path / "pred_case_999.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)

        run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir,
                           output_path=None, seed=0)
        niftis = list(tmp_path.glob("corrected*.nii.gz"))
        assert len(niftis) == 0

    def test_dice_computed_when_gt_provided(self, tmp_path):
        """Providing gt_path should populate dice_before and dice_after."""
        gt_dir    = self._write_atlas_library(tmp_path)
        pred_path = tmp_path / "pred_case_999.nii.gz"
        gt_path   = tmp_path / "gt_case_999.nii.gz"
        vol = _make_label_volume()
        _save_nifti(vol, pred_path)
        _save_nifti(vol, gt_path)

        result = run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir,
                                    gt_path=gt_path, seed=0)
        assert result.dice_before is not None
        assert result.dice_after  is not None
        # Identical pred and GT → Dice should be 1.0 for all present classes
        for cls, v in result.dice_before.items():
            if not np.isnan(v):
                assert abs(v - 1.0) < 1e-6, f"dice_before[{cls}] = {v:.4f}"

    def test_dice_none_without_gt(self, tmp_path):
        gt_dir    = self._write_atlas_library(tmp_path)
        pred_path = tmp_path / "pred_case_999.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)

        result = run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir, seed=0)
        assert result.dice_before is None
        assert result.dice_after  is None

    def test_disease_specific_mode(self, tmp_path):
        """disease_specific mode should run without error."""
        import json
        gt_dir = self._write_atlas_library(tmp_path, n=5)
        dm = {f"case_{i:03d}": [0]*8 for i in range(5)}
        dm_path = tmp_path / "disease_map.json"
        dm_path.write_text(json.dumps(dm))

        pred_path = tmp_path / "pred_case_999.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)

        result = run_atlas_pipeline(
            pred_path=pred_path,
            gt_folder=gt_dir,
            mode="disease_specific",
            disease_map_path=dm_path,
            seed=0,
        )
        assert result.mode == "disease_specific"
        assert result.corrected_labels.shape == SHAPE

    def test_empty_gt_folder_raises(self, tmp_path):
        """An empty GT folder should raise FileNotFoundError."""
        gt_dir = tmp_path / "empty_labels"
        gt_dir.mkdir()
        pred_path = tmp_path / "pred.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)

        with pytest.raises(FileNotFoundError):
            run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir)

    def test_deterministic_with_same_seed(self, tmp_path):
        """Two runs with the same seed should produce identical results."""
        gt_dir    = self._write_atlas_library(tmp_path)
        pred_path = tmp_path / "pred_case_999.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)

        r1 = run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir, seed=42)
        r2 = run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir, seed=42)
        assert r1.atlas_case_id == r2.atlas_case_id
        assert np.array_equal(r1.corrected_labels, r2.corrected_labels)

    def test_pca_registration_mode(self, tmp_path):
        """pca registration_mode should produce valid output."""
        gt_dir    = self._write_atlas_library(tmp_path)
        pred_path = tmp_path / "pred_case_999.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)

        result = run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir,
                                    registration_mode="pca", seed=0)
        assert result.corrected_labels.shape == SHAPE

    def test_summary_dict_flat_keys(self, tmp_path):
        """summary_dict() should return a flat dict with expected keys."""
        gt_dir    = self._write_atlas_library(tmp_path)
        pred_path = tmp_path / "pred_case_999.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)

        result = run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir, seed=0)
        d = result.summary_dict()
        for key in ["atlas_case_id", "atlas_disease_name", "mode", "was_relabeled"]:
            assert key in d, f"summary_dict missing key '{key}'"


# ---------------------------------------------------------------------------
# 6. Folder pipeline
# ---------------------------------------------------------------------------

class TestRunAtlasFolderPipeline:

    def test_returns_dataframe_with_one_row_per_case(self, tmp_path):
        from chd_postprocessing.atlas_pipeline import run_atlas_folder_pipeline
        import pandas as pd

        gt_dir   = tmp_path / "labels"
        pred_dir = tmp_path / "preds"
        out_dir  = tmp_path / "corrected"
        gt_dir.mkdir(); pred_dir.mkdir()

        vol = _make_label_volume()
        for i in range(3):
            _save_nifti(vol, gt_dir   / f"train_{i:03d}.nii.gz")
            _save_nifti(vol, pred_dir / f"pred_{i:03d}.nii.gz")

        df = run_atlas_folder_pipeline(
            pred_folder=pred_dir,
            gt_folder=gt_dir,
            output_folder=out_dir,
            mode="baseline",
            seed=0,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_output_files_created(self, tmp_path):
        from chd_postprocessing.atlas_pipeline import run_atlas_folder_pipeline

        gt_dir   = tmp_path / "labels"
        pred_dir = tmp_path / "preds"
        out_dir  = tmp_path / "corrected"
        gt_dir.mkdir(); pred_dir.mkdir()

        vol = _make_label_volume()
        for i in range(2):
            _save_nifti(vol, gt_dir   / f"train_{i:03d}.nii.gz")
            _save_nifti(vol, pred_dir / f"pred_{i:03d}.nii.gz")

        run_atlas_folder_pipeline(pred_dir, gt_dir, out_dir, seed=0)
        output_files = list(out_dir.glob("*.nii.gz"))
        assert len(output_files) == 2

    def test_empty_pred_folder_raises(self, tmp_path):
        from chd_postprocessing.atlas_pipeline import run_atlas_folder_pipeline

        gt_dir   = tmp_path / "labels"
        pred_dir = tmp_path / "empty_preds"
        out_dir  = tmp_path / "out"
        gt_dir.mkdir(); pred_dir.mkdir()
        _save_nifti(_make_label_volume(), gt_dir / "train_000.nii.gz")

        with pytest.raises(FileNotFoundError):
            run_atlas_folder_pipeline(pred_dir, gt_dir, out_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
