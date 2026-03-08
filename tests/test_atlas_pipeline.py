"""Tests for the atlas-based post-processing pipeline.

All tests are self-contained — no NIfTI files on disk are required.
Synthetic label volumes are constructed in-memory so correct outcomes
are known analytically.

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
    ComponentAssignment,
    LabelCorrectionResult,
    _find_all_components,
    _compute_component_overlaps,
    _initial_assignments,
    _resolve_multi_component_conflicts,
    compute_overlap_matrix,
    correct_labels_with_atlas,
    enforce_single_component,
    apply_label_mapping,
    apply_morphological_cleanup,
    optimal_label_mapping,
)
from chd_postprocessing.registration import _pca_axes, register_atlas_to_pred


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SHAPE     = (40, 40, 40)
SPACING   = (1.0, 1.0, 1.0)
LABEL_IDS = list(FOREGROUND_CLASSES)   # [1, 2, 3, 4, 5, 6, 7]


def _make_label_volume(shape=SHAPE) -> np.ndarray:
    """label k occupies x=[4(k-1):4k], all y, all z."""
    vol = np.zeros(shape, dtype=np.int32)
    for k in LABEL_IDS:
        x0, x1 = 4 * (k - 1), 4 * k
        vol[x0:x1, :, :] = k
    return vol


def _save_nifti(data: np.ndarray, path: Path) -> None:
    img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(img, str(path))


# ---------------------------------------------------------------------------
# 1. AtlasEntry and AtlasLibrary
# ---------------------------------------------------------------------------

class TestAtlasLibrary:

    def test_load_all_finds_files(self, tmp_path):
        vol = _make_label_volume()
        for cid in ["case_001", "case_002", "case_003"]:
            _save_nifti(vol, tmp_path / f"{cid}.nii.gz")
        lib = AtlasLibrary.load_all(tmp_path)
        assert len(lib) == 3

    def test_load_all_empty_folder(self, tmp_path):
        assert len(AtlasLibrary.load_all(tmp_path)) == 0

    def test_case_ids_parsed_correctly(self, tmp_path):
        _save_nifti(_make_label_volume(), tmp_path / "patient_042.nii.gz")
        lib = AtlasLibrary.load_all(tmp_path)
        assert lib.entries[0].case_id == "patient_042"

    def test_lazy_load(self, tmp_path):
        _save_nifti(_make_label_volume(), tmp_path / "case_001.nii.gz")
        lib = AtlasLibrary.load_all(tmp_path)
        entry = lib.entries[0]
        assert entry.labels is None
        entry.load()
        assert entry.labels is not None

    def test_select_random(self, tmp_path):
        vol = _make_label_volume()
        for i in range(5):
            _save_nifti(vol, tmp_path / f"case_{i:03d}.nii.gz")
        lib = AtlasLibrary.load_all(tmp_path)
        entry = lib.select_for_case([0] * 8, random.Random(0), mode="random")
        assert entry in lib.entries

    def test_select_best_match(self, tmp_path):
        import json
        vol = _make_label_volume()
        _save_nifti(vol, tmp_path / "normal.nii.gz")
        _save_nifti(vol, tmp_path / "hlhs.nii.gz")
        dm = {"normal": [0]*8, "hlhs": [1, 0, 0, 0, 0, 0, 0, 0]}
        (tmp_path / "dm.json").write_text(json.dumps(dm))
        lib = AtlasLibrary.load_all(tmp_path, disease_map_path=tmp_path / "dm.json")
        selected = lib.select_for_case([1,0,0,0,0,0,0,0], random.Random(0), mode="best_match")
        assert selected.case_id == "hlhs"

    def test_exclude_self(self, tmp_path):
        vol = _make_label_volume()
        for cid in ["case_001", "case_002"]:
            _save_nifti(vol, tmp_path / f"{cid}.nii.gz")
        lib = AtlasLibrary.load_all(tmp_path)
        for _ in range(20):
            e = lib.select_for_case([0]*8, random.Random(0), mode="random",
                                    exclude_case_id="case_001")
            assert e.case_id != "case_001"

    def test_hamming_distance(self, tmp_path):
        _save_nifti(_make_label_volume(), tmp_path / "case_001.nii.gz")
        lib = AtlasLibrary.load_all(tmp_path)
        entry = lib.entries[0]
        entry.disease_vec = [1, 0, 0, 0, 0, 0, 0, 0]
        assert entry.hamming_distance([1, 0, 0, 0, 0, 0, 0, 0]) == 0
        assert entry.hamming_distance([0, 0, 0, 0, 0, 0, 0, 0]) == 1


# ---------------------------------------------------------------------------
# 2. Registration — centroid and PCA sign fix
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_output_shape_matches_pred(self):
        atlas = _make_label_volume((50, 50, 50))
        pred  = _make_label_volume((40, 40, 40))
        reg   = register_atlas_to_pred(atlas, pred, SPACING)
        assert reg.shape == pred.shape

    def test_centroid_reduces_distance(self):
        pred  = _make_label_volume()
        atlas = np.roll(pred, 8, axis=0)

        def cm_dist(a, b):
            from scipy.ndimage import center_of_mass
            ca = np.array(center_of_mass((a > 0).astype(np.uint8)))
            cb = np.array(center_of_mass((b > 0).astype(np.uint8)))
            return float(np.linalg.norm(ca - cb))

        before = cm_dist(atlas, pred)
        reg    = register_atlas_to_pred(atlas, pred, SPACING, mode="centroid")
        after  = cm_dist(reg, pred)
        assert after < before

    def test_pca_sign_fix_produces_proper_rotation(self):
        """_pca_axes must return a matrix with det = +1 (no reflections)."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            coords = rng.standard_normal((500, 3))
            R = _pca_axes(coords)
            assert R.shape == (3, 3)
            assert abs(np.linalg.det(R) - 1.0) < 1e-9, \
                f"det(R) = {np.linalg.det(R):.6f}, expected +1"

    def test_pca_mode_runs(self):
        pred  = _make_label_volume()
        atlas = np.roll(pred, 4, axis=0)
        reg   = register_atlas_to_pred(atlas, pred, SPACING, mode="pca")
        assert reg.shape == pred.shape

    def test_empty_inputs_handled(self):
        pred  = _make_label_volume()
        zeros = np.zeros_like(pred)
        assert register_atlas_to_pred(zeros, pred, SPACING).shape == pred.shape
        assert register_atlas_to_pred(pred, zeros, SPACING).shape == pred.shape

    def test_invalid_mode_raises(self):
        vol = _make_label_volume()
        with pytest.raises(ValueError, match="Unknown registration mode"):
            register_atlas_to_pred(vol, vol, SPACING, mode="bad")


# ---------------------------------------------------------------------------
# 3. Legacy whole-label functions (backward compatibility)
# ---------------------------------------------------------------------------

class TestLegacyOverlapMatrix:

    def test_diagonal_one_for_identical(self):
        vol = _make_label_volume()
        M, ids = compute_overlap_matrix(vol, vol, LABEL_IDS)
        for i in range(len(ids)):
            assert abs(M[i, i] - 1.0) < 1e-6

    def test_off_diagonal_zero(self):
        vol = _make_label_volume()
        M, _ = compute_overlap_matrix(vol, vol, LABEL_IDS)
        N = len(LABEL_IDS)
        for i in range(N):
            for j in range(N):
                if i != j:
                    assert M[i, j] == 0.0

    def test_shape_n_by_n(self):
        vol = _make_label_volume()
        M, ids = compute_overlap_matrix(vol, vol, LABEL_IDS)
        assert M.shape == (len(LABEL_IDS), len(LABEL_IDS))


class TestLegacyMapping:

    def test_identity_for_identical(self):
        vol = _make_label_volume()
        M, ids = compute_overlap_matrix(vol, vol)
        assert optimal_label_mapping(M, ids) == {lbl: lbl for lbl in ids}

    def test_detects_whole_label_swap(self):
        vol_correct = _make_label_volume()
        vol_swapped = vol_correct.copy()
        vol_swapped[vol_correct == 6] = 7
        vol_swapped[vol_correct == 7] = 6
        M, ids = compute_overlap_matrix(vol_swapped, vol_correct)
        m = optimal_label_mapping(M, ids)
        assert m[6] == 7 or m[7] == 6

    def test_apply_swap(self):
        vol = _make_label_volume()
        mapping = {lbl: lbl for lbl in LABEL_IDS}
        mapping[6], mapping[7] = 7, 6
        result, changed = apply_label_mapping(vol, mapping)
        assert changed
        assert (result[vol == 6] == 7).all()
        assert (result[vol == 7] == 6).all()

    def test_apply_identity_no_change(self):
        vol = _make_label_volume()
        identity = {lbl: lbl for lbl in LABEL_IDS}
        result, changed = apply_label_mapping(vol, identity)
        assert not changed
        assert np.array_equal(result, vol)


# ---------------------------------------------------------------------------
# 4. New component-level internals
# ---------------------------------------------------------------------------

class TestFindAllComponents:

    def test_single_component_per_label(self):
        vol  = _make_label_volume()
        comps = _find_all_components(vol, LABEL_IDS)
        assert len(comps) == len(LABEL_IDS)
        for comp in comps:
            assert comp["size"] > 0
            assert comp["label_comp_idx"] == 1

    def test_multiple_components_detected(self):
        vol = np.zeros(SHAPE, dtype=np.int32)
        # Label 1 with two separated blobs
        vol[0:3, :, :] = 1
        vol[8:11, :, :] = 1
        comps = _find_all_components(vol, [1])
        assert len(comps) == 2

    def test_absent_label_produces_no_components(self):
        vol   = _make_label_volume()
        vol[vol == 7] = 0   # remove PA
        comps = _find_all_components(vol, LABEL_IDS)
        labels_found = {c["original_label"] for c in comps}
        assert 7 not in labels_found

    def test_component_sizes_sum_to_label_count(self):
        vol   = _make_label_volume()
        comps = _find_all_components(vol, LABEL_IDS)
        for lbl in LABEL_IDS:
            lbl_comps = [c for c in comps if c["original_label"] == lbl]
            total = sum(c["size"] for c in lbl_comps)
            assert total == int((vol == lbl).sum())


class TestComponentOverlaps:

    def test_perfect_overlap_diagonal(self):
        """When pred == atlas, each component should score 1.0 (IoC) against its own label."""
        vol     = _make_label_volume()
        comps   = _find_all_components(vol, LABEL_IDS)
        a_masks = {lbl: (vol == lbl) for lbl in LABEL_IDS}
        M       = _compute_component_overlaps(comps, a_masks, LABEL_IDS)
        for i, comp in enumerate(comps):
            j = LABEL_IDS.index(comp["original_label"])
            assert abs(M[i, j] - 1.0) < 1e-6

    def test_zero_overlap_for_disjoint(self):
        """Components and atlas labels occupying completely different voxels → 0."""
        vol   = _make_label_volume()
        comps = _find_all_components(vol, [1])   # only label 1
        # Atlas has label 2 only (no overlap with label 1's band)
        a_masks = {2: (vol == 2)}
        M = _compute_component_overlaps(comps, a_masks, [2])
        assert np.all(M == 0.0)

    def test_matrix_shape(self):
        vol     = _make_label_volume()
        comps   = _find_all_components(vol, LABEL_IDS)
        a_masks = {lbl: (vol == lbl) for lbl in LABEL_IDS}
        M       = _compute_component_overlaps(comps, a_masks, LABEL_IDS)
        assert M.shape == (len(comps), len(LABEL_IDS))


class TestResolveConflicts:

    def test_large_fragment_wins_small_gets_reassigned(self):
        """A tiny fragment competing for the same atlas label as a large one
        should be reassigned to its next-best match."""
        # Two components: large (100 vx) and tiny (2 vx), both overlap best
        # with label index 0.  The tiny one should be reassigned to index 1.
        n = 2
        n_labels = 3
        M = np.array([
            [0.8, 0.1, 0.0],   # large component — best is col 0
            [0.5, 0.4, 0.0],   # tiny component  — best is col 0, second-best col 1
        ])
        components = [
            {"size": 100, "original_label": 1},
            {"size": 2,   "original_label": 1},
        ]
        assignments = [0, 0]   # both want label index 0
        label_ids   = [1, 2, 3]

        resolved = _resolve_multi_component_conflicts(
            assignments, components, M, label_ids,
            min_component_fraction=0.1,  # 2 < 0.1*100 → tiny fragment qualifies
        )
        assert resolved[0] == 0   # large keeps label 0
        assert resolved[1] == 1   # tiny reassigned to label 1

    def test_no_conflict_unchanged(self):
        M = np.array([[0.9, 0.0], [0.0, 0.8]])
        components = [{"size": 50, "original_label": 1},
                      {"size": 50, "original_label": 2}]
        assignments = [0, 1]
        resolved = _resolve_multi_component_conflicts(
            assignments, components, M, [1, 2], min_component_fraction=0.01
        )
        assert resolved == [0, 1]


# ---------------------------------------------------------------------------
# 5. correct_labels_with_atlas (high-level)
# ---------------------------------------------------------------------------

class TestCorrectLabelsWithAtlas:

    def test_identity_when_pred_equals_atlas(self):
        vol    = _make_label_volume()
        result = correct_labels_with_atlas(vol, vol, label_ids=LABEL_IDS)
        assert isinstance(result, LabelCorrectionResult)
        assert not result.was_relabeled
        for lbl in LABEL_IDS:
            assert result.mapping_applied[lbl] == lbl

    def test_whole_label_swap_dominant_preserved(self):
        """Dominant components are locked — global AO/PA swap is NOT corrected
        by correct_labels_with_atlas alone.  Global swaps are handled upstream
        by correct_ao_pa_fragments (anatomy_priors module)."""
        vol_correct = _make_label_volume()
        vol_swapped = vol_correct.copy()
        vol_swapped[vol_correct == 6] = 7
        vol_swapped[vol_correct == 7] = 6
        result = correct_labels_with_atlas(vol_swapped, vol_correct,
                                           label_ids=LABEL_IDS,
                                           do_morphological_cleanup=False)
        # Dominant components locked → no relabeling from this function alone
        assert not result.was_relabeled
        # The full pipeline (run_atlas_pipeline) applies anatomy correction first

    def test_misclassified_fragment_reassigned(self):
        """A small fragment of label 5 (Myo) placed in the atlas's label-6 (AO)
        region should be reassigned to label 6."""
        vol_correct = _make_label_volume()
        vol_pred    = vol_correct.copy()

        # Inject a tiny Myo fragment (label 5) inside the AO band (x=[20:24]),
        # but NOT adjacent to the main Myo band (x=[16:20]) — gap of 2 voxels.
        vol_pred[22:24, 0:3, 0:3] = 5   # small fragment in AO region, 2-voxel gap from Myo

        result = correct_labels_with_atlas(vol_pred, vol_correct,
                                           label_ids=LABEL_IDS,
                                           do_morphological_cleanup=False)

        # That fragment should have been reassigned from 5 → 6
        assert result.was_relabeled
        reassigned = [ca for ca in result.component_assignments if ca.was_reassigned]
        assert any(ca.original_label == 5 and ca.assigned_label == 6
                   for ca in reassigned), \
            f"Expected a 5→6 reassignment; got: {reassigned}"

    def test_component_assignments_populated(self):
        vol    = _make_label_volume()
        result = correct_labels_with_atlas(vol, vol, label_ids=LABEL_IDS)
        assert isinstance(result.component_assignments, list)
        assert len(result.component_assignments) == len(LABEL_IDS)
        for ca in result.component_assignments:
            assert isinstance(ca, ComponentAssignment)
            assert ca.size > 0
            assert 0.0 <= ca.best_overlap <= 1.0

    def test_overlap_matrix_shape(self):
        """overlap_matrix should now be (n_components, n_labels) not (N, N)."""
        vol    = _make_label_volume()
        result = correct_labels_with_atlas(vol, vol, label_ids=LABEL_IDS)
        n_comps = len(result.component_assignments)
        assert result.overlap_matrix.shape == (n_comps, len(LABEL_IDS))

    def test_no_foreground_returns_unchanged(self):
        vol    = np.zeros(SHAPE, dtype=np.int32)
        result = correct_labels_with_atlas(vol, vol, label_ids=LABEL_IDS)
        assert np.array_equal(result.corrected_labels, vol)
        assert not result.was_relabeled

    def test_returns_correct_type(self):
        vol    = _make_label_volume()
        result = correct_labels_with_atlas(vol, vol)
        for attr in ("corrected_labels", "overlap_matrix", "mapping_applied",
                     "was_relabeled", "component_assignments", "reassignment_summary"):
            assert hasattr(result, attr)


# ---------------------------------------------------------------------------
# 6. Structural cleanup helpers
# ---------------------------------------------------------------------------

class TestEnforceSingleComponent:

    def test_single_component_unchanged(self):
        vol    = _make_label_volume()
        before = {lbl: int((vol == lbl).sum()) for lbl in LABEL_IDS}
        clean, info = enforce_single_component(vol, LABEL_IDS)
        for lbl in LABEL_IDS:
            assert int((clean == lbl).sum()) == before[lbl]

    def test_tiny_fragment_removed(self):
        vol = _make_label_volume()
        vol[35, 35, 35] = 1   # 1-voxel isolated fragment of label 1
        clean, info = enforce_single_component(vol, [1], min_component_fraction=0.01)
        assert info[1]["removed"] >= 1
        assert clean[35, 35, 35] == 0


# ---------------------------------------------------------------------------
# 7. Full pipeline (run_atlas_pipeline)
# ---------------------------------------------------------------------------

class TestRunAtlasPipeline:

    def _write_lib(self, tmp_path: Path, n: int = 5) -> Path:
        gt_dir = tmp_path / "labelsTr"
        gt_dir.mkdir()
        vol = _make_label_volume()
        for i in range(n):
            _save_nifti(vol, gt_dir / f"case_{i:03d}.nii.gz")
        return gt_dir

    def test_output_shape_matches_input(self, tmp_path):
        gt_dir    = self._write_lib(tmp_path)
        pred_path = tmp_path / "pred.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)
        result = run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir, seed=0)
        assert result.corrected_labels.shape == SHAPE

    def test_perturbation_mode_wiring(self, tmp_path):
        """Baseline applies perturbation; disease_specific does not (by default).

        Verified via source inspection — both flags must exist and
        create_synthetic_atlas must be called inside run_atlas_pipeline.
        """
        import chd_postprocessing.atlas_pipeline as m
        import inspect
        src = inspect.getsource(m.run_atlas_pipeline)
        assert "do_perturbation" in src, "do_perturbation parameter must exist"
        assert "do_anatomy_correction" in src, "do_anatomy_correction parameter must exist"
        assert "create_synthetic_atlas" in src, \
            "create_synthetic_atlas must be called in run_atlas_pipeline (for baseline)"
        # Verify the mode wiring: random_atlas → do_perturbation=True, anatomy=False
        assert 'mode == "random_atlas"' in src or "mode == 'random_atlas'" in src, \
            "mode-based default wiring must reference 'random_atlas'"

    def test_saved_to_disk(self, tmp_path):
        gt_dir    = self._write_lib(tmp_path)
        pred_path = tmp_path / "pred.nii.gz"
        out_path  = tmp_path / "corrected.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)
        run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir,
                           output_path=out_path, seed=0)
        assert out_path.exists()

    def test_dice_computed_when_gt_given(self, tmp_path):
        gt_dir    = self._write_lib(tmp_path)
        pred_path = tmp_path / "pred.nii.gz"
        gt_path   = tmp_path / "gt.nii.gz"
        vol = _make_label_volume()
        _save_nifti(vol, pred_path)
        _save_nifti(vol, gt_path)
        result = run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir,
                                    gt_path=gt_path, seed=0)
        assert result.dice_before is not None
        assert result.dice_after  is not None

    def test_disease_specific_mode(self, tmp_path):
        import json
        gt_dir = self._write_lib(tmp_path, n=5)
        dm = {f"case_{i:03d}": [0]*8 for i in range(5)}
        dm_path = tmp_path / "dm.json"
        dm_path.write_text(json.dumps(dm))
        pred_path = tmp_path / "pred.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)
        result = run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir,
                                    mode="disease_specific",
                                    disease_map_path=dm_path, seed=0)
        # "disease_specific" is a backward-compat alias for "disease_atlas_rules"
        assert result.mode == "disease_atlas_rules"
        assert result.corrected_labels.shape == SHAPE

    def test_empty_gt_folder_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        pred_path = tmp_path / "pred.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)
        with pytest.raises(FileNotFoundError):
            run_atlas_pipeline(pred_path=pred_path, gt_folder=empty)

    def test_deterministic_with_same_seed(self, tmp_path):
        gt_dir    = self._write_lib(tmp_path)
        pred_path = tmp_path / "pred.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)
        r1 = run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir, seed=7)
        r2 = run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir, seed=7)
        assert r1.atlas_case_id == r2.atlas_case_id
        assert np.array_equal(r1.corrected_labels, r2.corrected_labels)

    def test_component_assignments_in_result(self, tmp_path):
        gt_dir    = self._write_lib(tmp_path)
        pred_path = tmp_path / "pred.nii.gz"
        _save_nifti(_make_label_volume(), pred_path)
        result = run_atlas_pipeline(pred_path=pred_path, gt_folder=gt_dir, seed=0)
        # AtlasPipelineResult wraps LabelCorrectionResult; verify overlap_matrix
        # is (n_components, n_labels) not (n_labels, n_labels)
        n_labels = len(LABEL_IDS)
        assert result.overlap_matrix.shape[1] == n_labels
        assert result.overlap_matrix.shape[0] >= n_labels   # at least 1 comp per label


# ---------------------------------------------------------------------------
# 8. Folder pipeline
# ---------------------------------------------------------------------------

class TestRunAtlasFolderPipeline:

    def test_returns_dataframe(self, tmp_path):
        from chd_postprocessing.atlas_pipeline import run_atlas_folder_pipeline
        import pandas as pd

        gt   = tmp_path / "labels"; gt.mkdir()
        pred = tmp_path / "preds";  pred.mkdir()
        out  = tmp_path / "out"
        vol  = _make_label_volume()
        for i in range(3):
            _save_nifti(vol, gt   / f"train_{i:03d}.nii.gz")
            _save_nifti(vol, pred / f"pred_{i:03d}.nii.gz")

        df = run_atlas_folder_pipeline(pred, gt, out, mode="baseline", seed=0)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_output_files_written(self, tmp_path):
        from chd_postprocessing.atlas_pipeline import run_atlas_folder_pipeline

        gt   = tmp_path / "labels"; gt.mkdir()
        pred = tmp_path / "preds";  pred.mkdir()
        out  = tmp_path / "out"
        vol  = _make_label_volume()
        for i in range(2):
            _save_nifti(vol, gt   / f"train_{i:03d}.nii.gz")
            _save_nifti(vol, pred / f"pred_{i:03d}.nii.gz")

        run_atlas_folder_pipeline(pred, gt, out, seed=0)
        assert len(list(out.glob("*.nii.gz"))) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
