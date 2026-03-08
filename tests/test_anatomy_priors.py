"""Tests for anatomy_priors.correct_ao_pa_labels.

Synthetic 3D volumes are constructed where the spatial layout of LV, RV,
AO, and PA is known exactly, making test outcomes deterministic.

Volume layout (60 × 60 × 60 voxels, isotropic 1 mm spacing)
--------------------------------------------------------------

              x=0          x=30        x=60
              +------------+------------+
              |   LV (1)   |   RV (2)   |   z-slice (all z)
              |  x=5:25    |  x=35:55   |
              +------------+------------+
              |  AO / PA   |  PA / AO   |
              |  y=25:45   |  y=25:45   |
              +------------+------------+

In the **correct** layout:
  - AO (6) is at x=5:25, y=25:45  → adjacent to LV (x=5:25, y=5:25)
  - PA (7) is at x=35:55, y=25:45 → adjacent to RV (x=35:55, y=5:25)

In the **swapped** layout, AO and PA positions are exchanged.
"""

import numpy as np
import pytest

from chd_postprocessing.anatomy_priors import correct_ao_pa_labels, correct_ao_pa_fragments
from chd_postprocessing.config import LABELS, PUA_FLAG_INDEX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_volume(swap_ao_pa: bool = False) -> np.ndarray:
    """Build a 60×60×60 synthetic segmentation volume."""
    vol = np.zeros((60, 60, 60), dtype=np.int32)

    # Ventricles (full z-extent for simplicity)
    vol[5:25,  5:25, 5:55] = LABELS["LV"]   # left side
    vol[35:55, 5:25, 5:55] = LABELS["RV"]   # right side

    if swap_ao_pa:
        # AO placed near RV (wrong), PA placed near LV (wrong)
        vol[35:55, 25:45, 5:55] = LABELS["AO"]
        vol[5:25,  25:45, 5:55] = LABELS["PA"]
    else:
        # Correct: AO near LV, PA near RV
        vol[5:25,  25:45, 5:55] = LABELS["AO"]
        vol[35:55, 25:45, 5:55] = LABELS["PA"]

    return vol


SPACING_MM = (1.0, 1.0, 1.0)
DISEASE_VEC_NORMAL = [0] * 8   # no disease flags set
DISEASE_VEC_PUA    = [0] * 8
DISEASE_VEC_PUA[PUA_FLAG_INDEX] = 1


# ---------------------------------------------------------------------------
# Test 1: correct labeling is left untouched
# ---------------------------------------------------------------------------

def test_correct_labeling_not_swapped():
    """When AO/PA labels are anatomically correct, no swap should occur."""
    vol = _make_volume(swap_ao_pa=False)
    result = correct_ao_pa_labels(vol, DISEASE_VEC_NORMAL, SPACING_MM)

    assert result.was_swapped is False
    # Labels should be unchanged
    assert np.array_equal(result.corrected_labels, vol)


# ---------------------------------------------------------------------------
# Test 2: swapped labeling is corrected
# ---------------------------------------------------------------------------

def test_swapped_labeling_is_corrected():
    """When AO and PA labels are swapped, the correction should swap them back."""
    vol_swapped = _make_volume(swap_ao_pa=True)
    vol_correct = _make_volume(swap_ao_pa=False)

    result = correct_ao_pa_labels(vol_swapped, DISEASE_VEC_NORMAL, SPACING_MM)

    assert result.was_swapped is True
    assert result.skipped_reason is None
    assert np.array_equal(result.corrected_labels, vol_correct), (
        "After correction the label volume should match the anatomically correct layout"
    )


# ---------------------------------------------------------------------------
# Test 3: PuA cases are left untouched regardless of label order
# ---------------------------------------------------------------------------

def test_pua_case_not_corrected():
    """Cases with PuA=1 must be left exactly as-is (AO/PA fusion is expected)."""
    vol_swapped = _make_volume(swap_ao_pa=True)
    original    = vol_swapped.copy()

    result = correct_ao_pa_labels(vol_swapped, DISEASE_VEC_PUA, SPACING_MM)

    assert result.was_swapped is False
    assert result.skipped_reason is not None
    assert "PuA" in result.skipped_reason
    assert np.array_equal(result.corrected_labels, original)


def test_pua_case_correct_layout_also_untouched():
    """PuA cases with correct layout should also be returned unchanged."""
    vol_correct = _make_volume(swap_ao_pa=False)
    original    = vol_correct.copy()

    result = correct_ao_pa_labels(vol_correct, DISEASE_VEC_PUA, SPACING_MM)
    assert np.array_equal(result.corrected_labels, original)


# ---------------------------------------------------------------------------
# Test 4: disease_vec=None is treated as all-zeros (correction runs normally)
# ---------------------------------------------------------------------------

def test_none_disease_vec_applies_correction():
    """None disease_vec should trigger the same correction path as all-zeros."""
    vol_swapped  = _make_volume(swap_ao_pa=True)
    result_none  = correct_ao_pa_labels(vol_swapped, None, SPACING_MM)
    result_zeros = correct_ao_pa_labels(vol_swapped, DISEASE_VEC_NORMAL, SPACING_MM)

    assert result_none.was_swapped == result_zeros.was_swapped
    assert np.array_equal(result_none.corrected_labels, result_zeros.corrected_labels)


# ---------------------------------------------------------------------------
# Test 5: edge cases — missing structures
# ---------------------------------------------------------------------------

def test_missing_ao_skipped():
    vol = _make_volume(swap_ao_pa=False)
    vol[vol == LABELS["AO"]] = 0   # remove AO entirely

    result = correct_ao_pa_labels(vol, DISEASE_VEC_NORMAL, SPACING_MM)
    assert result.was_swapped is False
    assert result.needs_manual_review is True
    assert result.skipped_reason is not None


def test_missing_lv_skipped():
    vol = _make_volume(swap_ao_pa=False)
    vol[vol == LABELS["LV"]] = 0   # remove LV

    result = correct_ao_pa_labels(vol, DISEASE_VEC_NORMAL, SPACING_MM)
    assert result.was_swapped is False
    assert result.needs_manual_review is True


# ---------------------------------------------------------------------------
# Test 6: confidence score is high for unambiguous cases
# ---------------------------------------------------------------------------

def test_high_confidence_for_unambiguous_cases():
    """Clear adjacency → confidence should be well above the low threshold."""
    vol_correct = _make_volume(swap_ao_pa=False)
    vol_swapped = _make_volume(swap_ao_pa=True)

    res_correct = correct_ao_pa_labels(vol_correct, DISEASE_VEC_NORMAL, SPACING_MM)
    res_swapped = correct_ao_pa_labels(vol_swapped, DISEASE_VEC_NORMAL, SPACING_MM)

    # Both should have high confidence because the geometry is unambiguous
    assert res_correct.confidence_score > 0.5, (
        f"Expected high confidence for correct layout, got {res_correct.confidence_score}"
    )
    assert res_swapped.confidence_score > 0.5, (
        f"Expected high confidence for swapped layout, got {res_swapped.confidence_score}"
    )


# ---------------------------------------------------------------------------
# Test 7: adjacency_details are populated
# ---------------------------------------------------------------------------

def test_adjacency_details_populated():
    vol = _make_volume(swap_ao_pa=False)
    result = correct_ao_pa_labels(vol, DISEASE_VEC_NORMAL, SPACING_MM)

    details = result.adjacency_details
    for key in ("ao_lv_count", "ao_rv_count", "pa_lv_count", "pa_rv_count"):
        assert key in details, f"Missing key '{key}' in adjacency_details"

    # AO should overlap LV more than RV (correct layout)
    assert details["ao_lv_count"] > details["ao_rv_count"]
    assert details["pa_rv_count"] > details["pa_lv_count"]


# ---------------------------------------------------------------------------
# Test 8: anisotropic spacing does not crash
# ---------------------------------------------------------------------------

def test_anisotropic_spacing():
    """Anisotropic voxel spacing should be handled without error."""
    vol = _make_volume(swap_ao_pa=True)
    spacing = (1.5, 1.0, 0.5)   # anisotropic

    result = correct_ao_pa_labels(vol, DISEASE_VEC_NORMAL, spacing, dilation_radius_mm=3.0)
    # Should not raise; result should be a CorrectionResult
    assert hasattr(result, "corrected_labels")
    assert result.corrected_labels.shape == vol.shape


# ---------------------------------------------------------------------------
# Disease-aware tests for correct_ao_pa_fragments
# ---------------------------------------------------------------------------

def _make_fragment_volume() -> np.ndarray:
    """60×60×60 volume with clear vessel-ventricle geometry for fragment tests."""
    vol = np.zeros((60, 60, 60), dtype=np.int32)
    vol[5:25,  5:25, 5:55] = LABELS["LV"]   # left
    vol[35:55, 5:25, 5:55] = LABELS["RV"]   # right
    # Correct AO near LV, PA near RV
    vol[5:25,  25:45, 5:55] = LABELS["AO"]
    vol[35:55, 25:45, 5:55] = LABELS["PA"]
    return vol


DISEASE_VEC_TGA  = [0] * 8; DISEASE_VEC_TGA[7]  = 1
DISEASE_VEC_DORV = [0] * 8; DISEASE_VEC_DORV[4] = 1
DISEASE_VEC_TOF  = [0] * 8; DISEASE_VEC_TOF[6]  = 1
DISEASE_VEC_HLHS = [0] * 8; DISEASE_VEC_HLHS[0] = 1
DISEASE_VEC_PUA_TOF = [0] * 8
DISEASE_VEC_PUA_TOF[PUA_FLAG_INDEX] = 1
DISEASE_VEC_PUA_TOF[6] = 1


def test_tga_ao_near_rv_not_swapped():
    """TGA: AO should be near RV; a fragment near RV must NOT be relabelled."""
    vol = _make_fragment_volume()
    # Add an AO fragment near RV (correct for TGA)
    vol[37:43, 27:33, 5:55] = LABELS["AO"]

    corrected, log = correct_ao_pa_fragments(vol, DISEASE_VEC_TGA, SPACING_MM)

    # The fragment adjacent to RV should remain AO (correct under TGA)
    frag_region = corrected[37:43, 27:33, 5:55]
    assert np.all(frag_region == LABELS["AO"]), (
        "Under TGA, AO near RV should not be relabelled (it is anatomically correct)"
    )


def test_tga_pa_near_rv_relabelled():
    """TGA: PA should be near LV; a PA fragment near RV is wrong and should be relabelled."""
    vol = np.zeros((60, 60, 60), dtype=np.int32)
    vol[5:25,  5:25, 5:55] = LABELS["LV"]
    vol[35:55, 5:25, 5:55] = LABELS["RV"]
    # TGA: AO exits RV, PA exits LV — so main bodies placed accordingly
    vol[35:55, 25:45, 5:55] = LABELS["AO"]   # correct under TGA
    vol[5:25,  25:45, 5:55] = LABELS["PA"]   # correct under TGA
    # Add a PA fragment near RV (wrong for TGA — PA should be near LV)
    vol[37:43, 27:33, 5:55] = LABELS["PA"]

    corrected, log = correct_ao_pa_fragments(vol, DISEASE_VEC_TGA, SPACING_MM)

    # Fragment near RV should have been relabelled away from PA
    frag_region = corrected[37:43, 27:33, 5:55]
    assert not np.all(frag_region == LABELS["PA"]), (
        "Under TGA, PA near RV should be relabelled (PA should be near LV in TGA)"
    )


def test_dorv_ao_near_rv_not_relabelled():
    """DORV: both vessels exit RV; AO near RV must not be relabelled."""
    vol = _make_fragment_volume()
    # Inject AO fragment near RV (correct for DORV)
    vol[37:43, 27:33, 5:55] = LABELS["AO"]

    corrected, log = correct_ao_pa_fragments(vol, DISEASE_VEC_DORV, SPACING_MM)

    frag_region = corrected[37:43, 27:33, 5:55]
    assert np.all(frag_region == LABELS["AO"]), (
        "Under DORV, AO near RV should not be relabelled (both vessels exit RV)"
    )


def test_tof_ao_unconstrained_no_relabelling():
    """ToF: AO is unconstrained (overriding aorta); no AO fragment should be relabelled."""
    vol = _make_fragment_volume()
    # Inject AO fragment near RV (would be wrong for normal but is fine for ToF)
    vol[37:43, 27:33, 5:55] = LABELS["AO"]

    corrected, log = correct_ao_pa_fragments(vol, DISEASE_VEC_TOF, SPACING_MM)

    frag_region = corrected[37:43, 27:33, 5:55]
    # AO is unconstrained in ToF, so no AO relabelling should happen at all
    assert log["reassigned"] == [] or all(
        entry["original_label"] != LABELS["AO"] for entry in log["reassigned"]
    ), "Under ToF, AO fragments must not be relabelled (unconstrained)"


def test_hlhs_correction_skipped():
    """HLHS: skip_ao_pa_correction=True; volume returned unchanged."""
    vol = _make_fragment_volume()
    original = vol.copy()

    corrected, log = correct_ao_pa_fragments(vol, DISEASE_VEC_HLHS, SPACING_MM)

    assert np.array_equal(corrected, original), (
        "HLHS=1: correction must be skipped; volume must be returned unchanged"
    )
    assert log["skipped_disease"] is True, "skipped_disease flag must be True for HLHS"


def test_pua_plus_tof_skip_wins():
    """PuA+ToF: PuA has skip_ao_pa_correction=True; correction must be skipped."""
    vol = _make_fragment_volume()
    original = vol.copy()

    corrected, log = correct_ao_pa_fragments(vol, DISEASE_VEC_PUA_TOF, SPACING_MM)

    assert np.array_equal(corrected, original), (
        "PuA+ToF: PuA skip_ao_pa_correction wins; volume must be unchanged"
    )
    assert log["skipped_disease"] is True, "skipped_disease flag must be True for PuA+ToF"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
