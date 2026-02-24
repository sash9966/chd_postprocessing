"""Tests for connected_components module."""

import numpy as np
import pytest

from chd_postprocessing.config import LABELS
from chd_postprocessing.connected_components import cleanup_vessel_fragments, component_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vol_with_fragments(
    main_size: int = 20,
    orphan_size: int = 2,
    gap: int = 5,
) -> np.ndarray:
    """Create a volume with one large AO component and one tiny orphan.

    Layout (in 1-D schematic, replicated over 3 axes):
      [0 … main_size] = AO (large)
      [main_size + gap … main_size + gap + orphan_size] = AO (tiny orphan)
    """
    vol = np.zeros((60, 60, 60), dtype=np.int32)
    # Large main component
    vol[2:2 + main_size, 2:2 + main_size, 2:2 + main_size] = LABELS["AO"]
    # Small orphan — disconnected (gap of `gap` voxels)
    start = 2 + main_size + gap
    vol[start:start + orphan_size, 2:2 + orphan_size, 2:2 + orphan_size] = LABELS["AO"]
    return vol


# ---------------------------------------------------------------------------
# Test 1: single-component label is left unchanged
# ---------------------------------------------------------------------------

def test_single_component_unchanged():
    vol = np.zeros((30, 30, 30), dtype=np.int32)
    vol[5:25, 5:25, 5:25] = LABELS["AO"]
    original = vol.copy()

    cleaned, info = cleanup_vessel_fragments(vol)
    assert np.array_equal(cleaned, original)
    assert info[LABELS["AO"]]["removed"] == 0


# ---------------------------------------------------------------------------
# Test 2: tiny orphan is removed
# ---------------------------------------------------------------------------

def test_orphan_fragment_removed():
    vol = _vol_with_fragments(main_size=20, orphan_size=2)
    cleaned, info = cleanup_vessel_fragments(vol)

    ao_info = info[LABELS["AO"]]
    assert ao_info["n_components"] == 2
    assert ao_info["removed"] == 1
    assert ao_info["kept"] == 1

    # Orphan voxels should all be background after cleanup
    # The large component should remain as AO
    assert (cleaned == LABELS["AO"]).any(), "Main AO component should survive"


# ---------------------------------------------------------------------------
# Test 3: orphan is set to background (0)
# ---------------------------------------------------------------------------

def test_orphan_set_to_background():
    vol = _vol_with_fragments(main_size=20, orphan_size=2, gap=5)
    cleaned, _ = cleanup_vessel_fragments(vol)

    # Total AO voxels in cleaned should be less than in original
    assert (cleaned == LABELS["AO"]).sum() < (vol == LABELS["AO"]).sum()


# ---------------------------------------------------------------------------
# Test 4: large secondary component is preserved when above threshold
# ---------------------------------------------------------------------------

def test_large_secondary_component_kept():
    vol = np.zeros((60, 60, 60), dtype=np.int32)
    vol[2:20,  2:20,  2:20]  = LABELS["PA"]   # first  component: 18^3 voxels
    vol[30:48, 30:48, 30:48] = LABELS["PA"]   # second component: 18^3 voxels (same size)

    # min_component_fraction=0.01 → both 18^3 components are well above 1 %
    cleaned, info = cleanup_vessel_fragments(vol, min_component_fraction=0.01)
    assert info[LABELS["PA"]]["removed"] == 0
    assert info[LABELS["PA"]]["kept"] == 2


# ---------------------------------------------------------------------------
# Test 5: only specified vessel labels are cleaned
# ---------------------------------------------------------------------------

def test_only_specified_labels_cleaned():
    vol = np.zeros((40, 40, 40), dtype=np.int32)
    # One large + one tiny AO
    vol[2:20, 2:20, 2:20] = LABELS["AO"]
    vol[25:27, 2:4, 2:4]  = LABELS["AO"]
    # One large + one tiny LV — should NOT be cleaned
    vol[2:20, 22:40, 2:20] = LABELS["LV"]
    vol[25:27, 22:24, 2:4] = LABELS["LV"]

    original_lv = (vol == LABELS["LV"]).copy()
    cleaned, _ = cleanup_vessel_fragments(vol, vessel_label_ids=[LABELS["AO"]])

    # AO orphan removed
    assert (cleaned == LABELS["AO"]).sum() < (vol == LABELS["AO"]).sum()
    # LV is untouched
    assert np.array_equal(cleaned == LABELS["LV"], original_lv)


# ---------------------------------------------------------------------------
# Test 6: label absent — no crash
# ---------------------------------------------------------------------------

def test_label_absent_no_crash():
    vol = np.zeros((30, 30, 30), dtype=np.int32)
    vol[5:25, 5:25, 5:25] = LABELS["LV"]   # No AO or PA

    cleaned, info = cleanup_vessel_fragments(vol)
    assert np.array_equal(cleaned, vol)
    assert info[LABELS["AO"]]["n_components"] == 0
    assert info[LABELS["PA"]]["n_components"] == 0


# ---------------------------------------------------------------------------
# Test 7: component_summary diagnostic
# ---------------------------------------------------------------------------

def test_component_summary():
    vol = _vol_with_fragments(main_size=20, orphan_size=2)
    summary = component_summary(vol, LABELS["AO"])

    assert summary["n_components"] == 2
    assert len(summary["sizes"]) == 2
    # Largest fraction should be close to 1 (the big component dominates)
    assert summary["largest_fraction"] > 0.9


def test_component_summary_absent():
    vol = np.zeros((20, 20, 20), dtype=np.int32)
    summary = component_summary(vol, LABELS["AO"])
    assert summary["n_components"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
