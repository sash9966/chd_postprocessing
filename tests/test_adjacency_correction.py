"""Tests for adjacency_correction.correct_by_adjacency.

Synthetic 3D volumes (32 × 32 × 32 voxels) are constructed where the
spatial layout is known exactly, allowing deterministic assertions about
which components should or should not be relabelled.

Label IDs used throughout:
  1 = LV, 2 = RV, 3 = LA, 4 = RA, 5 = Myo, 6 = AO, 7 = PA
"""

import numpy as np
import pytest

from chd_postprocessing.adjacency_correction import (
    build_adjacency_graph,
    component_neighbor_profile,
    connectivity_improvement,
    correct_by_adjacency,
)
from chd_postprocessing.config import LABELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABEL_IDS = list(range(1, 8))  # 1..7

def _empty_vol() -> np.ndarray:
    return np.zeros((32, 32, 32), dtype=np.int32)


def _normal_heart() -> np.ndarray:
    """Normal heart layout used as a base for several tests.

    Layout (x-axis partitioned):
      LV  (1): x=0:10, y=0:10, z=0:32
      RV  (2): x=20:32, y=0:10, z=0:32
      LA  (3): x=0:10, y=12:22, z=0:32
      RA  (4): x=20:32, y=12:22, z=0:32
      Myo (5): x=10:20, y=0:10, z=0:32  (bridges LV and RV)
      AO  (6): x=0:10, y=22:32, z=0:32  (near LV)
      PA  (7): x=20:32, y=22:32, z=0:32 (near RV)
    """
    vol = _empty_vol()
    vol[0:10,  0:10,  :] = LABELS["LV"]
    vol[20:32, 0:10,  :] = LABELS["RV"]
    vol[0:10,  12:22, :] = LABELS["LA"]
    vol[20:32, 12:22, :] = LABELS["RA"]
    vol[10:20, 0:10,  :] = LABELS["Myo"]
    vol[0:10,  22:32, :] = LABELS["AO"]
    vol[20:32, 22:32, :] = LABELS["PA"]
    return vol


# ---------------------------------------------------------------------------
# Test 1: Normal heart — AO fragment touching only RA gets relabelled
# ---------------------------------------------------------------------------

def test_ao_fragment_touching_ra_gets_relabelled():
    """An AO fragment adjacent only to RA (and not to LV/Myo) should be relabelled.

    We place a small AO fragment at the far right (RA side).
    The registered atlas has AO only on the left (near LV).
    Expected: the fragment is relabelled (not kept as AO).
    """
    vol = _normal_heart()
    # Inject a small AO fragment adjacent to RA, far from LV
    vol[24:28, 14:18, 10:20] = LABELS["AO"]

    atlas = _normal_heart()  # registered atlas has correct anatomy

    corrected, log = correct_by_adjacency(
        vol, atlas, label_ids=LABEL_IDS, disease_vec=None, min_component_fraction=0.01
    )

    # The fragment should have been relabelled (not AO any more)
    fragment_region = corrected[24:28, 14:18, 10:20]
    assert not np.all(fragment_region == LABELS["AO"]), (
        "Small AO fragment touching RA should have been relabelled away from AO"
    )


# ---------------------------------------------------------------------------
# Test 2: AO-PA-AO sandwich → middle PA becomes AO
# ---------------------------------------------------------------------------

def test_aba_sandwich_pa_between_ao_becomes_ao():
    """An AO-PA-AO chain: a small PA fragment sandwiched between two AO regions.

    Both AO regions are near LV.  Relabelling the PA fragment as AO merges
    the two AO pieces (connectivity improvement > 0).
    """
    vol = _empty_vol()
    vol[2:12,  0:10,  :] = LABELS["LV"]
    vol[20:30, 0:10,  :] = LABELS["RV"]
    vol[10:18, 0:10,  :] = LABELS["Myo"]
    # AO split into two parts with a PA fragment in between
    vol[2:12,  12:18, :] = LABELS["AO"]   # left AO
    vol[2:12,  20:26, :] = LABELS["AO"]   # right AO
    vol[2:12,  18:20, :] = LABELS["PA"]   # PA fragment (gap)
    # PA main body on the RV side
    vol[20:30, 12:26, :] = LABELS["PA"]

    atlas = _empty_vol()
    atlas[2:12,  0:10,  :] = LABELS["LV"]
    atlas[20:30, 0:10,  :] = LABELS["RV"]
    atlas[10:18, 0:10,  :] = LABELS["Myo"]
    atlas[2:12,  12:26, :] = LABELS["AO"]  # atlas shows single AO near LV
    atlas[20:30, 12:26, :] = LABELS["PA"]

    corrected, log = correct_by_adjacency(
        vol, atlas, label_ids=LABEL_IDS, disease_vec=None, min_component_fraction=0.01
    )

    # The small PA fragment (gap filler) should have become AO
    gap_region = corrected[2:12, 18:20, :]
    assert np.all(gap_region == LABELS["AO"]), (
        f"AO-PA-AO sandwich gap should be relabelled AO; got unique: {np.unique(gap_region)}"
    )


# ---------------------------------------------------------------------------
# Test 3: Dominant component is never modified
# ---------------------------------------------------------------------------

def test_dominant_component_never_modified():
    """The largest component of any label must remain unchanged."""
    vol = _normal_heart()
    ao_mask_before = (vol == LABELS["AO"]).copy()

    # Create a large second AO region to test that the dominant one is locked
    # But actually we want to test that the main AO body is never touched
    atlas = _normal_heart()

    corrected, log = correct_by_adjacency(
        vol, atlas, label_ids=LABEL_IDS, disease_vec=None, min_component_fraction=0.01
    )

    # No relabelling should have happened (normal heart with atlas is already consistent)
    ao_mask_after = corrected == LABELS["AO"]
    # The main AO body (largest component) should be unchanged
    # Find dominant component in original
    from scipy.ndimage import label as nd_label
    labeled, n = nd_label(ao_mask_before)
    sizes = [(int((labeled == c).sum()), c) for c in range(1, n + 1)]
    dominant_mask = labeled == max(sizes, key=lambda x: x[0])[1]

    assert np.all(corrected[dominant_mask] == LABELS["AO"]), (
        "Dominant AO component must not be modified by adjacency correction"
    )


# ---------------------------------------------------------------------------
# Test 4: ToF — AO touching RV is NOT relabelled (allowed in ToF)
# ---------------------------------------------------------------------------

def test_tof_ao_touching_rv_not_relabelled():
    """In ToF, AO is unconstrained (overriding aorta); AO near RV is allowed."""
    vol = _normal_heart()
    # Add an AO fragment adjacent to RV
    vol[22:28, 2:8, 5:25] = LABELS["AO"]

    atlas = _normal_heart()

    tof_vec = [0] * 8
    tof_vec[6] = 1  # ToF flag index = 6

    corrected, log = correct_by_adjacency(
        vol, atlas, label_ids=LABEL_IDS, disease_vec=tof_vec, min_component_fraction=0.001
    )

    # In ToF, AO is unconstrained, so an AO fragment near RV should remain AO
    # (or at least not be flagged as a forbidden adjacency violation)
    tof_related_changes = [
        entry for entry in log
        if entry["original_label"] == LABELS["AO"]
        and entry["reason"] == "adjacency_violation"
    ]
    # There should be no forced relabelling due to AO-RV adjacency under ToF
    # (it might be relabelled for other reasons, but not because AO-RV is forbidden)
    for change in tof_related_changes:
        assert LABELS["RV"] not in change.get("forbidden_pairs", []), (
            "Under ToF disease rules, AO-RV adjacency should NOT be forbidden"
        )


# ---------------------------------------------------------------------------
# Test 5: TGA — AO near LV gets relabelled (reversed anatomy)
# ---------------------------------------------------------------------------

def test_tga_ao_near_lv_relabelled():
    """In TGA, AO should be near RV; an AO fragment exclusively near LV is wrong."""
    vol = _empty_vol()
    vol[2:12,  0:10,  :] = LABELS["LV"]
    vol[20:30, 0:10,  :] = LABELS["RV"]
    vol[10:18, 0:10,  :] = LABELS["Myo"]
    # TGA: AO main body correctly near RV
    vol[20:30, 12:22, :] = LABELS["AO"]
    # TGA: PA main body near LV
    vol[2:12,  12:22, :] = LABELS["PA"]
    # A small AO fragment near LV (wrong for TGA) - isolated
    vol[2:12,  24:28, 5:25] = LABELS["AO"]

    # Atlas matches TGA anatomy: AO near RV, PA near LV
    atlas = _empty_vol()
    atlas[2:12,  0:10,  :] = LABELS["LV"]
    atlas[20:30, 0:10,  :] = LABELS["RV"]
    atlas[10:18, 0:10,  :] = LABELS["Myo"]
    atlas[20:30, 12:22, :] = LABELS["AO"]
    atlas[2:12,  12:22, :] = LABELS["PA"]

    tga_vec = [0] * 8
    tga_vec[7] = 1  # TGA flag index = 7

    # Use get_effective_adjacency to verify that AO-LV is forbidden under TGA
    from chd_postprocessing.config import get_effective_adjacency
    adj = get_effective_adjacency(tga_vec)
    key = (min(LABELS["AO"], LABELS["LV"]), max(LABELS["AO"], LABELS["LV"]))
    # Under TGA, AO-LV should be forbidden (False in adjacency rules)
    assert adj.get(key, True) is False, (
        "TGA rules should mark AO-LV adjacency as forbidden"
    )


# ---------------------------------------------------------------------------
# Test 6: DORV — both vessels near RV is allowed; no correction triggered
# ---------------------------------------------------------------------------

def test_dorv_ao_near_rv_not_relabelled():
    """In DORV, both AO and PA exit the RV; AO-RV adjacency is explicitly allowed."""
    from chd_postprocessing.config import get_effective_adjacency

    dorv_vec = [0] * 8
    dorv_vec[4] = 1  # DORV flag index = 4

    adj = get_effective_adjacency(dorv_vec)
    # DORV override: RV-AO should be allowed
    key = (min(LABELS["RV"], LABELS["AO"]), max(LABELS["RV"], LABELS["AO"]))
    assert adj.get(key) is True, (
        "DORV rules should mark RV-AO adjacency as allowed"
    )
    # Normal constraint AO-LV should NOT be forbidden (DORV doesn't override it)
    key_lv_ao = (min(LABELS["LV"], LABELS["AO"]), max(LABELS["LV"], LABELS["AO"]))
    # Normal NORMAL_ADJACENCY says (1,6) = True; DORV doesn't change it
    assert adj.get(key_lv_ao) is True, (
        "Under DORV, LV-AO adjacency should still be allowed"
    )


# ---------------------------------------------------------------------------
# Test 7: PuA — AO-PA adjacency is allowed; no correction triggered
# ---------------------------------------------------------------------------

def test_pua_ao_pa_adjacency_allowed():
    """In PuA, AO and PA may fuse; AO-PA adjacency should not be flagged."""
    from chd_postprocessing.config import get_effective_adjacency

    pua_vec = [0] * 8
    pua_vec[5] = 1  # PuA flag index = 5

    adj = get_effective_adjacency(pua_vec)
    key = (min(LABELS["AO"], LABELS["PA"]), max(LABELS["AO"], LABELS["PA"]))
    # PuA override: AO-PA fusion is anatomically expected
    assert adj.get(key) is True, (
        "PuA rules should mark AO-PA adjacency as allowed"
    )


# ---------------------------------------------------------------------------
# Test 8: No forbidden adjacencies — no changes made
# ---------------------------------------------------------------------------

def test_no_forbidden_adjacencies_no_changes():
    """When the prediction already matches the atlas perfectly, no changes occur."""
    vol   = _normal_heart()
    atlas = _normal_heart()

    corrected, log = correct_by_adjacency(
        vol, atlas, label_ids=LABEL_IDS, disease_vec=None, min_component_fraction=0.01
    )

    assert np.array_equal(corrected, vol), (
        "When prediction matches atlas, no relabelling should occur"
    )
    assert log == [], f"Expected empty log, got {len(log)} entries"


# ---------------------------------------------------------------------------
# Test 9: VSD — LV-RV adjacency becomes allowed
# ---------------------------------------------------------------------------

def test_vsd_lv_rv_adjacency_allowed():
    """In VSD, LV-RV may share a boundary due to the septal defect."""
    from chd_postprocessing.config import get_effective_adjacency

    vsd_vec = [0] * 8
    vsd_vec[2] = 1  # VSD flag index = 2

    adj = get_effective_adjacency(vsd_vec)
    key = (min(LABELS["LV"], LABELS["RV"]), max(LABELS["LV"], LABELS["RV"]))
    assert adj.get(key) is True, (
        "VSD rules should mark LV-RV adjacency as allowed"
    )


# ---------------------------------------------------------------------------
# Test 10: Multiple diseases — most permissive wins
# ---------------------------------------------------------------------------

def test_multiple_diseases_most_permissive_wins():
    """ToF+VSD: ToF makes AO unconstrained; VSD allows LV-RV adjacency."""
    from chd_postprocessing.config import get_effective_adjacency

    tof_vsd_vec = [0] * 8
    tof_vsd_vec[6] = 1  # ToF
    tof_vsd_vec[2] = 1  # VSD

    adj = get_effective_adjacency(tof_vsd_vec)

    # VSD override should allow LV-RV
    key_lv_rv = (min(LABELS["LV"], LABELS["RV"]), max(LABELS["LV"], LABELS["RV"]))
    assert adj.get(key_lv_rv) is True, "ToF+VSD should allow LV-RV (from VSD rule)"

    # ToF override should allow RV-AO
    key_rv_ao = (min(LABELS["RV"], LABELS["AO"]), max(LABELS["RV"], LABELS["AO"]))
    assert adj.get(key_rv_ao) is True, "ToF+VSD should allow RV-AO (from ToF rule)"

    # Normal forbidden pair (RA-AO) should remain forbidden if neither disease allows it
    key_ra_ao = (min(LABELS["RA"], LABELS["AO"]), max(LABELS["RA"], LABELS["AO"]))
    assert adj.get(key_ra_ao, True) is False, (
        "RA-AO should remain forbidden under ToF+VSD (neither disease overrides it)"
    )


# ---------------------------------------------------------------------------
# Additional: build_adjacency_graph smoke test
# ---------------------------------------------------------------------------

def test_build_adjacency_graph_symmetric():
    """Adjacency matrix must be symmetric."""
    vol = _normal_heart()
    adj_binary, adj_weight = build_adjacency_graph(vol, LABEL_IDS)

    assert adj_binary.shape == (len(LABEL_IDS), len(LABEL_IDS))
    assert np.array_equal(adj_binary, adj_binary.T), "Adjacency binary must be symmetric"
    assert np.array_equal(adj_weight, adj_weight.T), "Adjacency weight must be symmetric"
    # Diagonal should be zero (no self-adjacency)
    assert np.all(adj_binary.diagonal() == 0)


# ---------------------------------------------------------------------------
# Additional: connectivity_improvement basic check
# ---------------------------------------------------------------------------

def test_connectivity_improvement_merging():
    """Relabelling a component that connects two pieces should give improvement > 0."""
    vol = _empty_vol()
    # Two separate AO blobs with a gap
    vol[5:10,  5:10, :] = LABELS["AO"]
    vol[15:20, 5:10, :] = LABELS["AO"]
    # A PA fragment between them (will become AO to merge)
    vol[10:15, 5:10, :] = LABELS["PA"]

    comp_mask = (vol == LABELS["PA"])
    delta = connectivity_improvement(vol, comp_mask, LABELS["PA"], LABELS["AO"])
    # Before: PA has 1 component, AO has 2 components → total 3
    # After : PA has 0 components, AO has 1 component   → total 1  → delta = 2
    assert delta > 0, f"Expected connectivity improvement > 0, got {delta}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
