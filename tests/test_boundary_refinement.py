"""Tests for boundary_refinement module.

Synthetic 32×32×32 volumes are used for deterministic assertions about
per-voxel boundary correction behavior.

Label IDs:
  1=LV, 2=RV, 3=LA, 4=RA, 5=Myo, 6=AO, 7=PA
"""
import numpy as np
import pytest

from chd_postprocessing.boundary_refinement import (
    find_boundary_zone,
    local_majority_label,
    centroid_distance_score,
    refine_label_boundary,
    refine_all_boundaries,
)
from chd_postprocessing.config import LABELS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LABEL_IDS = list(range(1, 8))

AO = LABELS["AO"]   # 6
PA = LABELS["PA"]   # 7
RA = LABELS["RA"]   # 4
LV = LABELS["LV"]   # 1


def _empty() -> np.ndarray:
    return np.zeros((32, 32, 32), dtype=np.int32)


def _normal_heart() -> np.ndarray:
    """Well-separated, anatomically correct heart layout.

    AO (x=0:14)  |  PA (x=18:32)  separated by a gap at x=14:18.
    Both present but not touching.  All other labels absent.
    """
    vol = _empty()
    vol[0:14, :, :] = AO
    vol[18:32, :, :] = PA
    return vol


# ---------------------------------------------------------------------------
# Test 1: boundary error corrected
# ---------------------------------------------------------------------------

def test_boundary_error_corrected():
    """An isolated AO blob deep inside a large RA region should be reassigned
    from AO → RA by refine_label_boundary(AO, RA).

    Layout:
      RA fills the entire volume.
      AO main body: x=28:32, y=28:32 (far corner, isolated from blob).
      AO error blob: 3×3×3 cube at (5:8, 5:8, 5:8) — completely surrounded by RA,
      far from the main AO body, so local majority and centroid both say RA.
    """
    vol = _empty()
    vol[:, :, :] = RA                  # fill with RA
    vol[28:32, 28:32, :] = AO          # main AO body (far corner)
    vol[5:8, 5:8, 5:8] = AO            # erroneous AO blob deep inside RA

    corrected, log = refine_label_boundary(
        vol, label_a=AO, label_b=RA,
        width_voxels=4, kernel_size=5, min_confidence=0.3, max_fraction=1.0,
    )

    # The erroneous blob is in zone_a (AO near RA) and should be reassigned
    # AO → RA → a_to_b > 0.
    assert log["a_to_b"] > 0, f"Expected a_to_b > 0, got {log}"
    # Most of the blob should now be RA
    blob_ao = int((corrected[5:8, 5:8, 5:8] == AO).sum())
    blob_total = 3 * 3 * 3
    assert blob_ao < blob_total, (
        f"Expected some blob voxels corrected to RA, but still {blob_ao}/{blob_total} AO"
    )


# ---------------------------------------------------------------------------
# Test 2: correct boundary untouched
# ---------------------------------------------------------------------------

def test_correct_boundary_untouched():
    """A clean AO–PA boundary with no mislabeling should produce zero changes.

    AO occupies x=0:16, PA occupies x=16:32 — perfectly separated halves.
    The majority label at each boundary voxel is already the correct one.
    """
    vol = _empty()
    vol[0:16, :, :] = AO
    vol[16:32, :, :] = PA

    corrected, log = refine_label_boundary(
        vol, label_a=AO, label_b=PA,
        width_voxels=3, kernel_size=5, min_confidence=0.6,
    )

    assert log["a_to_b"] == 0, f"Expected no AO→PA reassignment on clean boundary, got {log}"
    assert log["b_to_a"] == 0, f"Expected no PA→AO reassignment on clean boundary, got {log}"
    assert np.array_equal(corrected, vol), "Volume should be unchanged for correct boundary"


# ---------------------------------------------------------------------------
# Test 3: min_confidence threshold respected
# ---------------------------------------------------------------------------

def test_min_confidence_threshold():
    """With a very high confidence threshold, no voxel should be reassigned
    when signals are mixed (equal neighbors of each label).

    Layout: alternating AO/RA stripes of 1 voxel each — no clear majority.
    """
    vol = _empty()
    for i in range(32):
        if i % 2 == 0:
            vol[i, :, :] = AO
        else:
            vol[i, :, :] = RA

    corrected, log = refine_label_boundary(
        vol, label_a=AO, label_b=RA,
        width_voxels=2, kernel_size=3, min_confidence=0.9, max_fraction=1.0,
    )

    assert log["a_to_b"] == 0, f"High confidence threshold should prevent reassignment, got {log}"
    assert log["b_to_a"] == 0, f"High confidence threshold should prevent reassignment, got {log}"


# ---------------------------------------------------------------------------
# Test 4: max_fraction limit respected
# ---------------------------------------------------------------------------

def test_max_fraction_limit():
    """Even when all boundary voxels strongly prefer the candidate label,
    at most max_fraction of zone voxels are reassigned per pass.

    Layout: AO (x=0:8) | RA (x=8:32). The AO voxels at x=5:8 are in the
    boundary zone and surrounded by RA; they should prefer RA.
    With max_fraction=0.1, only 10% of the zone may be reassigned.
    """
    vol = _empty()
    vol[0:8, :, :] = AO
    vol[8:32, :, :] = RA

    # Count zone_a size: AO voxels within width_voxels=3 of RA
    zone_a, _ = find_boundary_zone(vol, AO, RA, width_voxels=3)
    zone_a_size = int(zone_a.sum())

    corrected, log = refine_label_boundary(
        vol, label_a=AO, label_b=RA,
        width_voxels=3, kernel_size=5, min_confidence=0.0, max_fraction=0.1,
    )

    max_allowed = max(1, int(np.ceil(0.1 * zone_a_size)))
    assert log["a_to_b"] <= max_allowed, (
        f"a_to_b={log['a_to_b']} exceeds 10% cap of {max_allowed} "
        f"(zone_a_size={zone_a_size})"
    )


# ---------------------------------------------------------------------------
# Test 5: AO–PA boundary swap
# ---------------------------------------------------------------------------

def test_ao_pa_boundary_swap():
    """PA voxels embedded in an AO region (surrounded mostly by AO) should be
    reassigned PA → AO by refine_label_boundary(AO, PA).

    Layout:
      AO: x=0:28 (large dominant region)
      PA: x=28:32 (main PA body)
      PA error patch: a small 3×3×3 cube at (10,10,10) labeled PA surrounded by AO.
    """
    vol = _empty()
    vol[0:28, :, :] = AO       # dominant AO
    vol[28:32, :, :] = PA      # main PA body
    # Embed a small erroneous PA patch deep inside AO
    vol[10:13, 10:13, 10:13] = PA  # PA error, surrounded by AO on all sides

    corrected, log = refine_label_boundary(
        vol, label_a=AO, label_b=PA,
        width_voxels=4, kernel_size=5, min_confidence=0.2, max_fraction=1.0,
    )

    # The erroneous PA patch is in zone_b (PA near AO); it should be reassigned
    # PA → AO (b_to_a > 0)
    assert log["b_to_a"] > 0, (
        f"Expected PA→AO reassignment for embedded patch, got {log}"
    )
    # The patch (or most of it) should now be AO
    patch_ao = int((corrected[10:13, 10:13, 10:13] == AO).sum())
    patch_total = 3 * 3 * 3
    assert patch_ao > patch_total // 2, (
        f"Expected majority of erroneous PA patch to be corrected to AO, "
        f"got {patch_ao}/{patch_total} AO voxels"
    )


# ---------------------------------------------------------------------------
# Test 6: refine_all_boundaries smoke test
# ---------------------------------------------------------------------------

def test_refine_all_boundaries_no_crash():
    """refine_all_boundaries should run without error on a normal-ish heart
    volume and return the same shape.  If the input is already correct
    (clean boundaries), the output should equal the input.
    """
    vol = _empty()
    vol[0:10,  0:16,  :] = LV
    vol[0:10,  16:32, :] = AO
    vol[22:32, 0:16,  :] = PA
    vol[10:22, :,     :] = RA

    corrected, log_list = refine_all_boundaries(
        vol,
        atlas_reg=None,
        label_ids=[LV, RA, AO, PA],
        disease_vec=None,
        width_voxels=2,
        kernel_size=3,
        min_confidence=0.9,  # very high threshold → no changes on clean boundaries
        max_passes=2,
    )

    assert corrected.shape == vol.shape, "Output shape should match input"
    assert isinstance(log_list, list), "log_list should be a list"
    # With very high confidence threshold on a relatively clean volume,
    # expect no or minimal changes
    total_changes = sum(e.get("a_to_b", 0) + e.get("b_to_a", 0) for e in log_list)
    assert total_changes == 0, (
        f"Expected no changes with min_confidence=0.9 on clean boundaries, "
        f"got {total_changes} changes: {log_list}"
    )
