"""Label and disease flag configuration for the imageCHD dataset."""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Segmentation label map — internal short keys used throughout the codebase
# (e.g. LABELS["AO"] = 6, LABELS["LV"] = 1)
# ---------------------------------------------------------------------------
LABELS: Dict[str, int] = {
    "background": 0,
    "LV":  1,   # left ventricle blood pool
    "RV":  2,   # right ventricle blood pool
    "LA":  3,   # left atrium
    "RA":  4,   # right atrium
    "Myo": 5,   # myocardium
    "AO":  6,   # aorta
    "PA":  7,   # pulmonary artery
}

# ---------------------------------------------------------------------------
# Display names matching dataset.json exactly.
# Used for all human-readable output: Dice result dicts, summary strings, CSV.
# Kept separate from LABELS so internal key lookups (LABELS["AO"]) still work.
# ---------------------------------------------------------------------------
LABEL_NAMES: Dict[int, str] = {
    0: "background",
    1: "LV-BP",
    2: "RV-BP",
    3: "LA",
    4: "RA",
    5: "Myo",
    6: "Aorta",
    7: "Pulmonary",
}

# Foreground class indices (excludes background)
FOREGROUND_CLASSES: List[int] = list(range(1, 8))

# ---------------------------------------------------------------------------
# Anatomical prior: each great vessel should be adjacent to its ventricle
#   AO exits from the left  ventricle (LV)
#   PA exits from the right ventricle (RV)
# When the network swaps AO/PA labels, "AO" will be adjacent to RV and
# "PA" to LV — the opposite of the correct anatomy below.
# ---------------------------------------------------------------------------
VESSEL_VENTRICLE_PRIOR: Dict[int, int] = {
    LABELS["AO"]: LABELS["LV"],  # 6 → 1
    LABELS["PA"]: LABELS["RV"],  # 7 → 2
}

# ---------------------------------------------------------------------------
# Disease flags (index order in the binary disease vector)
# ---------------------------------------------------------------------------
DISEASE_FLAGS: List[str] = [
    "HLHS",  # 0 – hypoplastic left heart syndrome
    "ASD",   # 1 – atrial septal defect
    "VSD",   # 2 – ventricular septal defect
    "AVSD",  # 3 – atrioventricular septal defect
    "DORV",  # 4 – double outlet right ventricle
    "PuA",   # 5 – pulmonary atresia  ← KEY FLAG
    "ToF",   # 6 – tetralogy of Fallot
    "TGA",   # 7 – transposition of the great arteries
]
PUA_FLAG_INDEX: int = 5  # index of PuA in the disease vector

# ---------------------------------------------------------------------------
# Post-processing defaults
# ---------------------------------------------------------------------------
# Dilation radius (mm) used to test vessel–ventricle adjacency.
# Large enough to bridge small segmentation gaps (~2–3 voxels at 1 mm spacing).
DEFAULT_DILATION_RADIUS_MM: float = 3.0

# Cases whose adjacency confidence falls below this threshold are flagged
# for manual review rather than automatically corrected.
CONFIDENCE_THRESHOLD: float = 0.15

# Connected-component cleanup: fragments smaller than this fraction of the
# largest component are removed (set to background).
MIN_COMPONENT_FRACTION: float = 0.01
