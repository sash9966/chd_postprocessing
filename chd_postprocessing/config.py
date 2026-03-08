"""Label and disease flag configuration for the imageCHD dataset."""

from typing import Dict, List, Optional, Tuple

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

# ---------------------------------------------------------------------------
# Disease-specific anatomy rules
# ---------------------------------------------------------------------------
# Per disease flag index (matching DISEASE_FLAGS order):
#   vessel_ventricle : {vessel_label: ventricle_label}  (None = unconstrained)
#   skip_ao_pa_correction : True  →  skip the AO/PA fragment correction entirely
# Labels not listed use the normal AO→LV / PA→RV prior.
DISEASE_ANATOMY_RULES: Dict[int, Dict] = {
    0: {  # HLHS — hypoplastic left heart syndrome
        "name": "HLHS",
        "vessel_ventricle": {},
        "skip_ao_pa_correction": True,
        "notes": "LV may be absent or tiny; AO/PA correction unreliable",
    },
    4: {  # DORV — double outlet right ventricle
        "name": "DORV",
        "vessel_ventricle": {6: 2, 7: 2},   # AO→RV, PA→RV  (both from RV)
        "skip_ao_pa_correction": False,
        "notes": "Both great vessels exit the right ventricle",
    },
    5: {  # PuA — pulmonary atresia
        "name": "PuA",
        "vessel_ventricle": {},
        "skip_ao_pa_correction": True,
        "notes": "AO/PA fusion is anatomically expected; correction skipped",
    },
    6: {  # ToF — tetralogy of Fallot
        "name": "ToF",
        "vessel_ventricle": {6: None, 7: 2},  # AO unconstrained, PA→RV
        "skip_ao_pa_correction": False,
        "notes": "Overriding aorta straddles the VSD; AO adjacency unconstrained",
    },
    7: {  # TGA — transposition of the great arteries
        "name": "TGA",
        "vessel_ventricle": {6: 2, 7: 1},   # AO→RV, PA→LV  (reversed)
        "skip_ao_pa_correction": False,
        "notes": "Great arteries are transposed; AO exits RV, PA exits LV",
    },
}

# ---------------------------------------------------------------------------
# Structural adjacency constraints
# ---------------------------------------------------------------------------
# NORMAL_ADJACENCY : expected label–label adjacency for a normal heart.
#   True  → the pair *should* be adjacent (anatomically expected)
#   False → the pair should *not* be adjacent (violation if seen)
# Keys are (min_label, max_label) tuples.
NORMAL_ADJACENCY: Dict[Tuple[int, int], bool] = {
    (1, 5): True,   # LV–Myo : always adjacent
    (2, 5): True,   # RV–Myo : always adjacent
    (1, 6): True,   # LV–AO  : aorta exits LV
    (2, 7): True,   # RV–PA  : pulmonary artery exits RV
    (3, 1): True,   # LA–LV  : left-side continuity
    (4, 2): True,   # RA–RV  : right-side continuity
    (3, 7): True,   # LA–PA  : anatomically plausible
    (6, 7): True,   # AO–PA  : great vessels are adjacent at root
    (4, 6): False,  # RA–AO  : should not be directly adjacent
    (3, 6): False,  # LA–AO  : should not be directly adjacent
}

# Disease-specific adjacency overrides — indexed by DISEASE_FLAGS position.
# Each override adds or removes adjacency constraints for that disease.
#   True  → override the pair to *allowed* (most permissive wins)
#   False → override the pair to *forbidden*
DISEASE_ADJACENCY_OVERRIDES: Dict[int, Dict[Tuple[int, int], bool]] = {
    0: {},                                              # HLHS  (no extra pairs)
    1: {(3, 4): True},                                  # ASD   : LA–RA may connect
    2: {(1, 2): True},                                  # VSD   : LV–RV may connect
    3: {(1, 2): True, (3, 4): True,                     # AVSD  : all septal borders
        (1, 4): True, (2, 3): True},
    4: {(2, 6): True},                                  # DORV  : RV–AO allowed
    5: {(6, 7): True},                                  # PuA   : AO–PA fusion allowed
    6: {(1, 2): True, (2, 6): True},                    # ToF   : VSD + RV–AO
    7: {(2, 6): True, (1, 7): True,                     # TGA   : reversed vessel exits
        (1, 6): False, (2, 7): False},                  #          normal exits forbidden
}


def get_effective_adjacency(
    disease_vec: Optional[List[int]] = None,
) -> Dict[Tuple[int, int], bool]:
    """Return the adjacency constraint dict for the given disease profile.

    Starts from :data:`NORMAL_ADJACENCY` and applies any relevant
    :data:`DISEASE_ADJACENCY_OVERRIDES`.  The "most permissive wins"
    rule is applied when multiple diseases are active:

    * An override that *allows* a pair always takes effect.
    * An override that *forbids* a pair only takes effect if the pair
      is not already allowed by another active disease or by the normal rules.

    Parameters
    ----------
    disease_vec : 8-element binary flag list, or ``None`` (use normal anatomy).

    Returns
    -------
    Dict mapping ``(min_label, max_label)`` tuples to ``bool``
    (``True`` = allowed, ``False`` = forbidden).
    """
    # Start from normal adjacency (normalise key ordering)
    adj: Dict[Tuple[int, int], bool] = {}
    for (a, b), allowed in NORMAL_ADJACENCY.items():
        adj[(min(a, b), max(a, b))] = allowed

    if disease_vec is None:
        return adj

    # Collect all active-disease overrides in two sets so that
    # "most permissive wins" applies *between diseases*, while each
    # disease can unconditionally override the normal-anatomy baseline.
    allowed_by_disease: set = set()
    forbidden_by_disease: set = set()

    for flag_idx, is_active in enumerate(disease_vec):
        if not is_active:
            continue
        if flag_idx not in DISEASE_ADJACENCY_OVERRIDES:
            continue
        for (a, b), allowed in DISEASE_ADJACENCY_OVERRIDES[flag_idx].items():
            key = (min(a, b), max(a, b))
            if allowed:
                allowed_by_disease.add(key)
            else:
                forbidden_by_disease.add(key)

    # Apply forbidden overrides first (these override normal anatomy).
    for key in forbidden_by_disease:
        adj[key] = False

    # Apply allowed overrides last (most permissive wins:
    # if any active disease explicitly allows a pair it stays allowed,
    # even if another disease or the forbidden set says otherwise).
    for key in allowed_by_disease:
        adj[key] = True

    return adj
