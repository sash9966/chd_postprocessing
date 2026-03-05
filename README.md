# CHD Post-Processing

Post-processing for nnU-Net congenital heart disease (CHD) segmentation.

**Problem:** nnU-Net achieves ~0.95 whole-heart Dice but only ~0.70–0.80 per-class Dice.
The network finds the correct spatial extent of the heart but misassigns labels to some
regions — e.g. part of the aorta is labeled as pulmonary artery, or a fragment of right
atrium appears as aorta. This package corrects those misassignments without discarding any
voxels.

---

## Installation

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `nibabel`, `pandas`.

---

## Labels (imageCHD dataset.json)

| ID | Structure |
|----|-----------|
| 0  | background |
| 1  | LV-BP (left ventricle blood pool) |
| 2  | RV-BP (right ventricle blood pool) |
| 3  | LA (left atrium) |
| 4  | RA (right atrium) |
| 5  | Myo (myocardium) |
| 6  | Aorta |
| 7  | Pulmonary |

---

## Quick Start

### Single prediction

```python
from chd_postprocessing.atlas_pipeline import run_atlas_pipeline

result = run_atlas_pipeline(
    pred_path        = "pred/ct_1004.nii.gz",
    gt_folder        = "labelsTr/",
    output_path      = "corrected/ct_1004.nii.gz",
    mode             = "disease_specific",   # or "baseline"
    disease_map_path = "disease_map.json",
    gt_path          = "gt/ct_1004.nii.gz",  # optional: enables Dice evaluation
)

print(result.reassignment_summary)
print("Dice before:", result.dice_before)
print("Dice after: ", result.dice_after)
```

### Whole folder

```python
from chd_postprocessing.atlas_pipeline import run_atlas_folder_pipeline

df = run_atlas_folder_pipeline(
    pred_folder      = "preds/",
    gt_folder        = "labelsTr/",
    output_folder    = "corrected/",
    mode             = "disease_specific",
    disease_map_path = "disease_map.json",
    gt_folder_eval   = "labelsTs/",
)
print(df.groupby("status").size())
```

### AO/PA fragment correction only (no atlas needed)

```python
from chd_postprocessing.anatomy_priors import correct_ao_pa_fragments

corrected, log = correct_ao_pa_fragments(
    labels      = pred_array,
    disease_vec = [0, 0, 0, 0, 0, 0, 0, 0],   # all-normal; PuA=1 skips correction
    spacing_mm  = (1.5, 1.5, 1.5),
)
print(f"Reassigned {len(log['reassigned'])} fragments")
```

---

## Pipeline Architecture

```
Input: nnU-Net prediction (.nii.gz)
  │
  ▼
Step 1 — Fragment-level AO/PA anatomy correction   [anatomy_priors.py]
  │  No registration needed.
  │  Each AO/PA connected component is tested for ventricle adjacency.
  │  AO fragment near RV → relabeled PA.  PA fragment near LV → relabeled AO.
  │  PuA disease exception respected.
  │
  ▼
Step 2 — Atlas selection                            [atlas.py]
  │  "baseline": random atlas from GT library
  │  "disease_specific": atlas with minimum Hamming distance on disease flags
  │
  ▼
Step 3 — Centroid registration                      [registration.py]
  │  Translates atlas foreground centroid to match prediction centroid.
  │  PCA mode available for additional rotation alignment.
  │
  ▼
Step 4 — Component-level IoC assignment             [label_correction.py]
  │  Finds all connected components per predicted label.
  │  For each EXTRA (non-dominant) component: compute IoC against every
  │  atlas label region. Reassign to atlas label with highest IoC.
  │  Dominant (largest) component of each label is never modified.
  │
  ▼
Step 5 — Morphological closing                      [label_correction.py]
  │  Fills small background holes inside AO and PA only.
  │
  ▼
Output: corrected label volume + AtlasPipelineResult
```

---

## Disease Map Format

`disease_map.json` maps case IDs to 8-element binary disease flag vectors:

```json
{
  "ct_1004_image": [0, 0, 1, 0, 0, 0, 0, 0],
  "ct_1005_image": [0, 0, 0, 0, 0, 1, 0, 0]
}
```

Flag order: `[HLHS, ASD, VSD, AVSD, DORV, PuA, ToF, TGA]`

---

## Module Reference

| Module | Purpose |
|--------|---------|
| `config.py` | Label IDs, disease flag names, default thresholds |
| `io_utils.py` | NIfTI load/save, disease map parsing, case ID resolution |
| `atlas.py` | `AtlasEntry`, `AtlasLibrary`, atlas selection logic |
| `registration.py` | Centroid/PCA rigid registration of atlas → prediction space |
| `anatomy_priors.py` | AO/PA ventricle-adjacency correction (global + fragment-level) |
| `label_correction.py` | Component-level IoC-based label reassignment |
| `evaluate.py` | Per-class Dice metrics, folder-level batch evaluation |
| `pipeline.py` | Legacy pipeline: CC cleanup + anatomy priors only |
| `atlas_pipeline.py` | Full pipeline: anatomy correction + atlas registration + IoC |

---

## Key Design Decisions

**Why IoC instead of Dice for atlas matching?**
Dice penalises size imbalance: a 50-voxel fragment perfectly inside a 6400-voxel
atlas region gets Dice ≈ 0.015, making it indistinguishable from noise. IoC
(= overlap / fragment_size) scores the same fragment at 1.0, reliably identifying
the correct atlas label regardless of the atlas region's total size.

**Why lock the dominant component?**
The nnU-Net prediction is mostly correct — the dominant (largest) connected component
of each label is almost certainly right. Reassigning it via an imperfectly registered
atlas introduces more errors than it fixes. Only secondary fragments (anatomically
implausible as separate disconnected structures) are candidates for reassignment.

**Why anatomy priors before atlas?**
The AO/PA adjacency test requires no registration and is highly reliable for the
most common error pattern. Running it first means the atlas step operates on
already-corrected vessel labels.

**Why not just take the largest connected component?**
That discards correctly-segmented voxels. If AO has two components — one main body
and one fragment that's actually PA — removing the small component destroys real PA
voxels. The correct action is to relabel the fragment, not delete it.

---

## Running Tests

```bash
python3 -m pytest tests/ -v
```

49 tests, all synthetic (no NIfTI files on disk required), run in ~1.3 s.

---

## Known Limitations & Roadmap

See `ANALYSIS.md` for a full breakdown of what's working, what's failing, and
prioritised improvements. Short summary:

1. **Registration quality** is the main bottleneck. Centroid-only alignment is too
   coarse for patients with very different cardiac geometries. Per-structure centroid
   registration or multi-atlas voting would improve fragment reassignment reliability.

2. **Atlas pool size** matters. With 10–15 atlases, disease-specific selection may
   not find a close match for rare disease combinations.

3. **Other vessel diseases** (DORV, TGA) produce atypical vessel-ventricle
   relationships that are not yet excluded from the AO/PA anatomy prior.
