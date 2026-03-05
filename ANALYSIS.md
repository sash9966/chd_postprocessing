# Post-Processing Analysis: What's Working, What's Failing, and Why

## The Problem

nnU-Net achieves **WH (whole-heart) Dice ≈ 0.95**, meaning it finds the correct spatial
extent of the heart almost perfectly. But individual class Dice scores are significantly
lower — Aorta ≈ 0.79, Pulmonary ≈ 0.70. The network predicts the right *shape* of the
heart but assigns wrong *labels* to some regions. Typical errors:

- A fragment of AO (aorta) is labeled as PA (pulmonary artery)
- A fragment of PA is labeled as AO
- A fragment of RA (right atrium) is labeled as AO

Because WH = 0.95 regardless of which post-processing is applied, **total foreground
voxel count is conserved**. Post-processing cannot add information the network didn't
find — it can only **reassign labels across voxels that already exist as foreground**.

---

## Current Architecture

```
atlas_pipeline.py
  └─ correct_labels_with_atlas()        ← main entry point
       ├─ _find_all_components()         ← scipy CC per label
       ├─ _compute_component_overlaps()  ← IoC matrix (n_comps × n_labels)
       ├─ _initial_assignments()         ← argmax per row
       ├─ _resolve_multi_component_conflicts()  ← keep largest, reassign tiny
       └─ apply_morphological_cleanup()  ← close holes in AO/PA

anatomy_priors.py                        ← SEPARATE module, NOT used in atlas pipeline
  └─ correct_ao_pa_labels()             ← ventricle-adjacency swap detection
```

---

## Root Cause Analysis

### Issue 1: Centroid Registration is Too Coarse

`register_atlas_to_pred` (centroid mode) translates the atlas so its foreground centroid
matches the prediction's foreground centroid. This is a **single global translation** of
the entire atlas.

**Why this fails:** Two CHD patients can have dramatically different cardiac geometries
even after centroid alignment. After registration, the atlas AO region might land 15–20 mm
away from the prediction's AO region. When we then compute IoC between a misclassified
fragment and atlas regions, the atlas is not in the right place to give useful signal.

**Consequence:** Components that should get reassigned get IoC ≈ 0 for ALL atlas labels
(because the atlas is globally misaligned), so they keep their original (wrong) label.
Components that should stay the same might get reassigned to the wrong label if the atlas
happens to land another label's region on top of them.

**Evidence:** Every individual class Dice score is *lower* after atlas post-processing
than the raw prediction. If the registration were working, at least some classes should
improve. All-class degradation points to systematic assignment errors from poor registration.

### Issue 2: The anatomy_priors.py Module Exists But Is NOT Integrated

`anatomy_priors.py` implements exactly the right tool for AO/PA correction:
- No registration needed
- Uses anatomy: AO exits LV, PA exits RV
- Tests adjacency via binary dilation
- Handles PuA disease exception
- Already has confidence scores and manual-review flags

**This module is used in `pipeline.py` but completely absent from `atlas_pipeline.py`.**
This is the biggest missed opportunity in the codebase.

The anatomy prior also handles only the **global swap case** (both AO and PA are entirely
on the wrong side). It does not handle the **fragment case** (a few voxels of AO are
labeled as PA while the main AO body is correct). This is the extension that's needed.

### Issue 3: IoC Applied to All Components Including Large Correct Ones

The current `_initial_assignments()` runs IoC-argmax on every component, including the
large, likely-correct dominant components of each label. If registration is off by even
5–10 mm, a correctly-labeled large LV component might get the highest IoC with atlas
RV (if the atlas RV happens to land there), causing a massive false relabeling.

The algorithm should **conserve large dominant components** and only operate on the
extra fragments.

### Issue 4: Conflict Resolution Threshold Too Aggressive

`_resolve_multi_component_conflicts` reassigns fragments smaller than
`min_component_fraction × largest` (default 1%). For a 6400-voxel LV, only fragments
below 64 voxels are candidates. But clinically significant mislabeled fragments (e.g.,
200-voxel AO fragment labeled as PA) are well above this threshold and don't get
reassigned.

---

## What IS Working

1. **IoC metric over Dice**: The switch from Dice to intersection-over-component is
   correct. A 50-voxel fragment inside atlas AO scores 1.0 with IoC vs 0.015 with Dice.
   The metric is right; the registration quality undermines it.

2. **No-removal guarantee**: Removing `enforce_single_component` was the right call.
   Voxels are never set to background 0, so WH Dice cannot decrease.

3. **PCA sign fix in registration**: The det=+1 enforcement in `_pca_axes` prevents
   mirror-flips that would swap left/right structures.

4. **Component decomposition**: Breaking predictions into connected components per label
   is the right granularity for this problem.

5. **`anatomy_priors.py`**: The ventricle-adjacency logic is sound and accurate. It just
   needs to be (a) integrated into the atlas pipeline and (b) extended to the fragment level.

---

## Recommended Fixes (Priority Order)

### Fix 1 — Fragment-Level Anatomy Correction for AO/PA (HIGH PRIORITY)
Extend `anatomy_priors.py` with a per-component version of the adjacency test:
- For each connected component of AO: test if it's adjacent to LV (correct) or RV (wrong)
- Components adjacent to RV → relabel as PA
- For each connected component of PA: test if it's adjacent to RV (correct) or LV (wrong)
- Components adjacent to LV → relabel as AO
- No registration needed. Pure anatomy.

This directly fixes the user's primary observed error (AO/PA fragment swaps).

### Fix 2 — Conservative Atlas: Only Reassign Non-Dominant Components (HIGH PRIORITY)
Change `correct_labels_with_atlas` to:
1. Find the largest component for each label → **lock its label** (never reassign it)
2. Only apply IoC-based reassignment to the *extra* (non-dominant) components
3. Optionally apply anatomy priors as a final pass

This prevents the large false-relabeling caused by imperfect registration.

### Fix 3 — Per-Structure Registration (MEDIUM PRIORITY)
Instead of one centroid for all foreground, compute a per-label centroid translation.
For each component in the prediction, find the atlas label whose centroid is nearest,
then locally translate the atlas mask for that label before computing IoC. This is
still simple (no deformable registration) but orders of magnitude more accurate.

### Fix 4 — Multi-Atlas Voting (LOW PRIORITY, HIGH IMPACT)
Select 3–5 atlases (best disease-profile matches) and let each one vote on the label
for each misclassified fragment (majority rules). This averages out individual atlas
registration errors and is much more robust than single-atlas correction.

---

## Recommended Pipeline Order

```
1. nnU-Net raw prediction
2. Fragment-level AO/PA anatomy correction (Fix 1 — no registration, reliable)
3. Atlas-based IoC reassignment for non-dominant components only (Fix 2)
4. Morphological closing (AO/PA only, only fill background gaps)
```

Step 2 handles the most common and clinically important errors without any registration
risk. Step 3 catches remaining fragment errors for all 7 classes.

---

## Module Dependency Map

```
config.py          ← constants, label IDs, disease flags
io_utils.py        ← NIfTI I/O, disease map loading
atlas.py           ← AtlasEntry, AtlasLibrary, create_synthetic_atlas
registration.py    ← centroid/PCA rigid registration
anatomy_priors.py  ← AO/PA ventricle-adjacency correction  [UNDERUSED]
label_correction.py← component-level IoC assignment        [OVERUSED]
connected_components.py ← CC cleanup utilities
evaluate.py        ← Dice metrics
pipeline.py        ← uses anatomy_priors + CC cleanup      [SEPARATE]
atlas_pipeline.py  ← uses registration + label_correction  [SEPARATE]
```

`pipeline.py` and `atlas_pipeline.py` are two parallel pipelines that don't share their
best features. The ideal pipeline merges both: anatomy priors first, then atlas-guided
fragment reassignment for residual errors.
