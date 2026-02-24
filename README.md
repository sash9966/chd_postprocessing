# nnunet_chd_postprocessing

Anatomy-informed post-processing for nnU-Net congenital heart disease (CHD) segmentation outputs.

## The Problem

nnU-Net achieves excellent whole-heart segmentation (~0.95 mean Dice) on the imageCHD dataset, but confuses **aorta (AO)** and **pulmonary artery (PA)** labels in a subset of cases.  The vessels are segmented in the right spatial location — they are just assigned the wrong class label.  This happens because some training cases have **pulmonary atresia (PuA)** where AO and PA genuinely fuse, and the network learns to sometimes apply that pattern to normal cases where it should not.

This repository applies a simple anatomical rule to detect and correct these label swaps post-hoc, without any retraining.

## Anatomical Prior

```
Left  ventricle (LV) ──── Aorta         (AO, label 6)
Right ventricle (RV) ──── Pulmonary art. (PA, label 7)
```

When labels are swapped, the predicted "AO" region will be *adjacent to the RV* and the predicted "PA" will be *adjacent to the LV*.  The correction algorithm detects this pattern by dilating each vessel mask and counting voxel overlap with each ventricle.  If the overlap ratios contradict the expected anatomy, the labels are exchanged.

**PuA exception**: in pulmonary atresia (disease flag index 5), AO/PA fusion is anatomically correct and the correction is skipped.

## Repository Structure

```
chd_postprocessing/
    config.py               — label names, disease flag ordering, defaults
    io_utils.py             — NIfTI load/save, disease_map.json loading
    anatomy_priors.py       — core AO/PA label correction algorithm
    connected_components.py — small-fragment cleanup (pre-processing step)
    evaluate.py             — per-class Dice computation, folder evaluation
    pipeline.py             — configurable step-chain pipeline
scripts/
    run_postprocessing.py   — CLI: run pipeline + optional evaluation
    evaluate_before_after.py — compare two prediction folders vs GT
    visualize_case.py       — 2D orthogonal slice visualisation
tests/
    test_anatomy_priors.py
    test_connected_components.py
```

## Installation

```bash
git clone https://github.com/your-org/nnunet_chd_postprocessing.git
cd nnunet_chd_postprocessing
pip install -e .
```

Requirements: `nibabel`, `numpy`, `scipy`, `pandas`, `matplotlib`.

## Quick Start

### Post-process a folder of predictions

```bash
python scripts/run_postprocessing.py \
    --input_folder  /path/to/nnunet_predictions/ \
    --output_folder /path/to/corrected/ \
    --disease_map   disease_map.json
```

### Post-process and evaluate against ground truth

```bash
python scripts/run_postprocessing.py \
    --input_folder  /path/to/nnunet_predictions/ \
    --output_folder /path/to/corrected/ \
    --disease_map   disease_map.json \
    --gt_folder     /path/to/ground_truth/ \
    --evaluate
```

This prints a table like:

```
Class      Before    After        Δ
------------------------------------------
LV         0.9612   0.9612   +0.0000
RV         0.9401   0.9401   +0.0000
LA         0.9155   0.9155   +0.0000
RA         0.9023   0.9023   +0.0000
Myo        0.8834   0.8834   +0.0000
AO         0.7821   0.9103   +0.1282  ◄
PA         0.7654   0.9014   +0.1360  ◄
mean_fg    0.8929   0.9306   +0.0377
```

### Compare two prediction folders

```bash
python scripts/evaluate_before_after.py \
    --before /path/to/original/ \
    --after  /path/to/corrected/ \
    --gt     /path/to/ground_truth/ \
    --output results.csv
```

Includes paired Wilcoxon signed-rank test per class.

### Visualise a case

```bash
# Single prediction (3 orthogonal views)
python scripts/visualize_case.py --pred ct_1001_image.nii.gz

# Before vs after comparison
python scripts/visualize_case.py \
    --pred      /path/to/original/ct_1001_image.nii.gz \
    --corrected /path/to/corrected/ct_1001_image.nii.gz \
    --output    ct_1001_comparison.png
```

### Python API

```python
from chd_postprocessing.pipeline import run_pipeline, run_folder_pipeline

# Single file
result = run_pipeline(
    pred_path="ct_1001_image.nii.gz",
    output_path="corrected/ct_1001_image.nii.gz",
    disease_vec=[0, 0, 0, 0, 0, 0, 0, 0],   # [HLHS, ASD, VSD, AVSD, DORV, PuA, ToF, TGA]
)
print(result["correction"]["was_swapped"])     # True / False
print(result["correction"]["confidence_score"])

# Whole folder
results = run_folder_pipeline(
    input_folder="predictions/",
    output_folder="corrected/",
    disease_map_path="disease_map.json",
)
```

## Post-processing Steps

| Step | Description | Default |
|------|-------------|---------|
| `cc_cleanup` | Remove AO/PA fragments smaller than 1 % of the largest connected component. Prevents orphan islands from distorting adjacency scores. | ✓ |
| `adjacency_correction` | Swap AO/PA labels if the current labeling contradicts the ventricle-adjacency prior. PuA cases are skipped. | ✓ |

Run only a subset of steps:

```bash
python scripts/run_postprocessing.py --steps adjacency_correction ...
```

## Label Convention

| Label | ID | Structure |
|-------|-----|-----------|
| background | 0 | — |
| LV  | 1 | Left ventricle |
| RV  | 2 | Right ventricle |
| LA  | 3 | Left atrium |
| RA  | 4 | Right atrium |
| Myo | 5 | Myocardium |
| AO  | 6 | Aorta |
| PA  | 7 | Pulmonary artery |

## Disease Flag Convention

`disease_map.json` maps case IDs to 8-element binary vectors:

```json
{
  "ct_1001_image": [0, 0, 0, 0, 0, 0, 0, 0],
  "ct_1030_image": [0, 0, 1, 0, 0, 1, 0, 0]
}
```

Flag order: `[HLHS, ASD, VSD, AVSD, DORV, **PuA**, ToF, TGA]` — **PuA is index 5**.

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Citation

If you use this code in your research, please cite the nnU-Net paper and the imageCHD dataset:

- Isensee et al. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods.
- Xu et al. (2020). *imageCHD: A 3D Computed Tomography Image Dataset for Classification of Congenital Heart Disease.* MICCAI.
