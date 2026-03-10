#!/usr/bin/env python3
"""Pipeline diagnostic sanity checks.

Runs 5 checks on the atlas post-processing pipeline and logs everything
to a timestamped file for debugging.

Usage:
    python scripts/sanity_check.py \
        --pred-folder /path/to/predictions \
        --gt-folder /path/to/ground_truth_labels \
        --disease-map /path/to/disease_map.json \
        --max-cases 3

Output:
    sanity_check_YYYYMMDD_HHMMSS.log
"""

import argparse
import datetime
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import label as nd_label

# ── Ensure the package is importable ──────────────────────────────────────────
# Add repo root to path so chd_postprocessing is importable even without pip install
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from chd_postprocessing.atlas import AtlasLibrary, create_synthetic_atlas
from chd_postprocessing.atlas_pipeline import run_atlas_pipeline
from chd_postprocessing.anatomy_priors import correct_ao_pa_fragments
from chd_postprocessing.config import (
    DISEASE_ANATOMY_RULES, DISEASE_FLAGS, FOREGROUND_CLASSES,
    LABEL_NAMES, LABELS,
)
from chd_postprocessing.evaluate import dice_per_class
from chd_postprocessing.io_utils import (
    get_disease_vec, get_voxel_spacing, load_disease_map, load_nifti,
    resolve_case_id,
)
from chd_postprocessing.registration import register_atlas_to_pred

try:
    from chd_postprocessing.adjacency_correction import correct_by_adjacency
    HAS_ADJACENCY = True
except ImportError:
    HAS_ADJACENCY = False

try:
    from chd_postprocessing.registration import register_atlas_per_structure
    HAS_PER_STRUCTURE = True
except ImportError:
    HAS_PER_STRUCTURE = False


# ── Logging helper ────────────────────────────────────────────────────────────

class Logger:
    """Writes to both stdout and a log file simultaneously."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.file = open(log_path, "w")
        self.section_results = []  # (name, status, detail)

    def log(self, msg: str = ""):
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()

    def header(self, title: str):
        sep = "=" * 70
        self.log(f"\n{sep}")
        self.log(f"  {title}")
        self.log(sep)

    def subheader(self, title: str):
        self.log(f"\n--- {title} ---")

    def result(self, name: str, status: str, detail: str = ""):
        """Record a check result. status: PASS, FAIL, WARN, INFO"""
        self.section_results.append((name, status, detail))
        icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "INFO": "ℹ"}.get(status, "?")
        self.log(f"  [{icon} {status}] {name}")
        if detail:
            for line in detail.strip().split("\n"):
                self.log(f"         {line}")

    def summary(self):
        self.header("SUMMARY")
        passes = sum(1 for _, s, _ in self.section_results if s == "PASS")
        fails = sum(1 for _, s, _ in self.section_results if s == "FAIL")
        warns = sum(1 for _, s, _ in self.section_results if s == "WARN")
        self.log(f"  Total checks: {len(self.section_results)}")
        self.log(f"  PASS: {passes}  |  FAIL: {fails}  |  WARN: {warns}")
        self.log("")
        for name, status, detail in self.section_results:
            icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "INFO": "ℹ"}.get(status, "?")
            self.log(f"  [{icon}] {name}: {status}")
        self.log(f"\nLog saved to: {self.log_path}")

    def close(self):
        self.file.close()


# ── Helper functions ──────────────────────────────────────────────────────────

STRUCTURE_NAMES = {1: "LV", 2: "RV", 3: "LA", 4: "RA", 5: "Myo", 6: "AO", 7: "PA"}


def find_gt_for_case(case_id: str, gt_folder: Path) -> Path | None:
    """Try to find the GT file for a case ID, handling naming variants."""
    base = case_id.replace("_image", "")
    for variant in [case_id, base, base + "_image"]:
        for ext in [".nii.gz", ".nii"]:
            cand = gt_folder / f"{variant}{ext}"
            if cand.exists():
                return cand
    return None


def component_report(labels: np.ndarray, label_ids=None) -> str:
    """Return a multi-line string describing connected components per label."""
    if label_ids is None:
        label_ids = list(range(1, 8))
    lines = []
    for lbl in label_ids:
        mask = labels == lbl
        if not mask.any():
            lines.append(f"  {STRUCTURE_NAMES.get(lbl, lbl)} (label {lbl}): 0 voxels")
            continue
        labeled, n = nd_label(mask)
        sizes = sorted([int((labeled == c).sum()) for c in range(1, n + 1)], reverse=True)
        lines.append(
            f"  {STRUCTURE_NAMES.get(lbl, lbl)} (label {lbl}): "
            f"{int(mask.sum()):,} voxels, {n} component(s), "
            f"sizes={sizes[:5]}{'...' if len(sizes) > 5 else ''}"
        )
    return "\n".join(lines)


def dice_report(pred: np.ndarray, gt: np.ndarray, label_ids=None) -> dict:
    """Compute per-class Dice and return as dict."""
    if label_ids is None:
        label_ids = list(range(1, 8))
    return dice_per_class(pred, gt, label_ids)


# ── Sanity Checks ─────────────────────────────────────────────────────────────

def check_1_pipeline_execution(
    pred_path: Path, gt_path: Path, gt_folder: Path,
    disease_map_path: Path, log: Logger,
):
    """Check 1: Does run_atlas_pipeline actually execute and change labels?"""
    log.header("CHECK 1: Pipeline Execution")
    log.log(f"  Pred: {pred_path.name}")
    log.log(f"  GT:   {gt_path.name}")

    for mode in ["random_atlas", "disease_atlas", "disease_atlas_rules"]:
        log.subheader(f"Mode: {mode}")
        try:
            result = run_atlas_pipeline(
                pred_path=str(pred_path),
                gt_folder=str(gt_folder),
                disease_map_path=str(disease_map_path),
                gt_path=str(gt_path),
                mode=mode,
                seed=42,
            )
            log.log(f"  Atlas used: {result.atlas_case_id} ({result.atlas_disease_name})")
            log.log(f"  Was relabeled: {result.was_relabeled}")
            log.log(f"  Reassignment summary: {result.reassignment_summary}")

            if result.dice_before and result.dice_after:
                log.log(f"\n  Per-class Dice (before -> after):")
                for cls in result.dice_before:
                    b = result.dice_before[cls]
                    a = result.dice_after[cls]
                    d = a - b
                    marker = "▲" if d > 0.001 else "▼" if d < -0.001 else "="
                    log.log(f"    {cls:12s}: {b:.4f} -> {a:.4f}  (Δ={d:+.5f}) {marker}")

                valid = [
                    result.dice_after[c] - result.dice_before[c]
                    for c in result.dice_before
                    if not np.isnan(result.dice_before[c]) and not np.isnan(result.dice_after[c])
                ]
                if valid:
                    mean_delta = np.mean(valid)
                    log.log(f"    {'Mean Δ':12s}: {mean_delta:+.5f}")

                if result.was_relabeled:
                    log.result(f"Check1/{mode}", "PASS", "Pipeline made changes")
                else:
                    log.result(f"Check1/{mode}", "WARN", "Pipeline ran but made NO changes")
            else:
                log.result(f"Check1/{mode}", "WARN", "Dice not computed (GT issue?)")

            if result.adjacency_log:
                log.log(f"\n  Adjacency corrections: {len(result.adjacency_log)}")
                for entry in result.adjacency_log[:5]:
                    log.log(f"    {entry.get('original_label')} -> {entry.get('new_label')} "
                            f"({entry.get('component_size')} vx, {entry.get('reason')})")

            if result.boundary_log:
                total_br = sum(e.get("a_to_b", 0) + e.get("b_to_a", 0)
                               for e in result.boundary_log)
                log.log(f"\n  Boundary refinement voxels: {total_br}")

        except Exception as exc:
            log.result(f"Check1/{mode}", "FAIL", f"Exception: {exc}\n{traceback.format_exc()}")


def check_2_prediction_structure(
    pred_path: Path, gt_path: Path, log: Logger,
):
    """Check 2: What do the prediction and GT look like? How many components per label?"""
    log.header("CHECK 2: Prediction & GT Structure")

    pred = np.asarray(nib.load(str(pred_path)).dataobj).astype(np.int32)
    gt = np.asarray(nib.load(str(gt_path)).dataobj).astype(np.int32)

    log.log(f"  Pred shape: {pred.shape}, GT shape: {gt.shape}")
    log.log(f"  Pred unique labels: {sorted(np.unique(pred).tolist())}")
    log.log(f"  GT unique labels:   {sorted(np.unique(gt).tolist())}")

    log.subheader("Prediction components")
    log.log(component_report(pred))

    log.subheader("Ground truth components")
    log.log(component_report(gt))

    # Identify multi-component labels in prediction
    multi_comp_labels = []
    for lbl in range(1, 8):
        mask = pred == lbl
        if mask.any():
            _, n = nd_label(mask)
            if n > 1:
                multi_comp_labels.append((lbl, n))

    if multi_comp_labels:
        log.result("Check2/multi_component", "INFO",
                   f"Labels with >1 component: {[(STRUCTURE_NAMES[l], n) for l, n in multi_comp_labels]}\n"
                   "These fragments are candidates for component-level correction.")
    else:
        log.result("Check2/multi_component", "WARN",
                   "ALL labels have exactly 1 component in prediction.\n"
                   "Component-level correction (IoC, adjacency) cannot fix anything.\n"
                   "Errors must be at contiguous boundaries — need voxel-level refinement.")

    # Per-class Dice
    log.subheader("Per-class Dice (raw prediction vs GT)")
    dice = dice_report(pred, gt)
    for lbl in sorted(dice.keys()):
        name = STRUCTURE_NAMES.get(lbl, str(lbl))
        log.log(f"  {name:4s} (label {lbl}): {dice[lbl]:.4f}")

    # Whole-heart dice
    wh_pred = pred > 0
    wh_gt = gt > 0
    wh_inter = int((wh_pred & wh_gt).sum())
    wh_denom = int(wh_pred.sum()) + int(wh_gt.sum())
    wh_dice = 2 * wh_inter / wh_denom if wh_denom > 0 else 0
    log.log(f"  WH  (whole-heart): {wh_dice:.4f}")

    # Find where errors are: voxels where pred != gt but both are foreground
    both_fg = (pred > 0) & (gt > 0)
    mislabeled = both_fg & (pred != gt)
    n_mislabeled = int(mislabeled.sum())
    n_fg = int(both_fg.sum())
    if n_fg > 0:
        log.log(f"\n  Foreground voxels where pred != gt: {n_mislabeled:,} / {n_fg:,} "
                f"({100*n_mislabeled/n_fg:.1f}%)")

    # Show confusion: which labels get swapped most?
    if n_mislabeled > 0:
        log.subheader("Label confusion matrix (pred vs GT, foreground mismatches only)")
        confused = {}
        for lbl_pred in range(1, 8):
            for lbl_gt in range(1, 8):
                if lbl_pred == lbl_gt:
                    continue
                count = int(((pred == lbl_pred) & (gt == lbl_gt)).sum())
                if count > 0:
                    confused[(lbl_pred, lbl_gt)] = count

        for (lp, lg), count in sorted(confused.items(), key=lambda x: -x[1])[:10]:
            pname = STRUCTURE_NAMES.get(lp, str(lp))
            gname = STRUCTURE_NAMES.get(lg, str(lg))
            log.log(f"  Pred={pname:4s}  GT={gname:4s}  : {count:,} voxels")

        log.result("Check2/confusion", "INFO", "See confusion matrix above for main error patterns")


def check_3_registration_quality(
    pred_path: Path, gt_folder: Path, disease_map_path: Path, log: Logger,
):
    """Check 3: How well does the registered atlas overlap with the prediction?"""
    log.header("CHECK 3: Atlas Registration Quality")

    img = nib.load(str(pred_path))
    pred = np.asarray(img.dataobj).astype(np.int32)
    spacing = get_voxel_spacing(img.header)
    case_id = resolve_case_id(pred_path.name)

    dm = load_disease_map(str(disease_map_path))
    vec = get_disease_vec(dm, case_id) or [0] * 8
    disease_str = "+".join(DISEASE_FLAGS[i] for i, f in enumerate(vec) if f) or "Normal"
    log.log(f"  Case: {case_id}, Disease: {disease_str}")

    library = AtlasLibrary.load_all(str(gt_folder), disease_map_path=str(disease_map_path))
    rng = random.Random(42)

    base_id = case_id.replace("_image", "")
    exclude = [case_id, base_id, base_id + "_image"]

    for sel_mode, label in [("random", "Random atlas"), ("best_match", "Disease-matched atlas")]:
        log.subheader(label)
        entry = library.select_for_case(vec, rng, mode=sel_mode, exclude_case_ids=exclude)
        entry.load()
        log.log(f"  Selected: {entry.case_id} ({entry.disease_name}), "
                f"hamming={entry.hamming_distance(vec)}")

        # Global centroid registration
        reg = register_atlas_to_pred(entry.labels, pred, spacing)

        log.log(f"\n  Global centroid registration — IoC per label:")
        any_good = False
        for lbl in range(1, 8):
            pred_mask = pred == lbl
            atlas_mask = reg == lbl
            pred_size = int(pred_mask.sum())
            atlas_size = int(atlas_mask.sum())
            inter = int((pred_mask & atlas_mask).sum())
            ioc = inter / pred_size if pred_size > 0 else 0
            dice_val = 2 * inter / (pred_size + atlas_size) if (pred_size + atlas_size) > 0 else 0
            name = STRUCTURE_NAMES.get(lbl, str(lbl))
            quality = "GOOD" if ioc > 0.3 else "OK" if ioc > 0.1 else "POOR"
            log.log(f"    {name:4s}: pred={pred_size:>7,}  atlas_reg={atlas_size:>7,}  "
                    f"overlap={inter:>7,}  IoC={ioc:.3f}  Dice={dice_val:.3f}  [{quality}]")
            if ioc > 0.1:
                any_good = True

        if any_good:
            log.result(f"Check3/{sel_mode}/global", "PASS", "Some labels have IoC > 0.10")
        else:
            log.result(f"Check3/{sel_mode}/global", "FAIL",
                       "ALL labels have IoC < 0.10 with global centroid registration.\n"
                       "IoC-based atlas correction is effectively dead for this case.")

        # Per-structure registration (if available)
        if HAS_PER_STRUCTURE:
            try:
                per_struct = register_atlas_per_structure(entry.labels, pred, list(range(1, 8)))
                log.log(f"\n  Per-structure registration — IoC per label:")
                any_good_ps = False
                for lbl in range(1, 8):
                    pred_mask = pred == lbl
                    if lbl in per_struct:
                        atlas_mask = per_struct[lbl]
                        pred_size = int(pred_mask.sum())
                        atlas_size = int(atlas_mask.sum())
                        inter = int((pred_mask & atlas_mask).sum())
                        ioc = inter / pred_size if pred_size > 0 else 0
                        name = STRUCTURE_NAMES.get(lbl, str(lbl))
                        quality = "GOOD" if ioc > 0.3 else "OK" if ioc > 0.1 else "POOR"
                        log.log(f"    {name:4s}: IoC={ioc:.3f}  [{quality}]")
                        if ioc > 0.1:
                            any_good_ps = True

                if any_good_ps:
                    log.result(f"Check3/{sel_mode}/per_structure", "PASS",
                               "Per-structure improves alignment")
                else:
                    log.result(f"Check3/{sel_mode}/per_structure", "WARN",
                               "Per-structure still poor")
            except Exception as exc:
                log.result(f"Check3/{sel_mode}/per_structure", "WARN",
                           f"Per-structure registration failed: {exc}")


def check_4_anatomy_priors(
    pred_path: Path, gt_path: Path, disease_map_path: Path, log: Logger,
):
    """Check 4: Do anatomy priors (AO/PA ventricle-adjacency) improve anything?"""
    log.header("CHECK 4: Anatomy Priors (AO/PA correction)")

    img = nib.load(str(pred_path))
    pred = np.asarray(img.dataobj).astype(np.int32)
    gt = np.asarray(nib.load(str(gt_path)).dataobj).astype(np.int32)
    spacing = get_voxel_spacing(img.header)
    case_id = resolve_case_id(pred_path.name)

    dm = load_disease_map(str(disease_map_path))
    vec = get_disease_vec(dm, case_id) or [0] * 8
    disease_str = "+".join(DISEASE_FLAGS[i] for i, f in enumerate(vec) if f) or "Normal"

    log.log(f"  Case: {case_id}, Disease: {disease_str}")

    for flag_idx, is_active in enumerate(vec):
        if is_active and flag_idx in DISEASE_ANATOMY_RULES:
            rule = DISEASE_ANATOMY_RULES[flag_idx]
            log.log(f"  Active rule: {rule['name']} — {rule['notes']}")
            if rule.get("skip_ao_pa_correction"):
                log.log(f"    -> AO/PA correction will be SKIPPED")

    # Dice before
    dice_before = dice_report(pred, gt)

    # Fragment-level correction
    corrected_frag, frag_log = correct_ao_pa_fragments(pred, vec, spacing)
    n_frag_reassigned = len(frag_log.get("reassigned", []))
    log.log(f"\n  Fragment-level correction: {n_frag_reassigned} fragments reassigned")
    if frag_log.get("skipped_disease"):
        log.log(f"  SKIPPED due to disease flag")
    for entry in frag_log.get("reassigned", []):
        orig = STRUCTURE_NAMES.get(entry["original_label"], str(entry["original_label"]))
        new = STRUCTURE_NAMES.get(entry["assigned_label"], str(entry["assigned_label"]))
        log.log(f"    {orig} -> {new}  ({entry['size']} vx, "
                f"correct_score={entry['correct_score']}, wrong_score={entry['wrong_score']})")

    dice_after_frag = dice_report(corrected_frag, gt)

    log.subheader("Fragment-level AO/PA correction Dice change")
    for lbl in sorted(dice_before.keys()):
        b = dice_before[lbl]
        a = dice_after_frag[lbl]
        d = a - b
        marker = "▲" if d > 0.001 else "▼" if d < -0.001 else "="
        name = STRUCTURE_NAMES.get(lbl, str(lbl))
        log.log(f"  {name:4s}: {b:.4f} -> {a:.4f}  (Δ={d:+.5f}) {marker}")

    # Global swap test
    log.subheader("Global AO<->PA swap test (diagnostic only)")
    swapped = pred.copy()
    ao_mask = pred == 6
    pa_mask = pred == 7
    swapped[ao_mask] = 7
    swapped[pa_mask] = 6
    dice_swapped = dice_report(swapped, gt)

    ao_improve = dice_swapped[6] - dice_before[6]
    pa_improve = dice_swapped[7] - dice_before[7]
    log.log(f"  AO: {dice_before[6]:.4f} -> {dice_swapped[6]:.4f}  (Δ={ao_improve:+.5f})")
    log.log(f"  PA: {dice_before[7]:.4f} -> {dice_swapped[7]:.4f}  (Δ={pa_improve:+.5f})")

    if ao_improve > 0.01 and pa_improve > 0.01:
        log.result("Check4/global_swap", "FAIL",
                   "Global AO<->PA swap IMPROVES both labels!\n"
                   "The entire AO and PA are on the wrong side.\n"
                   "correct_ao_pa_labels should catch this — check if it's being called.")
    elif ao_improve > 0.01 or pa_improve > 0.01:
        log.result("Check4/global_swap", "WARN",
                   "Partial swap benefit — some fragments are swapped, not all.")
    else:
        log.result("Check4/global_swap", "PASS",
                   "Global swap does NOT help — AO/PA are mostly on the correct side.")

    if n_frag_reassigned > 0:
        valid = [dice_after_frag[l] - dice_before[l]
                 for l in dice_before
                 if not np.isnan(dice_before[l]) and not np.isnan(dice_after_frag[l])]
        mean_delta = np.mean(valid) if valid else 0.0
        if mean_delta > 0:
            log.result("Check4/fragment_correction", "PASS",
                       f"Fragment correction improved mean Dice by {mean_delta:+.5f}")
        else:
            log.result("Check4/fragment_correction", "WARN",
                       f"Fragment correction changed labels but mean Dice went {mean_delta:+.5f}")
    else:
        log.result("Check4/fragment_correction", "WARN",
                   "No fragments were reassigned by anatomy priors.\n"
                   "Either all AO/PA components have correct ventricle adjacency,\n"
                   "or errors are at contiguous boundaries (not separate fragments).")


def check_5_cached_results(out_root: Path, log: Logger):
    """Check 5: Are there old cached NIfTI files that would prevent recomputation?"""
    log.header("CHECK 5: Cached Output Files")

    if not out_root.exists():
        log.result("Check5/cache", "PASS",
                   f"Output root {out_root} does not exist — no cache issue.")
        return

    cached_files = list(out_root.rglob("*.nii.gz"))
    if cached_files:
        log.log(f"  Found {len(cached_files)} cached NIfTI files under {out_root}:")
        for f in sorted(cached_files)[:20]:
            rel = f.relative_to(out_root)
            import os
            mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f))
            log.log(f"    {rel}  (modified: {mtime.strftime('%Y-%m-%d %H:%M')})")
        if len(cached_files) > 20:
            log.log(f"    ... and {len(cached_files) - 20} more")

        log.result("Check5/cache", "FAIL",
                   f"{len(cached_files)} cached files found!\n"
                   "The notebook skips recomputation when output files exist.\n"
                   "DELETE these files and rerun the notebook to use the updated pipeline.\n"
                   f"Run: rm -rf {out_root}/*")
    else:
        log.result("Check5/cache", "PASS", "No cached NIfTI files found.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline diagnostic sanity checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Single-case quick check:
  python scripts/sanity_check.py \\
      --pred-folder /path/to/preds \\
      --gt-folder   /path/to/gt \\
      --disease-map /path/to/disease_map.json \\
      --case-id ct_1004 \\
      --checks 1,2,4

NOTE: Check 1 (pipeline execution) always recomputes — it never reads cached files.
""",
    )
    parser.add_argument("--pred-folder", type=str, required=True,
                        help="Folder containing nnU-Net prediction NIfTI files")
    parser.add_argument("--gt-folder", type=str, required=True,
                        help="Folder containing ground-truth label NIfTI files (atlas library)")
    parser.add_argument("--disease-map", type=str, required=True,
                        help="Path to disease_map.json")
    parser.add_argument("--out-root", type=str, default=None,
                        help="Path to atlas_postprocessed output root (for cache check, Check 5)")
    parser.add_argument("--case-id", type=str, default=None,
                        help="Run on this specific case only (e.g. ct_1004). "
                             "Overrides --max-cases.")
    parser.add_argument("--max-cases", type=int, default=3,
                        help="Maximum number of cases to check (default: 3). "
                             "Ignored when --case-id is set.")
    parser.add_argument("--checks", type=str, default="1,2,3,4,5",
                        help="Comma-separated list of checks to run (default: 1,2,3,4,5). "
                             "E.g. --checks 1,2,4 skips registration (3) and cache (5).")
    parser.add_argument("--log-dir", type=str, default=".",
                        help="Directory to write the log file (default: current dir)")
    args = parser.parse_args()

    # Parse which checks to run
    try:
        enabled_checks = {int(c.strip()) for c in args.checks.split(",")}
    except ValueError:
        print(f"ERROR: --checks must be comma-separated integers, got: {args.checks!r}")
        sys.exit(1)

    # Setup
    pred_folder = Path(args.pred_folder)
    gt_folder = Path(args.gt_folder)
    disease_map_path = Path(args.disease_map)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"sanity_check_{timestamp}.log"
    log = Logger(log_path)

    log.header("PIPELINE DIAGNOSTIC SANITY CHECKS")
    log.log(f"  Timestamp:    {timestamp}")
    log.log(f"  Pred folder:  {pred_folder}")
    log.log(f"  GT folder:    {gt_folder}")
    log.log(f"  Disease map:  {disease_map_path}")
    log.log(f"  Case filter:  {args.case_id or '(all, up to max-cases)'}")
    log.log(f"  Max cases:    {args.max_cases if not args.case_id else 'N/A (--case-id set)'}")
    log.log(f"  Checks:       {sorted(enabled_checks)}")
    log.log(f"  Log file:     {log_path}")
    log.log(f"  NOTE: Check 1 always recomputes (no output_path → no caching).")

    # Validate paths
    for label, path in [
        ("Pred folder", pred_folder),
        ("GT folder", gt_folder),
        ("Disease map", disease_map_path),
    ]:
        if not path.exists():
            log.log(f"\nERROR: {label} does not exist: {path}")
            log.close()
            return

    # Find prediction files
    pred_files = sorted(pred_folder.glob("*.nii.gz"))
    if not pred_files:
        log.log(f"\nERROR: No .nii.gz files in {pred_folder}")
        log.close()
        return

    log.log(f"\n  Found {len(pred_files)} prediction files")

    # Filter to a single case if --case-id was given
    if args.case_id:
        target = args.case_id.replace(".nii.gz", "").replace(".nii", "")
        pred_files = [
            p for p in pred_files
            if resolve_case_id(p.name) == target or p.stem.replace(".nii", "") == target
        ]
        if not pred_files:
            log.log(f"\nERROR: No prediction file matched --case-id={args.case_id!r}")
            log.log(f"  Available: {[resolve_case_id(p.name) for p in sorted(pred_folder.glob('*.nii.gz'))[:10]]}")
            log.close()
            return
        log.log(f"  Filtered to 1 case: {pred_files[0].name}")

    # Check 5 first (cache check) — doesn't need individual cases
    if 5 in enabled_checks and args.out_root:
        check_5_cached_results(Path(args.out_root), log)

    # Process cases
    max_cases = 1 if args.case_id else args.max_cases
    cases_processed = 0
    for pred_path in pred_files:
        if cases_processed >= max_cases:
            break

        case_id = resolve_case_id(pred_path.name)
        gt_path = find_gt_for_case(case_id, gt_folder)
        if gt_path is None:
            log.log(f"\n  Skipping {case_id}: no GT file found in {gt_folder}")
            continue

        log.header(f"CASE: {case_id}")
        log.log(f"  Pred: {pred_path}")
        log.log(f"  GT:   {gt_path}")

        if 2 in enabled_checks:
            try:
                check_2_prediction_structure(pred_path, gt_path, log)
            except Exception as exc:
                log.result(f"Check2/{case_id}", "FAIL", f"Exception: {exc}\n{traceback.format_exc()}")

        if 3 in enabled_checks:
            try:
                check_3_registration_quality(pred_path, gt_folder, disease_map_path, log)
            except Exception as exc:
                log.result(f"Check3/{case_id}", "FAIL", f"Exception: {exc}\n{traceback.format_exc()}")

        if 4 in enabled_checks:
            try:
                check_4_anatomy_priors(pred_path, gt_path, disease_map_path, log)
            except Exception as exc:
                log.result(f"Check4/{case_id}", "FAIL", f"Exception: {exc}\n{traceback.format_exc()}")

        if 1 in enabled_checks:
            try:
                check_1_pipeline_execution(pred_path, gt_path, gt_folder, disease_map_path, log)
            except Exception as exc:
                log.result(f"Check1/{case_id}", "FAIL", f"Exception: {exc}\n{traceback.format_exc()}")

        cases_processed += 1

    if cases_processed == 0:
        log.log("\nWARNING: No cases were processed. Check that GT files exist for prediction cases.")

    # Summary
    log.summary()
    log.close()
    print(f"\nDone. Full log: {log_path}")


if __name__ == "__main__":
    main()
