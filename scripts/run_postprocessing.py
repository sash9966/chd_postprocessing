#!/usr/bin/env python3
"""CLI entry point for CHD segmentation post-processing.

Examples
--------
Basic (no evaluation)::

    python scripts/run_postprocessing.py \\
        --input_folder  /path/to/nnunet_predictions/ \\
        --output_folder /path/to/corrected/ \\
        --disease_map   disease_map.json

With evaluation against ground truth::

    python scripts/run_postprocessing.py \\
        --input_folder  /path/to/nnunet_predictions/ \\
        --output_folder /path/to/corrected/ \\
        --disease_map   disease_map.json \\
        --gt_folder     /path/to/ground_truth/ \\
        --evaluate

CC cleanup only (no adjacency correction)::

    python scripts/run_postprocessing.py \\
        --input_folder /path/to/preds/ \\
        --output_folder /path/to/out/ \\
        --steps cc_cleanup
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as ``python scripts/run_postprocessing.py`` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chd_postprocessing.config import FOREGROUND_CLASSES, LABEL_NAMES
from chd_postprocessing.evaluate import evaluate_folder, summarise
from chd_postprocessing.pipeline import AVAILABLE_STEPS, run_folder_pipeline


def _print_summary_table(results: list) -> None:
    """Print a concise summary of pipeline results."""
    n_ok      = sum(1 for r in results if r.get("status") == "ok")
    n_err     = sum(1 for r in results if r.get("status") == "error")
    n_swapped = sum(
        1 for r in results
        if r.get("correction") and r["correction"].get("was_swapped")
    )
    n_review  = sum(
        1 for r in results
        if r.get("correction") and r["correction"].get("needs_manual_review")
    )
    n_pua     = sum(
        1 for r in results
        if r.get("correction") and "PuA" in (r["correction"].get("skipped_reason") or "")
    )

    print("\n" + "=" * 60)
    print("Post-processing summary")
    print("=" * 60)
    print(f"  Cases processed:        {n_ok}")
    print(f"  Errors:                 {n_err}")
    print(f"  AO/PA labels swapped:   {n_swapped}")
    print(f"  PuA (skipped):          {n_pua}")
    print(f"  Flagged for review:     {n_review}")
    print("=" * 60)

    if n_review > 0:
        print("\nCases flagged for manual review:")
        for r in results:
            if r.get("correction") and r["correction"].get("needs_manual_review"):
                conf = r["correction"].get("confidence_score", float("nan"))
                reason = r["correction"].get("skipped_reason", "")
                print(f"  {r['case_id']:30s}  confidence={conf:.3f}  {reason}")


def _print_dice_comparison(before_df: pd.DataFrame, after_df: pd.DataFrame) -> None:
    """Print per-class mean Dice before and after, highlighting AO and PA."""
    col_order = [LABEL_NAMES[c] for c in FOREGROUND_CLASSES if LABEL_NAMES[c] in before_df.columns]
    if "mean_fg" in before_df.columns:
        col_order.append("mean_fg")

    before_mean = before_df[col_order].mean()
    after_mean  = after_df[col_order].mean()
    delta       = after_mean - before_mean

    print("\n" + "=" * 65)
    print(f"{'Class':<10} {'Before':>8} {'After':>8} {'Δ':>8}")
    print("-" * 65)
    for col in col_order:
        b = before_mean[col]
        a = after_mean[col]
        d = delta[col]
        marker = " ◄" if col in ("AO", "PA") else ""
        print(f"{col:<10} {b:>8.4f} {a:>8.4f} {d:>+8.4f}{marker}")
    print("=" * 65)
    print("◄ = primary target classes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process nnU-Net CHD segmentations (AO/PA label correction).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_folder",  required=True,  help="Folder of input prediction NIfTI files")
    parser.add_argument("--output_folder", required=True,  help="Destination folder for corrected files")
    parser.add_argument("--disease_map",   default=None,   help="Path to disease_map.json")
    parser.add_argument("--gt_folder",     default=None,   help="Ground-truth folder (enables evaluation)")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Compute Dice before and after post-processing (requires --gt_folder)",
    )
    parser.add_argument(
        "--steps", nargs="+", default=None,
        choices=AVAILABLE_STEPS,
        help="Pipeline steps to run (default: all)",
    )
    parser.add_argument(
        "--dilation_radius_mm", type=float, default=3.0,
        help="Adjacency dilation radius in mm",
    )
    parser.add_argument(
        "--file_pattern", default="*.nii.gz",
        help="Glob pattern for input files",
    )
    parser.add_argument(
        "--save_report", default=None,
        help="Save per-case report as a JSON file",
    )
    args = parser.parse_args()

    if args.evaluate and args.gt_folder is None:
        parser.error("--evaluate requires --gt_folder")

    # -----------------------------------------------------------------------
    # Optional: evaluate BEFORE post-processing
    # -----------------------------------------------------------------------
    before_df = None
    if args.evaluate:
        print(f"Evaluating input predictions vs ground truth …")
        before_df = evaluate_folder(args.input_folder, args.gt_folder, file_pattern=args.file_pattern)
        print(f"  Evaluated {len(before_df)} cases.")

    # -----------------------------------------------------------------------
    # Run pipeline
    # -----------------------------------------------------------------------
    print(f"\nRunning pipeline on {args.input_folder} …")
    results = run_folder_pipeline(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        disease_map_path=args.disease_map,
        steps=args.steps,
        dilation_radius_mm=args.dilation_radius_mm,
        file_pattern=args.file_pattern,
    )

    _print_summary_table(results)

    # -----------------------------------------------------------------------
    # Optional: evaluate AFTER post-processing and compare
    # -----------------------------------------------------------------------
    if args.evaluate and before_df is not None:
        print(f"\nEvaluating corrected predictions …")
        after_df = evaluate_folder(args.output_folder, args.gt_folder, file_pattern=args.file_pattern)
        _print_dice_comparison(before_df, after_df)

    # -----------------------------------------------------------------------
    # Optional: save per-case report
    # -----------------------------------------------------------------------
    if args.save_report:
        # Strip numpy arrays from the report before serialising
        serialisable = []
        for r in results:
            r2 = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
            serialisable.append(r2)
        Path(args.save_report).write_text(json.dumps(serialisable, indent=2))
        print(f"\nReport saved to {args.save_report}")

    print("\nDone.")


if __name__ == "__main__":
    main()
