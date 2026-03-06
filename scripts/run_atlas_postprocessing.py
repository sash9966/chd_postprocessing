"""Command-line interface for atlas-based CHD post-processing.

Examples
--------
Single file, baseline::

    python run_atlas_postprocessing.py \\
        --pred pred.nii.gz \\
        --gt-folder labelsTr/ \\
        --output corrected.nii.gz \\
        --mode baseline

Whole folder, disease-specific with evaluation::

    python run_atlas_postprocessing.py \\
        --pred preds/ \\
        --gt-folder labelsTr/ \\
        --output corrected/ \\
        --mode disease_specific \\
        --disease-map disease_map.json \\
        --gt-eval labelsTsValidation/ \\
        --results results.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Atlas-based post-processing for CHD nnU-Net segmentations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--pred",        required=True,
                   help="Prediction NIfTI file OR folder of NIfTI files.")
    p.add_argument("--gt-folder",   required=True,
                   help="Folder containing training GT label NIfTI files (atlas library).")
    p.add_argument("--output",      required=True,
                   help="Output NIfTI file (single) or folder (batch).")
    p.add_argument("--disease-map", default=None,
                   help="Path to disease_map.json.")
    p.add_argument("--gt-eval",     default=None,
                   help="GT folder for before/after Dice evaluation (batch mode).")
    p.add_argument("--results",     default=None,
                   help="CSV path to save per-case results (batch mode).")

    # Pipeline settings
    p.add_argument("--mode", choices=["baseline", "disease_specific"],
                   default="baseline",
                   help="Atlas selection mode.")
    p.add_argument("--registration-mode", choices=["centroid", "pca"],
                   default="centroid",
                   help="Registration strategy.")
    p.add_argument("--seed",        type=int, default=42,
                   help="Random seed.")
    p.add_argument("--min-overlap", type=float, default=0.01,
                   help="Minimum Dice for a label to participate in reassignment.")
    p.add_argument("--no-morphology", action="store_true",
                   help="Disable morphological closing / hole-filling.")

    return p


def main() -> None:
    args = _build_parser().parse_args()

    from chd_postprocessing.atlas_pipeline import (
        run_atlas_pipeline,
        run_atlas_folder_pipeline,
    )

    pred_path = Path(args.pred)
    is_folder = pred_path.is_dir()

    common_kwargs = dict(
        gt_folder=args.gt_folder,
        disease_map_path=args.disease_map,
        mode=args.mode,
        seed=args.seed,
        registration_mode=args.registration_mode,
        min_overlap=args.min_overlap,
        do_morphological_cleanup=not args.no_morphology,
    )

    if is_folder:
        print(f"Batch mode: {pred_path}  →  {args.output}")
        df = run_atlas_folder_pipeline(
            pred_folder=pred_path,
            output_folder=args.output,
            gt_folder_eval=args.gt_eval,
            **common_kwargs,
        )
        if args.results:
            df.to_csv(args.results, index=False)
            print(f"Results saved to {args.results}")
        else:
            print(df.to_string())
    else:
        print(f"Single file: {pred_path}  →  {args.output}")
        result = run_atlas_pipeline(
            pred_path=pred_path,
            output_path=args.output,
            **common_kwargs,
        )
        print(f"Atlas used:  {result.atlas_case_id} ({result.atlas_disease_name})")
        print(f"Relabeled:   {result.was_relabeled}")
        print(result.reassignment_summary)
        if result.dice_before:
            print("\nDice before:", {k: f"{v:.4f}" for k, v in result.dice_before.items()})
        if result.dice_after:
            print("Dice after: ", {k: f"{v:.4f}" for k, v in result.dice_after.items()})


if __name__ == "__main__":
    main()
