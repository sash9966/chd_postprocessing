#!/usr/bin/env python3
"""Compare per-class Dice scores from two prediction folders against ground truth.

Prints a side-by-side table and runs a paired Wilcoxon signed-rank test per class.

Example
-------
::

    python scripts/evaluate_before_after.py \\
        --before  /path/to/original_predictions/ \\
        --after   /path/to/corrected_predictions/ \\
        --gt      /path/to/ground_truth/ \\
        --output  results_comparison.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chd_postprocessing.config import FOREGROUND_CLASSES, LABEL_NAMES
from chd_postprocessing.evaluate import evaluate_folder, summarise


def wilcoxon_pvalue(before: pd.Series, after: pd.Series) -> float:
    """Paired Wilcoxon signed-rank test; returns p-value (nan if not enough data)."""
    diff = after.dropna() - before.dropna()
    diff = diff.dropna()
    if len(diff) < 4 or diff.abs().sum() == 0:
        return float("nan")
    try:
        result = stats.wilcoxon(diff, alternative="two-sided")
        return float(result.pvalue)
    except Exception:
        return float("nan")


def _stars(p: float) -> str:
    if np.isnan(p):
        return "  "
    if p < 0.001:
        return "***"
    if p < 0.01:
        return " **"
    if p < 0.05:
        return "  *"
    return "   "


def compare(
    before_folder: str | Path,
    after_folder: str | Path,
    gt_folder: str | Path,
    file_pattern: str = "*.nii.gz",
) -> pd.DataFrame:
    """Evaluate both folders and return a comparison DataFrame.

    Returns a DataFrame indexed by class name with columns:
    ``before_mean``, ``before_std``, ``after_mean``, ``after_std``,
    ``delta_mean``, ``p_value``.
    """
    print(f"Evaluating BEFORE ({before_folder}) …")
    before_df = evaluate_folder(before_folder, gt_folder, file_pattern=file_pattern)
    print(f"  {len(before_df)} cases evaluated.")

    print(f"Evaluating AFTER  ({after_folder}) …")
    after_df = evaluate_folder(after_folder, gt_folder, file_pattern=file_pattern)
    print(f"  {len(after_df)} cases evaluated.")

    # Align on common cases
    common = before_df.index.intersection(after_df.index)
    if len(common) == 0:
        raise ValueError("No common cases found between before and after folders.")
    before_df = before_df.loc[common]
    after_df  = after_df.loc[common]

    col_order = [LABEL_NAMES[c] for c in FOREGROUND_CLASSES if LABEL_NAMES[c] in before_df.columns]
    if "mean_fg" in before_df.columns:
        col_order.append("mean_fg")

    rows = []
    for col in col_order:
        b = before_df[col].dropna()
        a = after_df[col].dropna()
        pval = wilcoxon_pvalue(before_df[col], after_df[col])
        rows.append({
            "class":        col,
            "before_mean":  b.mean(),
            "before_std":   b.std(),
            "after_mean":   a.mean(),
            "after_std":    a.std(),
            "delta_mean":   a.mean() - b.mean(),
            "p_value":      pval,
        })

    return pd.DataFrame(rows).set_index("class")


def _print_table(comp: pd.DataFrame) -> None:
    header = (
        f"{'Class':<10} "
        f"{'Before':>9} "
        f"{'After':>9} "
        f"{'Δ':>8} "
        f"{'p-value':>9} "
        f"{'sig':>4}"
    )
    print("\n" + "=" * len(header))
    print("Per-class Dice comparison (mean ± std)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for cls, row in comp.iterrows():
        marker = " ◄" if cls in ("AO", "PA") else "  "
        sig = _stars(row["p_value"])
        pv_str = f"{row['p_value']:.4f}" if not np.isnan(row["p_value"]) else "  n/a "
        print(
            f"{cls:<10} "
            f"{row['before_mean']:>6.4f}±{row['before_std']:.3f} "
            f"{row['after_mean']:>6.4f}±{row['after_std']:.3f} "
            f"{row['delta_mean']:>+8.4f} "
            f"{pv_str:>9} "
            f"{sig:>4}"
            f"{marker}"
        )
    print("=" * len(header))
    print("* p<0.05  ** p<0.01  *** p<0.001  (paired Wilcoxon, two-sided)")
    print("◄ = primary target classes")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare per-class Dice before and after post-processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--before",  required=True, help="Folder of original nnU-Net predictions")
    parser.add_argument("--after",   required=True, help="Folder of post-processed predictions")
    parser.add_argument("--gt",      required=True, help="Ground-truth folder")
    parser.add_argument("--output",  default=None,  help="Save comparison table as CSV")
    parser.add_argument("--file_pattern", default="*.nii.gz")
    args = parser.parse_args()

    comp = compare(args.before, args.after, args.gt, file_pattern=args.file_pattern)
    _print_table(comp)

    if args.output:
        comp.to_csv(args.output)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
