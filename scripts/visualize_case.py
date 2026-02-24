#!/usr/bin/env python3
"""Quick 2D slice visualisation of a segmentation case.

Shows three orthogonal mid-planes (axial / sagittal / coronal).
Optionally shows before and after post-processing side by side.

Example
-------
::

    # Single prediction
    python scripts/visualize_case.py --pred ct_1001_image.nii.gz

    # Before vs after comparison
    python scripts/visualize_case.py \\
        --pred   /path/to/original/ct_1001_image.nii.gz \\
        --corrected /path/to/corrected/ct_1001_image.nii.gz \\
        --output ct_1001_comparison.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chd_postprocessing.config import LABEL_NAMES, LABELS
from chd_postprocessing.io_utils import load_nifti

# ---------------------------------------------------------------------------
# Colour map: one distinct colour per label
# ---------------------------------------------------------------------------
_LABEL_COLORS = {
    0: (0.0,  0.0,  0.0),   # background — black
    1: (0.85, 0.2,  0.2),   # LV         — red
    2: (0.2,  0.4,  0.85),  # RV         — blue
    3: (1.0,  0.55, 0.0),   # LA         — orange
    4: (0.0,  0.8,  0.8),   # RA         — cyan
    5: (0.9,  0.85, 0.1),   # Myo        — yellow
    6: (0.1,  0.75, 0.2),   # AO         — green
    7: (0.8,  0.1,  0.8),   # PA         — magenta
}

_N_LABELS = 8
_CMAP_COLORS = [_LABEL_COLORS[i] for i in range(_N_LABELS)]
_CMAP = mcolors.ListedColormap(_CMAP_COLORS, name="chd_labels")
_NORM = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, _N_LABELS), ncolors=_N_LABELS)


def _legend_handles():
    return [
        mpatches.Patch(color=_LABEL_COLORS[i], label=LABEL_NAMES.get(i, str(i)))
        for i in range(_N_LABELS)
        if i != 0  # skip background
    ]


def _mid(arr: np.ndarray, axis: int) -> int:
    return arr.shape[axis] // 2


def _slices(vol: np.ndarray):
    """Return mid-plane slices: axial (z), sagittal (x), coronal (y)."""
    return (
        vol[:, :, _mid(vol, 2)],    # axial    (H × W)
        vol[_mid(vol, 0), :, :],    # sagittal (W × D)
        vol[:, _mid(vol, 1), :],    # coronal  (H × D)
    )


def visualize_case(
    pred_path: str | Path,
    corrected_path: str | Path | None = None,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> None:
    """Render orthogonal mid-plane slices of one or two segmentations.

    Parameters
    ----------
    pred_path : path to the (original) prediction NIfTI
    corrected_path : optional path to the post-processed NIfTI.
                     When provided, adds a second row showing the corrected output.
    output_path : save figure here (PNG/PDF).  If ``None``, displays interactively.
    title : figure suptitle (defaults to the case filename)
    """
    pred_data, _, _ = load_nifti(pred_path)
    pred_data = pred_data.astype(np.int32)

    volumes   = [pred_data]
    row_labels = ["Original"]

    if corrected_path is not None:
        corr_data, _, _ = load_nifti(corrected_path)
        volumes.append(corr_data.astype(np.int32))
        row_labels.append("Post-processed")

    n_rows = len(volumes)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(12, 4 * n_rows),
        squeeze=False,
    )

    view_names = ["Axial (z)", "Sagittal (x)", "Coronal (y)"]

    for row_idx, (vol, row_lbl) in enumerate(zip(volumes, row_labels)):
        slices = _slices(vol)
        for col_idx, (slc, vname) in enumerate(zip(slices, view_names)):
            ax = axes[row_idx][col_idx]
            im = ax.imshow(
                slc.T,                # transpose so axes look anatomically natural
                cmap=_CMAP,
                norm=_NORM,
                interpolation="nearest",
                origin="lower",
            )
            ax.set_title(f"{row_lbl} — {vname}", fontsize=9)
            ax.axis("off")

    # Single legend on the right
    fig.legend(
        handles=_legend_handles(),
        loc="center right",
        fontsize=8,
        title="Labels",
        bbox_to_anchor=(1.0, 0.5),
    )

    suptitle = title or Path(pred_path).name
    fig.suptitle(suptitle, fontsize=11, y=1.01)
    plt.tight_layout()

    if output_path:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualise CHD segmentation slices (before/after comparison).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pred",      required=True, help="Input (original) prediction NIfTI")
    parser.add_argument("--corrected", default=None,  help="Post-processed prediction NIfTI (optional)")
    parser.add_argument("--output",    default=None,  help="Output image path (PNG/PDF)")
    parser.add_argument("--title",     default=None,  help="Figure title")
    args = parser.parse_args()

    visualize_case(
        pred_path=args.pred,
        corrected_path=args.corrected,
        output_path=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
