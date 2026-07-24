#!/usr/bin/env python3
"""Plot Totten no-sliding vs max-sliding η from exported eta_maps.npz.

Run after Archive/plot_totten_sliding_comparison.py has written eta_maps.npz
(or after rsync of that file from the cluster).

  python scripts/plot_totten_eta_no_vs_max.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "outputs/figures/vi/totten_sliding_comparison/eta_maps.npz"
DEFAULT_OUT = ROOT / "outputs/figures/vi/totten_sliding_comparison"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main():
    args = parse_args()
    if not args.npz.is_file():
        raise FileNotFoundError(
            f"Missing {args.npz}. On the cluster run:\n"
            "  cd Archive && python plot_totten_sliding_comparison.py --checkpoint latest\n"
            "then rsync outputs/figures/vi/totten_sliding_comparison/eta_maps.npz here."
        )

    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm

    try:
        import seaborn as sns

        eta_cmap = LinearSegmentedColormap.from_list(
            "Blues_d", sns.color_palette("Blues_d", n_colors=9)
        ).reversed()
    except ImportError:
        # Matplotlib fallback approximating seaborn Blues_d (dark → light blue).
        eta_cmap = LinearSegmentedColormap.from_list(
            "Blues_d",
            [
                "#08306B",
                "#08519C",
                "#2171B5",
                "#4292C6",
                "#6BAED6",
                "#9ECAE1",
                "#C6DBEF",
                "#DEEBF7",
            ],
        )

    with np.load(args.npz) as data:
        x_km = data["x_km"]
        y_km = data["y_km"]
        eta_no = data["eta_no_sliding"]
        eta_max = data["eta_max_sliding"]
        delta = data["delta_eta_max_minus_no"]
        c_no = float(data["friction_C_no"]) if "friction_C_no" in data.files else 100.0
        c_max = float(data["friction_C_max"]) if "friction_C_max" in data.files else 0.001

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pos = np.concatenate(
        [
            eta_no[np.isfinite(eta_no) & (eta_no > 0)],
            eta_max[np.isfinite(eta_max) & (eta_max > 0)],
        ]
    )
    eta_norm = LogNorm(
        vmin=max(float(np.percentile(pos, 5)), 1e-3),
        vmax=float(np.percentile(pos, 95)),
    )
    d_lim = max(
        abs(float(np.nanpercentile(delta, 2))),
        abs(float(np.nanpercentile(delta, 98))),
        1e-4,
    )
    d_norm = TwoSlopeNorm(vcenter=0.0, vmin=-d_lim, vmax=d_lim)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), constrained_layout=True)
    panels = (
        (axes[0], eta_no, f"no sliding η (C={c_no:g})", eta_cmap, eta_norm, "η (MPa·yr)"),
        (axes[1], eta_max, f"max sliding η (C={c_max:g})", eta_cmap, eta_norm, "η (MPa·yr)"),
        (axes[2], delta, "Δη = max − no sliding", "RdBu_r", d_norm, "η (MPa·yr)"),
    )
    for ax, field, title, cmap, norm, unit in panels:
        image = ax.pcolormesh(x_km, y_km, field, shading="auto", cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label=unit)
    fig.suptitle("Totten VI viscosity: basal-sliding end-members", fontsize=13)
    out = args.output_dir / "eta_viscosity_no_vs_max.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
