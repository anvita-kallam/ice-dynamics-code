#!/usr/bin/env python3
"""Regenerate Totten no vs max sliding loss / diagnostics plots from metrics CSVs.

Pull fresh CSVs first (unit-corrected Jul-28 runs):

  rsync -avz login.ds:~/ice-dynamics/Archive/logs/metrics_vi_only_log_vi_only_totten_{no,max}_sliding.csv \\
    Archive/logs/

  python scripts/plot_totten_loss_curves.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Archive"
DEFAULT_OUT = ROOT / "outputs/figures/vi/totten_sliding_comparison/loss_curves"

CASES = (
    ("no_sliding", ARCHIVE / "logs/metrics_vi_only_log_vi_only_totten_no_sliding.csv"),
    ("max_sliding", ARCHIVE / "logs/metrics_vi_only_log_vi_only_totten_max_sliding.csv"),
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(ARCHIVE))
    from training_metrics import load_vi_only_metrics, plot_vi_only_metrics

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = {}

    for case, csv_path in CASES:
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Missing {csv_path}\n"
                "Pull from cluster:\n"
                "  rsync -avz login.ds:~/ice-dynamics/Archive/logs/"
                "metrics_vi_only_log_vi_only_totten_{no,max}_sliding.csv Archive/logs/"
            )
        mtime = csv_path.stat().st_mtime
        metrics = load_vi_only_metrics(csv_path, None)
        plot_dir = args.output_dir / case
        saved = plot_vi_only_metrics(metrics, plot_dir)
        loaded[case] = metrics
        print(f"{case}: epochs {int(metrics['epoch'][0])}..{int(metrics['epoch'][-1])} "
              f"n={len(metrics['epoch'])}  mtime={mtime:.0f}")
        print(f"  wrote {[p.name for p in saved]}")

    # Overlay of key losses
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for case, color in (("no_sliding", "tab:blue"), ("max_sliding", "tab:orange")):
        df = pd.read_csv(dict(CASES)[case])
        ep = df["epoch"]
        axes[0, 0].plot(ep, df["train_total"], color=color, label=f"{case} train", alpha=0.9)
        if "test_total" in df:
            axes[0, 0].plot(ep, df["test_total"], color=color, ls="--", alpha=0.7,
                            label=f"{case} test")
        axes[0, 1].plot(ep, df["train_phys"], color=color, label=f"{case} train", alpha=0.9)
        if "test_phys" in df:
            axes[0, 1].plot(ep, df["test_phys"], color=color, ls="--", alpha=0.7,
                            label=f"{case} test")
        axes[1, 0].plot(ep, df["train_data"], color=color, label=f"{case} train", alpha=0.9)
        if "test_data" in df:
            axes[1, 0].plot(ep, df["test_data"], color=color, ls="--", alpha=0.7,
                            label=f"{case} test")
        axes[1, 1].plot(ep, df["train_kl"], color=color, label=f"{case} train", alpha=0.9)
        if "test_kl" in df:
            axes[1, 1].plot(ep, df["test_kl"], color=color, ls="--", alpha=0.7,
                            label=f"{case} test")

    axes[0, 0].set_title("Total ELBO")
    axes[0, 1].set_title("Physics NLL")
    axes[1, 0].set_title("Data NLL")
    axes[1, 1].set_title("KL (+ η soft prior)")
    # Total/physics cross ~0 (early ~1e4 → late ~−10); use symlog.
    # Data/KL stay positive → plain log.
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_yscale("symlog", linthresh=1.0)
    for ax in (axes[1, 0], axes[1, 1]):
        ax.set_yscale("log")
    for ax in axes.ravel():
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Totten VI — no_sliding (C=100) vs max_sliding (C=0.001)\n"
        "unit-corrected η bounds",
        fontsize=12,
    )
    overlay = args.output_dir / "recommended_losses_overlay.png"
    fig.savefig(overlay, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {overlay}")

    # Collapse overlay if columns present
    if any(
        np.any(np.isfinite(loaded[c].get("eta_frac_at_floor", np.array([np.nan]))))
        for c in loaded
    ):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), constrained_layout=True)
        for case, color in (("no_sliding", "tab:blue"), ("max_sliding", "tab:orange")):
            m = loaded[case]
            ep = m["epoch"]
            for ax, key, title, ylabel in (
                (axes[0], "eta_frac_at_floor", "Fraction at η_min", "fraction"),
                (axes[1], "eta_pred_mean", "Mean η", "MPa·yr"),
                (axes[2], "eta_log_shift", "η_log_shift", "log units"),
            ):
                vals = m.get(key)
                if vals is None:
                    continue
                mask = np.isfinite(vals) & np.isfinite(ep)
                if mask.any():
                    ax.plot(ep[mask], vals[mask], color=color, label=case)
                ax.set_title(title)
                ax.set_xlabel("epoch")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
        fig.suptitle("Totten VI — viscosity collapse diagnostics", fontsize=12)
        collapse = args.output_dir / "eta_collapse_overlay.png"
        fig.savefig(collapse, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {collapse}")
    else:
        print("note: no eta_frac_at_floor columns — CSV may be from pre-instrumentation run")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
