#!/usr/bin/env python3
"""Offline Exp 2/6: grounded vs full-domain η correlation from eta_maps.npz.

No torch required. Uses existing Totten comparison NPZ + geometry for grounding.

  python scripts/totten_c_sensitivity_baseline_from_npz.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "outputs/figures/vi/totten_sliding_comparison/eta_maps.npz"
DATA = ROOT / "data/real/totten/totten_archive_vi_2022.npz"
OUT = ROOT / "outputs/figures/vi/totten_c_sensitivity/baseline_unit_corrected"


def grounded_mask(s, h, year=3600 * 24 * 365.25):
    g = 9.81 * year ** 2
    rho_ice = 917.0 / year ** 2 * 1.0e-6
    rho_water = 1024.0 / year ** 2 * 1.0e-6
    water_depth = np.maximum(-(s - h), 0.0)
    tau_c = 0.5 * np.maximum(rho_ice * g * h - rho_water * g * water_depth, 0.0)
    return np.isfinite(tau_c) & (tau_c > 0.0)


def stats(eta, mask):
    v = eta[mask]
    v = v[np.isfinite(v) & (v > 0)]
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "max_over_min": float(np.max(v) / max(np.min(v), 1e-30)),
        "log10_range": float(np.log10(np.max(v)) - np.log10(np.min(v))),
    }


def corr_log10(a, b, mask):
    aa, bb = a[mask], b[mask]
    ok = np.isfinite(aa) & np.isfinite(bb) & (aa > 0) & (bb > 0)
    return float(np.corrcoef(np.log10(aa[ok]), np.log10(bb[ok]))[0, 1])


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(NPZ)
    eta_no = z["eta_no_sliding"]
    eta_mx = z["eta_max_sliding"]
    geom = z["geom"].astype(bool)
    x_km, y_km = z["x_km"], z["y_km"]

    d = np.load(DATA)
    s = d["s"] if "s" in d.files else d["surface"]
    h = d["h"] if "h" in d.files else d["thickness"]
    grounded = grounded_mask(s.astype(float), h.astype(float)) & geom
    floating = geom & ~grounded
    d_eta = np.where(geom, eta_mx - eta_no, np.nan)

    summary = {
        "tag": "baseline_unit_corrected",
        "source_npz": str(NPZ),
        "frac_grounded": float(grounded.sum() / max(geom.sum(), 1)),
        "eta_stats_full": {"no": stats(eta_no, geom), "max": stats(eta_mx, geom)},
        "eta_stats_grounded": {"no": stats(eta_no, grounded), "max": stats(eta_mx, grounded)},
        "eta_stats_floating": {"no": stats(eta_no, floating), "max": stats(eta_mx, floating)},
        "log10_eta_corr_full": corr_log10(eta_no, eta_mx, geom),
        "log10_eta_corr_grounded": corr_log10(eta_no, eta_mx, grounded),
        "log10_eta_corr_floating": corr_log10(eta_no, eta_mx, floating),
        "delta_eta": {
            "mean_abs_full": float(np.nanmean(np.abs(d_eta[geom]))),
            "mean_abs_grounded": float(np.nanmean(np.abs(d_eta[grounded]))),
            "mean_abs_floating": float(np.nanmean(np.abs(d_eta[floating]))),
            "max_abs_full": float(np.nanmax(np.abs(d_eta[geom]))),
            "max_abs_grounded": float(np.nanmax(np.abs(d_eta[grounded]))),
        },
    }

    # losses from existing CSVs if present
    def last_loss(csv_path: Path):
        if not csv_path.is_file():
            return None
        import pandas as pd

        row = pd.read_csv(csv_path).iloc[-1]
        return {
            "train_total": float(row["train_total"]),
            "train_phys": float(row["train_phys"]),
            "epoch": int(row["epoch"]),
        }

    summary["losses"] = {
        "no": last_loss(ROOT / "Archive/logs/metrics_vi_only_log_vi_only_totten_no_sliding.csv"),
        "max": last_loss(ROOT / "Archive/logs/metrics_vi_only_log_vi_only_totten_max_sliding.csv"),
    }
    summary["eta_log_shift"] = {"no": None, "max": None}
    meta = ROOT / "outputs/figures/vi/totten_sliding_comparison/sliding_comparison_summary.json"
    if meta.is_file():
        m = json.loads(meta.read_text())
        summary["eta_log_shift"] = {
            "no": m.get("meta_from_training", {}).get("no_shift"),
            "max": m.get("meta_from_training", {}).get("max_shift"),
        }

    lim = max(abs(float(np.nanpercentile(d_eta, 2))), abs(float(np.nanpercentile(d_eta, 98))), 1e-4)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    for ax, field, title in (
        (axes[0], d_eta, "Δη full"),
        (axes[1], np.where(grounded, d_eta, np.nan), "Δη grounded"),
    ):
        img = ax.pcolormesh(x_km, y_km, field, shading="auto", cmap="RdBu_r", norm=norm)
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02)
    dg = d_eta[grounded]
    axes[2].hist(dg[np.isfinite(dg)], bins=60, color="steelblue", alpha=0.85)
    axes[2].axvline(0, color="k", ls="--")
    axes[2].set_title(
        f"corr_full={summary['log10_eta_corr_full']:.3f}  "
        f"corr_g={summary['log10_eta_corr_grounded']:.3f}"
    )
    axes[2].set_xlabel("Δη grounded")
    fig.suptitle("Baseline unit-corrected — grounded vs full diagnostics")
    fig.savefig(OUT / "delta_eta_grounded.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for ax, mask, title in (
        (axes[0], geom, f"full r={summary['log10_eta_corr_full']:.3f}"),
        (axes[1], grounded, f"grounded r={summary['log10_eta_corr_grounded']:.3f}"),
    ):
        ax.scatter(np.log10(eta_mx[mask]), np.log10(eta_no[mask]), s=3, alpha=0.25)
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("log10 η max")
        ax.set_ylabel("log10 η no")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    fig.savefig(OUT / "eta_corr_full_vs_grounded.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    (OUT / "experiment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "corr_full": summary["log10_eta_corr_full"],
        "corr_grounded": summary["log10_eta_corr_grounded"],
        "corr_floating": summary["log10_eta_corr_floating"],
        "mean_abs_deta_grounded": summary["delta_eta"]["mean_abs_grounded"],
        "frac_grounded": summary["frac_grounded"],
    }, indent=2))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
