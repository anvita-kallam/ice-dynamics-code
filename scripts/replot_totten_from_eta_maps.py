#!/usr/bin/env python3
"""Replot Totten no vs max sliding η maps from eta_maps.npz (no torch).

Uses the Jul-28 unit-corrected predictions already written by
plot_totten_sliding_comparison.py on the cluster.

  python scripts/replot_totten_from_eta_maps.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "outputs/figures/vi/totten_sliding_comparison/eta_maps.npz"
DEFAULT_OUT = ROOT / "outputs/figures/vi/totten_sliding_comparison/unit_corrected"
PAS_PER_MPA_YR = 3.15576e13
ETA_MIN_NEW = 1.0e-3  # MPa·yr (converted from 1e3 Pa·yr)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--summary-json", type=Path, default=None,
                   help="Optional sliding_comparison_summary.json for metadata")
    return p.parse_args()


def stats(arr, floor=ETA_MIN_NEW):
    v = arr[np.isfinite(arr) & (arr > 0)]
    return {
        "n": int(v.size),
        "min": float(np.min(v)),
        "median": float(np.median(v)),
        "mean": float(np.mean(v)),
        "max": float(np.max(v)),
        "std": float(np.std(v)),
        "max_over_min": float(np.max(v) / max(np.min(v), 1e-30)),
        "log10_range": float(np.log10(np.max(v)) - np.log10(np.min(v))),
        "frac_at_floor": float(np.mean(v <= floor * (1.0 + 1e-3))),
        "frac_le_1": float(np.mean(v <= 1.001)),
    }


def add_map(ax, x_km, y_km, field, title, *, cmap, norm=None, vmin=None, vmax=None):
    image = ax.pcolormesh(
        x_km, y_km, field, shading="auto", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax
    )
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    return image


def main():
    args = parse_args()
    if not args.npz.is_file():
        raise FileNotFoundError(args.npz)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    with np.load(args.npz) as z:
        x_km = z["x_km"]
        y_km = z["y_km"]
        geom = z["geom"]
        eta_no = z["eta_no_sliding"]
        eta_mx = z["eta_max_sliding"]
        log10_no = z["log10_eta_no_sliding"]
        log10_mx = z["log10_eta_max_sliding"]
        std_no = z["log10_std_no_sliding"]
        std_mx = z["log10_std_max_sliding"]
        d_eta = z["delta_eta_max_minus_no"]
        d_log10 = z["delta_log10_eta_no_minus_max"]
        d_std = z["delta_log10_std_no_minus_max"]
        c_no = float(z["friction_C_no"])
        c_mx = float(z["friction_C_max"])

    meta = {}
    summary_path = args.summary_json or (args.npz.parent / "sliding_comparison_summary.json")
    if summary_path.is_file():
        meta = json.loads(summary_path.read_text())

    s_no = stats(eta_no)
    s_mx = stats(eta_mx)

    # Shared norms
    pos = np.concatenate([
        eta_no[np.isfinite(eta_no) & (eta_no > 0)],
        eta_mx[np.isfinite(eta_mx) & (eta_mx > 0)],
    ])
    eta_norm = LogNorm(
        vmin=max(float(np.percentile(pos, 5)), 1e-3),
        vmax=float(np.percentile(pos, 95)),
    )
    std_vals = np.concatenate([
        std_no[np.isfinite(std_no)],
        std_mx[np.isfinite(std_mx)],
    ])
    std_lo, std_hi = float(np.percentile(std_vals, 5)), float(np.percentile(std_vals, 95))
    d_lim = max(abs(float(np.nanpercentile(d_log10, 2))),
                abs(float(np.nanpercentile(d_log10, 98))), 1e-3)
    d_norm = TwoSlopeNorm(vcenter=0.0, vmin=-d_lim, vmax=d_lim)
    ds_lim = max(abs(float(np.nanpercentile(d_std, 2))),
                 abs(float(np.nanpercentile(d_std, 98))), 1e-4)
    ds_norm = TwoSlopeNorm(vcenter=0.0, vmin=-ds_lim, vmax=ds_lim)
    d_eta_lim = max(abs(float(np.nanpercentile(d_eta, 2))),
                    abs(float(np.nanpercentile(d_eta, 98))), 1e-4)
    d_eta_norm = TwoSlopeNorm(vcenter=0.0, vmin=-d_eta_lim, vmax=d_eta_lim)

    subtitle = (
        f"η bounds: min={ETA_MIN_NEW:g}, max=1e4 MPa·yr  |  "
        f"frac@floor=0  |  no median={s_no['median']:.2f}, max median={s_mx['median']:.2f}"
    )

    # --- 3×2 main comparison -------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(10.5, 14.0), constrained_layout=True)
    panels = (
        (axes[0, 0], eta_no, f"no_sliding η mean\n(C={c_no:g})", "viridis", eta_norm, None, None, "η (MPa·yr)"),
        (axes[0, 1], eta_mx, f"max_sliding η mean\n(C={c_mx:g})", "viridis", eta_norm, None, None, "η (MPa·yr)"),
        (axes[1, 0], std_no, "no_sliding log10 η std", "magma", None, std_lo, std_hi, "log10"),
        (axes[1, 1], std_mx, "max_sliding log10 η std", "magma", None, std_lo, std_hi, "log10"),
        (axes[2, 0], d_log10, "Δ log10 η\n(no − max)", "RdBu_r", d_norm, None, None, "log10"),
        (axes[2, 1], d_std, "Δ log10 η std\n(no − max)", "RdBu_r", ds_norm, None, None, "log10"),
    )
    for ax, field, title, cmap, norm, vmin, vmax, unit in panels:
        image = add_map(ax, x_km, y_km, field, title, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label=unit)
    fig.suptitle(
        "Totten sequential VI — unit-corrected η bounds\n" + subtitle,
        fontsize=12,
    )
    maps_path = args.output_dir / "eta_mean_std_comparison.png"
    fig.savefig(maps_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- Viscosity trio ------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), constrained_layout=True)
    for ax, field, title, cmap, norm, unit in (
        (axes[0], eta_no, f"no sliding η\n(C={c_no:g})", "viridis", eta_norm, "η (MPa·yr)"),
        (axes[1], eta_mx, f"max sliding η\n(C={c_mx:g})", "viridis", eta_norm, "η (MPa·yr)"),
        (axes[2], d_eta, "Δη = max − no sliding", "RdBu_r", d_eta_norm, "η (MPa·yr)"),
    ):
        image = add_map(ax, x_km, y_km, field, title, cmap=cmap, norm=norm)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label=unit)
    fig.suptitle("Totten VI viscosity (unit-corrected bounds)", fontsize=13)
    vis_path = args.output_dir / "eta_viscosity_no_vs_max.png"
    fig.savefig(vis_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- Pa·s version --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), constrained_layout=True)
    eta_no_pas = eta_no * PAS_PER_MPA_YR
    eta_mx_pas = eta_mx * PAS_PER_MPA_YR
    d_pas = d_eta * PAS_PER_MPA_YR
    pos_pas = pos * PAS_PER_MPA_YR
    pas_norm = LogNorm(
        vmin=max(float(np.percentile(pos_pas, 5)), 1e10),
        vmax=float(np.percentile(pos_pas, 95)),
    )
    d_pas_lim = max(abs(float(np.nanpercentile(d_pas, 2))),
                    abs(float(np.nanpercentile(d_pas, 98))), 1.0)
    d_pas_norm = TwoSlopeNorm(vcenter=0.0, vmin=-d_pas_lim, vmax=d_pas_lim)
    for ax, field, title, cmap, norm, unit in (
        (axes[0], eta_no_pas, f"no sliding η\n(C={c_no:g})", "viridis", pas_norm, "η (Pa·s)"),
        (axes[1], eta_mx_pas, f"max sliding η\n(C={c_mx:g})", "viridis", pas_norm, "η (Pa·s)"),
        (axes[2], d_pas, "Δη = max − no", "RdBu_r", d_pas_norm, "η (Pa·s)"),
    ):
        image = add_map(ax, x_km, y_km, field, title, cmap=cmap, norm=norm)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label=unit)
    fig.suptitle(
        f"Totten VI viscosity in Pa·s  (1 MPa·yr = {PAS_PER_MPA_YR:.3g} Pa·s)",
        fontsize=12,
    )
    pas_path = args.output_dir / "eta_viscosity_no_vs_max_Pa_s.png"
    fig.savefig(pas_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- Diagnostics: scatter + hist + floor --------------------------------
    no_f = eta_no[geom & np.isfinite(eta_no) & (eta_no > 0)]
    mx_f = eta_mx[geom & np.isfinite(eta_mx) & (eta_mx > 0)]
    no_s = std_no[geom & np.isfinite(std_no)]
    mx_s = std_mx[geom & np.isfinite(std_mx)]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    axes[0].scatter(np.log10(mx_f), np.log10(no_f), s=3, alpha=0.25, c="steelblue")
    lims = [
        min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]),
        max(axes[0].get_xlim()[1], axes[0].get_ylim()[1]),
    ]
    axes[0].plot(lims, lims, "k--", lw=1)
    r = float(np.corrcoef(np.log10(mx_f), np.log10(no_f))[0, 1])
    axes[0].set_xlabel("log10 η max_sliding")
    axes[0].set_ylabel("log10 η no_sliding")
    axes[0].set_title(f"η mean agreement (r={r:.3f})")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(np.log10(no_f), bins=60, alpha=0.55, label="no_sliding", color="C0")
    axes[1].hist(np.log10(mx_f), bins=60, alpha=0.55, label="max_sliding", color="C1")
    axes[1].axvline(np.log10(ETA_MIN_NEW), color="k", ls=":", lw=1.2, label=f"η_min={ETA_MIN_NEW:g}")
    axes[1].axvline(0.0, color="gray", ls="--", lw=1.0, label="old floor η=1")
    axes[1].set_xlabel("log10 η (MPa·yr)")
    axes[1].set_ylabel("count")
    axes[1].legend(fontsize=7)
    axes[1].set_title(
        f"η distributions\n"
        f"frac≤1: no={s_no['frac_le_1']:.2f}, max={s_mx['frac_le_1']:.2f}  |  "
        f"frac@η_min: {s_no['frac_at_floor']:.3f}/{s_mx['frac_at_floor']:.3f}"
    )
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(no_s, bins=60, alpha=0.55, label="no_sliding", color="C0")
    axes[2].hist(mx_s, bins=60, alpha=0.55, label="max_sliding", color="C1")
    axes[2].set_xlabel("posterior std (log10 η)")
    axes[2].set_ylabel("count")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Uncertainty distributions")
    axes[2].grid(True, alpha=0.3)
    diag_path = args.output_dir / "eta_sliding_diagnostics.png"
    fig.savefig(diag_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- Ratio map -----------------------------------------------------------
    ratio = np.where(geom, eta_no / np.maximum(eta_mx, 1e-12), np.nan)
    ratio_lim = max(
        abs(float(np.nanpercentile(np.log10(ratio), 2))),
        abs(float(np.nanpercentile(np.log10(ratio), 98))),
        1e-3,
    )
    ratio_norm = TwoSlopeNorm(vcenter=0.0, vmin=-ratio_lim, vmax=ratio_lim)
    fig, ax = plt.subplots(figsize=(6.2, 7.2), constrained_layout=True)
    image = add_map(
        ax, x_km, y_km, np.log10(ratio), "log10(η_no / η_max)",
        cmap="RdBu_r", norm=ratio_norm,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label="log10 ratio")
    ratio_path = args.output_dir / "eta_mean_log_ratio.png"
    fig.savefig(ratio_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    out_summary = {
        "source_npz": str(args.npz),
        "eta_min_MPa_yr": ETA_MIN_NEW,
        "eta_max_MPa_yr": 1.0e4,
        "friction_C_no": c_no,
        "friction_C_max": c_mx,
        "no_sliding": s_no,
        "max_sliding": s_mx,
        "comparison": {
            "log10_eta_corr": r,
            "mean_abs_eta_diff_max_minus_no": float(np.nanmean(np.abs(d_eta))),
            "median_eta_diff_max_minus_no": float(np.nanmedian(d_eta)),
            "mean_abs_log10_eta_diff": float(np.nanmean(np.abs(d_log10))),
        },
        "meta_from_training": {
            "no_shift": meta.get("no_sliding", {}).get("eta_log_shift"),
            "max_shift": meta.get("max_sliding", {}).get("eta_log_shift"),
            "no_epoch": meta.get("no_sliding", {}).get("epoch"),
            "max_epoch": meta.get("max_sliding", {}).get("epoch"),
        },
        "figures": [str(p) for p in (maps_path, vis_path, pas_path, diag_path, ratio_path)],
    }
    out_json = args.output_dir / "sliding_comparison_summary.json"
    out_json.write_text(json.dumps(out_summary, indent=2) + "\n")

    print(json.dumps({k: out_summary[k] for k in (
        "no_sliding", "max_sliding", "comparison")}, indent=2))
    for p in (maps_path, vis_path, pas_path, diag_path, ratio_path, out_json):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
