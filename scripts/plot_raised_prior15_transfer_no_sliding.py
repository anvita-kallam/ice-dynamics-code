#!/usr/bin/env python3
"""Diagnostic plots for raised_prior (η_init=15) transfer predict.

Reads compact eta_maps.npz exports (from posterior HDF5) and writes the same
style of figures used for the η_init=15 in-domain suite.

  # no_sliding (defaults)
  python scripts/plot_raised_prior15_transfer_no_sliding.py

  # A40 soft-ice transfer
  python scripts/plot_raised_prior15_transfer_no_sliding.py \\
    --transfer-npz outputs/figures/vi_only/raised_prior15_on_A40/eta_maps.npz \\
    --output-dir outputs/figures/vi_only/raised_prior15_on_A40 \\
    --summary outputs/figures/vi_only/raised_prior15_on_A40/posterior_summary.json \\
    --target-label "A40 more_sliding"
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRANSFER = ROOT / "outputs/figures/vi_only/raised_prior15_on_no_sliding/eta_maps.npz"
DEFAULT_INDOMAIN = ROOT / "outputs/figures/vi_only/raised_prior_center_more_sliding/eta_maps.npz"
DEFAULT_OUT = ROOT / "outputs/figures/vi_only/raised_prior15_on_no_sliding"
DEFAULT_SUMMARY = DEFAULT_OUT / "posterior_summary.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transfer-npz", type=Path, default=DEFAULT_TRANSFER)
    p.add_argument("--indomain-npz", type=Path, default=DEFAULT_INDOMAIN)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument(
        "--target-label",
        type=str,
        default="no_sliding",
        help="Label for the transfer target case in figure titles",
    )
    p.add_argument(
        "--error-clim-scale",
        type=float,
        default=1.0,
        help="Multiply diverging error color limits (>1 makes residuals look milder)",
    )
    return p.parse_args()


def log10_safe(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values) & (values > 0)
    out[mask] = np.log10(values[mask])
    return out


def xy_km(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 2 and y.ndim == 2:
        return x[0, :] / 1e3, y[:, 0] / 1e3
    return np.asarray(x).ravel() / 1e3, np.asarray(y).ravel() / 1e3


def add_map(ax, x_km, y_km, field, title, cmap="viridis", norm=None, vmin=None, vmax=None):
    kw = {"shading": "auto", "cmap": cmap}
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"] = vmin
        kw["vmax"] = vmax
    im = ax.pcolormesh(x_km, y_km, field, **kw)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    return im


def load_maps(path: Path):
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    geom = data["geom_mask"].astype(bool)
    eta_ref = np.asarray(data["eta_ref"], float)
    eta_mean = np.asarray(data["eta_mean"], float)
    latent = np.asarray(data["eta_latent_std"], float)
    log10_std = latent / math.log(10.0)
    eval_mask = (
        geom
        & np.isfinite(eta_ref)
        & np.isfinite(eta_mean)
        & np.isfinite(log10_std)
        & (eta_ref > 0)
        & (eta_mean > 0)
        & (log10_std > 0)
    )
    return {
        "x": data["x"],
        "y": data["y"],
        "eta_ref": eta_ref,
        "eta_mean": eta_mean,
        "latent_std": latent,
        "log10_std": log10_std,
        "eval_mask": eval_mask,
        "ux": np.asarray(data["ux"], float),
        "uy": np.asarray(data["uy"], float),
        "u_hat": np.asarray(data["u"], float) if "u" in data else None,
        "v_hat": np.asarray(data["v"], float) if "v" in data else None,
        "s": np.asarray(data["surface"], float),
        "h": np.asarray(data["thickness"], float),
        "b": np.asarray(data["bed"], float),
        "s_hat": np.asarray(data["s"], float) if "s" in data else None,
        "h_hat": np.asarray(data["h"], float) if "h" in data else None,
        "b_hat": np.asarray(data["b"], float) if "b" in data else None,
    }


def metrics_from_maps(d):
    m = d["eval_mask"]
    log_err = log10_safe(d["eta_mean"]) - log10_safe(d["eta_ref"])
    pred = np.log10(d["eta_mean"][m])
    ref = np.log10(d["eta_ref"][m])
    return {
        "log10_eta_r": float(np.corrcoef(pred, ref)[0, 1]),
        "log10_eta_rmse": float(np.sqrt(np.mean(log_err[m] ** 2))),
        "log10_eta_bias": float(np.mean(log_err[m])),
        "eta_pred_mean": float(np.mean(d["eta_mean"][m])),
        "eta_ref_mean": float(np.mean(d["eta_ref"][m])),
        "eta_mean_ratio": float(np.mean(d["eta_mean"][m]) / np.mean(d["eta_ref"][m])),
        "calibration_within_1sigma": float(
            np.mean(np.abs(log_err[m]) <= d["log10_std"][m])
        ),
        "calibration_within_2sigma": float(
            np.mean(np.abs(log_err[m]) <= 2.0 * d["log10_std"][m])
        ),
        "n": int(m.sum()),
    }


def main():
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LogNorm, TwoSlopeNorm

    if not args.transfer_npz.is_file():
        raise FileNotFoundError(args.transfer_npz)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    d = load_maps(args.transfer_npz)
    metrics = metrics_from_maps(d)
    if args.summary.is_file():
        # Prefer evaluate_vi_only metrics when present
        metrics.update(
            {
                k: v
                for k, v in json.loads(args.summary.read_text()).items()
                if k in metrics or k.startswith("log10") or k.startswith("eta_") or k.startswith("calibration")
            }
        )

    s = max(1, int(args.stride))
    x_km, y_km = xy_km(d["x"], d["y"])
    if s > 1:
        x_km = x_km[::s]
        y_km = y_km[::s]

    def down(field):
        return field[::s, ::s]

    mask = down(d["eval_mask"])
    eta_ref = np.where(mask, down(d["eta_ref"]), np.nan)
    eta_mean = np.where(mask, down(d["eta_mean"]), np.nan)
    eta_diff = np.where(mask, down(d["eta_mean"]) - down(d["eta_ref"]), np.nan)
    log_ref = np.where(mask, log10_safe(down(d["eta_ref"])), np.nan)
    log_mean = np.where(mask, log10_safe(down(d["eta_mean"])), np.nan)
    log_diff = np.where(mask, log_mean - log_ref, np.nan)
    eta_std = np.where(mask, down(d["latent_std"]), np.nan)
    log10_std = np.where(mask, down(d["log10_std"]), np.nan)

    # --- 1) truth | estimate | diff ---
    eta_vals = np.concatenate([d["eta_ref"][d["eval_mask"]], d["eta_mean"][d["eval_mask"]]])
    eta_norm = LogNorm(
        vmin=max(float(np.percentile(eta_vals, 2)), 1e-3),
        vmax=float(np.percentile(eta_vals, 98)),
    )
    diff_lim = max(float(eta_norm.vmax), 1e-6) * float(args.error_clim_scale)
    log_vals = np.concatenate([log_ref[mask].ravel(), log_mean[mask].ravel()])
    log_vals = log_vals[np.isfinite(log_vals)]
    log_lo, log_hi = np.percentile(log_vals, [2, 98])
    # Color limit from residual magnitude (not absolute log η), then optional scale-up.
    err_abs = np.abs(log_diff[mask].ravel())
    err_abs = err_abs[np.isfinite(err_abs)]
    log_diff_lim = max(float(np.percentile(err_abs, 98)), 1e-3) * float(args.error_clim_scale)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.12, wspace=0.28, hspace=0.55)
    target = args.target_label
    xmin = float(np.nanmin(x_km))
    xmax = float(np.nanmax(x_km))
    ymin = float(np.nanmin(y_km))
    ymax = float(np.nanmax(y_km))

    def add_map_eq(ax, field, title, cmap="viridis", norm=None, vmin=None, vmax=None):
        im = add_map(ax, x_km, y_km, field, title, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", adjustable="box")
        return im

    im00 = add_map_eq(axes[0, 0], eta_ref, rf"Truth $\eta$ ({target})", cmap="magma", norm=eta_norm)
    im01 = add_map_eq(axes[0, 1], eta_mean, r"Estimate $\eta$ (raised prior→transfer)", cmap="magma", norm=eta_norm)
    im02 = add_map_eq(
        axes[0, 2], eta_diff, r"Estimate − truth $\eta$",
        cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-diff_lim, vmax=diff_lim),
    )
    im10 = add_map_eq(axes[1, 0], log_ref, r"Truth $\log_{10}\eta$", cmap="magma")
    im11 = add_map_eq(axes[1, 1], log_mean, r"Estimate $\log_{10}\eta$", cmap="magma")
    im12 = add_map_eq(
        axes[1, 2], log_diff, r"$\log_{10}$ estimate − $\log_{10}$ truth",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-log_diff_lim, vmax=log_diff_lim),
    )
    im10.set_clim(log_lo, log_hi)
    im11.set_clim(log_lo, log_hi)
    panels = [
        (axes[0, 0], im00, r"$\eta$ (MPa·yr)"),
        (axes[0, 1], im01, r"$\eta$ (MPa·yr)"),
        (axes[0, 2], im02, r"$\Delta\eta$"),
        (axes[1, 0], im10, None),
        (axes[1, 1], im11, None),
        (axes[1, 2], im12, None),
    ]
    fig.suptitle(
        rf"Transfer: raised_prior ($\eta_{{\mathrm{{init}}}}=15$) on {target}  |  "
        rf"$\log_{{10}}$ bias={metrics['log10_eta_bias']:.3f}, "
        rf"RMSE={metrics['log10_eta_rmse']:.3f}, $r$={metrics['log10_eta_r']:.3f}",
        fontsize=12,
    )
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    cbar_h, gap = 0.018, 0.070
    for ax, im, label in panels:
        p0 = inv.transform(ax.transData.transform((xmin, ymin)))
        p1 = inv.transform(ax.transData.transform((xmax, ymin)))
        cax = fig.add_axes([p0[0], p0[1] - gap - cbar_h, p1[0] - p0[0], cbar_h])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        if label:
            cbar.set_label(label, fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    path = args.output_dir / "eta_truth_estimate_diff.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")

    # --- 2) uncertainty + scatter + residual hist ---
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    im = add_map(axes[0], x_km, y_km, eta_std, r"VI latent $\theta$ std", cmap="viridis")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)

    rng = np.random.default_rng(0)
    flat_idx = np.flatnonzero(d["eval_mask"])
    if flat_idx.size > 25_000:
        flat_idx = rng.choice(flat_idx, size=25_000, replace=False)
    scatter_ref = log10_safe(d["eta_ref"].ravel()[flat_idx])
    scatter_pred = log10_safe(d["eta_mean"].ravel()[flat_idx])
    axes[1].plot(scatter_ref, scatter_pred, ".", ms=1.2, alpha=0.25, color="tab:blue")
    lo = float(np.nanmin([scatter_ref.min(), scatter_pred.min()]))
    hi = float(np.nanmax([scatter_ref.max(), scatter_pred.max()]))
    axes[1].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[1].set_xlabel(r"truth $\log_{10}\eta$")
    axes[1].set_ylabel(r"estimate $\log_{10}\eta$")
    axes[1].set_title(rf"Scatter  $r$={metrics['log10_eta_r']:.3f}")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(log_diff[mask].ravel(), bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[2].axvline(0.0, color="k", ls="--", lw=1)
    axes[2].axvline(metrics["log10_eta_bias"], color="tab:red", ls="-", lw=1.2, label="bias")
    axes[2].set_xlabel(r"$\log_{10}$ estimate − truth")
    axes[2].set_ylabel("count")
    axes[2].set_title("Residual histogram")
    axes[2].legend(frameon=False)
    path = args.output_dir / "eta_uncertainty_scatter.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    # --- 3) 1σ coverage ---
    log_err_full = log10_safe(d["eta_mean"]) - log10_safe(d["eta_ref"])
    within_1 = np.abs(log_err_full) <= d["log10_std"]
    within_2 = np.abs(log_err_full) <= (2.0 * d["log10_std"])
    zscore = np.full_like(log_err_full, np.nan)
    zscore[d["eval_mask"]] = log_err_full[d["eval_mask"]] / d["log10_std"][d["eval_mask"]]
    frac1 = float(np.mean(within_1[d["eval_mask"]]))
    frac2 = float(np.mean(within_2[d["eval_mask"]]))
    coverage_map = np.full(d["eta_mean"].shape, np.nan)
    coverage_map[d["eval_mask"] & within_1] = 1.0
    coverage_map[d["eval_mask"] & ~within_1] = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8), constrained_layout=True)
    cmap = ListedColormap(["#c0392b", "#27ae60"])
    im = add_map(
        axes[0], x_km, y_km, down(coverage_map),
        rf"within 1$\sigma$: {frac1:.1%} (n={metrics.get('n', int(d['eval_mask'].sum())):,})",
        cmap=cmap, vmin=0.0, vmax=1.0,
    )
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["miss", "hit"])

    abs_z = np.abs(zscore[d["eval_mask"]])
    axes[1].hist(abs_z, bins=60, color="#4c78a8", alpha=0.85, edgecolor="none")
    axes[1].axvline(1.0, color="k", ls="--", lw=1.5, label=r"1$\sigma$")
    axes[1].axvline(2.0, color="0.4", ls=":", lw=1.5, label=r"2$\sigma$")
    axes[1].set_xlabel(r"$|\log_{10}\eta$ error$|$ / posterior std")
    axes[1].set_ylabel("count")
    axes[1].set_title(rf"normalized residuals (2$\sigma$ cover={frac2:.1%})")
    axes[1].legend(frameon=False)

    from math import erf, sqrt

    levels = np.linspace(0.5, 3.0, 26)
    empir = np.array([float(np.mean(abs_z <= lev)) for lev in levels])
    nom = np.array([erf(lev / sqrt(2.0)) for lev in levels])
    axes[2].plot(nom, empir, "o-", color="#4c78a8", ms=4, label="empirical")
    axes[2].plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    axes[2].scatter([erf(1 / sqrt(2))], [frac1], s=60, c="#e45756", zorder=3, label=r"1$\sigma$")
    axes[2].set_xlabel("nominal Gaussian coverage")
    axes[2].set_ylabel("empirical coverage")
    axes[2].set_xlim(0.3, 1.01)
    axes[2].set_ylim(0.3, 1.01)
    axes[2].set_aspect("equal")
    axes[2].set_title("coverage calibration")
    axes[2].legend(frameon=False, loc="lower right")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(
        rf"Transfer raised_prior $\eta_{{\mathrm{{init}}}}=15$ on {target} — 1$\sigma$ coverage",
        fontsize=12,
    )
    path = args.output_dir / "eta_1sigma_coverage.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")

    # --- 4) speed truth | estimate | diff (PINN is OOD) ---
    if d["u_hat"] is not None and d["v_hat"] is not None:
        speed_ref = np.hypot(d["ux"], d["uy"])
        speed_hat = np.hypot(d["u_hat"], d["v_hat"])
        sp_ref = np.where(mask, down(speed_ref), np.nan)
        sp_hat = np.where(mask, down(speed_hat), np.nan)
        sp_diff = np.where(mask, down(speed_hat) - down(speed_ref), np.nan)
        sp_vals = speed_ref[d["eval_mask"]]
        sp_vmax = float(np.percentile(sp_vals[np.isfinite(sp_vals)], 98))
        dlim = max(abs(float(np.nanpercentile(sp_diff, 2))), abs(float(np.nanpercentile(sp_diff, 98))), 1.0)
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), constrained_layout=True)
        im0 = add_map(axes[0], x_km, y_km, sp_ref, f"Truth speed ({target})", cmap="viridis", vmin=0, vmax=sp_vmax)
        im1 = add_map(axes[1], x_km, y_km, sp_hat, "PINN speed (trained on more_sliding A=20)", cmap="viridis", vmin=0, vmax=sp_vmax)
        im2 = add_map(
            axes[2], x_km, y_km, sp_diff, "Estimate − truth speed",
            cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-dlim, vmax=dlim),
        )
        for im, ax, lab in ((im0, axes[0], "m/yr"), (im1, axes[1], "m/yr"), (im2, axes[2], "m/yr")):
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=lab)
        fig.suptitle(f"State transfer (frozen MeanNet is OOD on {target})", fontsize=12)
        path = args.output_dir / "speed_truth_estimate_diff.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")

    # --- 5) comparison: same posterior vs more_sliding / no_sliding truths ---
    if args.indomain_npz.is_file():
        di = load_maps(args.indomain_npz)
        mi = metrics_from_maps(di)
        xi, yi = xy_km(di["x"], di["y"])
        xi, yi = xi[::s], yi[::s]
        log_more_truth = np.where(di["eval_mask"], log10_safe(di["eta_ref"]), np.nan)[::s, ::s]
        panels_log = [
            log_more_truth[np.isfinite(log_more_truth)],
            log_ref[mask],
            log_mean[mask],
        ]
        vals = np.concatenate([p.ravel() for p in panels_log])
        lo, hi = np.percentile(vals, [2, 98])
        err_more = np.where(
            di["eval_mask"],
            log10_safe(di["eta_mean"]) - log10_safe(di["eta_ref"]),
            np.nan,
        )[::s, ::s]
        clim = max(
            abs(float(np.nanpercentile(err_more, 2))),
            abs(float(np.nanpercentile(err_more, 98))),
            abs(float(np.nanpercentile(log_diff, 2))),
            abs(float(np.nanpercentile(log_diff, 98))),
            0.3,
        ) * float(args.error_clim_scale)

        fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.2))
        fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.12, wspace=0.30, hspace=0.55)
        xmin = float(min(np.nanmin(xi), np.nanmin(x_km)))
        xmax = float(max(np.nanmax(xi), np.nanmax(x_km)))
        ymin = float(min(np.nanmin(yi), np.nanmin(y_km)))
        ymax = float(max(np.nanmax(yi), np.nanmax(y_km)))

        def add_map_eq(ax, xx, yy, field, title, **kw):
            im = add_map(ax, xx, yy, field, title, **kw)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal", adjustable="box")
            return im

        panels = []
        im = add_map_eq(
            axes[0, 0], xi, yi, log_more_truth,
            rf"more_sliding truth  (mean={mi['eta_ref_mean']:.1f})",
            cmap="magma", vmin=lo, vmax=hi,
        )
        panels.append((axes[0, 0], im, None))
        im = add_map_eq(
            axes[0, 1], x_km, y_km, log_ref,
            rf"{target} truth  (mean={metrics['eta_ref_mean']:.1f})",
            cmap="magma", vmin=lo, vmax=hi,
        )
        panels.append((axes[0, 1], im, None))
        im = add_map_eq(
            axes[0, 2], x_km, y_km, log_mean,
            rf"raised_prior estimate  (mean={metrics['eta_pred_mean']:.1f})",
            cmap="magma", vmin=lo, vmax=hi,
        )
        panels.append((axes[0, 2], im, None))

        im = add_map_eq(
            axes[1, 0], xi, yi, err_more,
            rf"in-domain error  $r$={mi['log10_eta_r']:.3f}",
            cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-clim, vmax=clim),
        )
        panels.append((axes[1, 0], im, None))
        im = add_map_eq(
            axes[1, 1], x_km, y_km, log_diff,
            rf"transfer error  $r$={metrics['log10_eta_r']:.3f}",
            cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-clim, vmax=clim),
        )
        panels.append((axes[1, 1], im, None))

        # Table panel: keep content fully inside axes (avoid figure-edge clip).
        axes[1, 2].set_axis_off()
        axes[1, 2].set_title("same raised_prior ckpt", fontsize=11, pad=10)
        table = axes[1, 2].table(
            cellText=[
                [r"$\log_{10}$ $r$", f"{mi['log10_eta_r']:.3f}", f"{metrics['log10_eta_r']:.3f}"],
                [r"$\log_{10}$ RMSE", f"{mi['log10_eta_rmse']:.3f}", f"{metrics['log10_eta_rmse']:.3f}"],
                [r"$\log_{10}$ bias", f"{mi['log10_eta_bias']:.3f}", f"{metrics['log10_eta_bias']:.3f}"],
                [r"$\eta$ mean ratio", f"{mi['eta_mean_ratio']:.3f}", f"{metrics['eta_mean_ratio']:.3f}"],
                [r"1$\sigma$ cover", f"{mi['calibration_within_1sigma']:.1%}", f"{metrics['calibration_within_1sigma']:.1%}"],
            ],
            colLabels=["", "in-domain", "transfer"],
            loc="center",
            cellLoc="center",
            bbox=[0.05, 0.18, 0.90, 0.68],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.45)

        fig.suptitle(
            rf"In-domain more_sliding vs transfer to {target}",
            fontsize=12,
        )
        fig.canvas.draw()
        inv = fig.transFigure.inverted()
        cbar_h, gap = 0.018, 0.070
        for ax, im, label in panels:
            p0 = inv.transform(ax.transData.transform((xmin, ymin)))
            p1 = inv.transform(ax.transData.transform((xmax, ymin)))
            cax = fig.add_axes([p0[0], p0[1] - gap - cbar_h, p1[0] - p0[0], cbar_h])
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            if label:
                cbar.set_label(label, fontsize=9)
            cbar.ax.tick_params(labelsize=8)
        path = args.output_dir / "transfer_vs_indomain_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.35)
        plt.close(fig)
        print(f"wrote {path}")
    else:
        print(f"skip comparison (missing {args.indomain_npz})")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
