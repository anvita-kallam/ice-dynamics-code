#!/usr/bin/env python3
"""Plot 1σ coverage diagnostics for a VI-only posterior vs spin-up η truth."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Archive"
DEFAULT_H5 = (
    ARCHIVE
    / "outputs/vi_only_eta_bias_suite/eta_bias_v1/raised_prior_center/posterior_samples.h5"
)
DEFAULT_NPZ = (
    ROOT
    / "outputs/spinup/production/more_sliding"
    / "SteadyState_more_sliding_10500yr_ramp4000_1refine_grid.npz"
)
DEFAULT_OUT = ROOT / "outputs/figures/vi/raised_prior_center_15/eta_1sigma_coverage.png"


def log10_safe(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values) & (values > 0)
    out[mask] = np.log10(values[mask])
    return out


def restore_masked(values: np.ndarray, grid_shape: tuple[int, ...], geom: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.shape == grid_shape:
        return values
    if values.ndim == 1 and values.size == int(geom.sum()):
        out = np.full(grid_shape, np.nan, dtype=np.float64)
        out[geom] = values
        return out
    raise ValueError(f"Cannot restore field of shape {values.shape} onto {grid_shape}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5", type=Path, default=DEFAULT_H5)
    p.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--title", type=str, default=r"$\eta_{\mathrm{init}}=15$ 1$\sigma$ coverage")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with np.load(args.npz) as z:
        eta_ref = np.asarray(z["viscosity"], dtype=np.float64)
        x = np.asarray(z["x"], dtype=np.float64)
        y = np.asarray(z["y"], dtype=np.float64)
        geom_ref = (
            np.isfinite(z["surface"])
            & np.isfinite(z["thickness"])
            & np.isfinite(z["bed"])
        )
    grid_shape = eta_ref.shape

    with h5py.File(args.h5, "r") as f:
        geom = f["geom_mask"][...].astype(bool) if "geom_mask" in f else geom_ref
        if geom.shape != grid_shape:
            geom = geom_ref
        eta_mean = restore_masked(f["eta_mean"][...], grid_shape, geom)
        if "eta_latent_std" in f:
            latent_std = restore_masked(f["eta_latent_std"][...], grid_shape, geom)
            log10_std = latent_std / math.log(10.0)
        elif "eta_std" in f:
            eta_std = restore_masked(f["eta_std"][...], grid_shape, geom)
            log10_std = eta_std / (eta_mean * math.log(10.0))
        else:
            raise KeyError("Need eta_latent_std or eta_std in HDF5")

    eval_mask = (
        geom
        & geom_ref
        & np.isfinite(eta_mean)
        & np.isfinite(eta_ref)
        & np.isfinite(log10_std)
        & (eta_ref > 0)
        & (eta_mean > 0)
        & (log10_std > 0)
    )
    log_err = log10_safe(eta_mean) - log10_safe(eta_ref)
    within_1 = np.abs(log_err) <= log10_std
    within_2 = np.abs(log_err) <= (2.0 * log10_std)
    zscore = np.full(grid_shape, np.nan, dtype=np.float64)
    zscore[eval_mask] = log_err[eval_mask] / log10_std[eval_mask]

    frac1 = float(np.mean(within_1[eval_mask]))
    frac2 = float(np.mean(within_2[eval_mask]))
    n = int(eval_mask.sum())

    coverage_map = np.full(grid_shape, np.nan, dtype=np.float64)
    coverage_map[eval_mask & within_1] = 1.0
    coverage_map[eval_mask & ~within_1] = 0.0

    x_km = (x[0, :] if x.ndim == 2 else x) / 1.0e3
    y_km = (y[:, 0] if y.ndim == 2 else y) / 1.0e3
    if x_km.ndim > 1:
        x_km = x_km.ravel()
    if y_km.ndim > 1:
        y_km = y_km.ravel()
    # Prefer 1D axes from mesh if needed
    if x.ndim == 2 and y.ndim == 2:
        x_km = x[0, :] / 1.0e3
        y_km = y[:, 0] / 1.0e3

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 3.8), constrained_layout=True)

    cmap = ListedColormap(["#c0392b", "#27ae60"])
    im = axes[0].pcolormesh(
        x_km, y_km, coverage_map, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x (km)")
    axes[0].set_ylabel("y (km)")
    axes[0].set_title(rf"within 1$\sigma$: {frac1:.1%} (n={n:,})")
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(["miss", "hit"])

    abs_z = np.abs(zscore[eval_mask])
    axes[1].hist(abs_z, bins=60, color="#4c78a8", alpha=0.85, edgecolor="none")
    axes[1].axvline(1.0, color="k", ls="--", lw=1.5, label=r"1$\sigma$")
    axes[1].axvline(2.0, color="0.4", ls=":", lw=1.5, label=r"2$\sigma$")
    axes[1].set_xlabel(r"$|\log_{10}\eta$ error$|$ / posterior std")
    axes[1].set_ylabel("count")
    axes[1].set_title(rf"normalized residuals (2$\sigma$ cover={frac2:.1%})")
    axes[1].legend(frameon=False)

    # Reliability-style cumulative coverage vs nominal Gaussian levels
    levels = np.linspace(0.5, 3.0, 26)
    empir = np.array([float(np.mean(abs_z <= lev)) for lev in levels])
    # Gaussian CDF for |Z| <= lev: erf(lev / sqrt(2))
    from math import erf, sqrt
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

    fig.suptitle(args.title, fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")
    print(f"calibration_within_1sigma={frac1:.6f}")
    print(f"calibration_within_2sigma={frac2:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
