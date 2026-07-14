#!/usr/bin/env python3
"""Validate VI predict HDF5 vs spin-up NPZ (more_sliding). No Jupyter required."""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Archive"
H5_PATH = ARCHIVE / "outputs/more_sliding_posterior_samples_torch.h5"
NPZ_PATH = (
    ROOT
    / "outputs/spinup/production/more_sliding/SteadyState_more_sliding_10500yr_ramp4000_1refine_grid.npz"
)
FIG_DIR = ROOT / "outputs/figures/vi/more_sliding"
PLOT_STRIDE = 4  # downsample maps for speed/memory (1 = full resolution)
PLOT_MAPS = False  # set True for pcolormesh maps (heavier)


def log10_safe(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    mask = values > 0
    out[mask] = np.log10(values[mask])
    return out


def load_data():
    if not H5_PATH.exists():
        raise FileNotFoundError(H5_PATH)
    if not NPZ_PATH.exists():
        raise FileNotFoundError(NPZ_PATH)

    with h5py.File(H5_PATH) as f:
        x = f["x"][...]
        y = f["y"][...]
        geom = f["geom_mask"][...].astype(bool)
        eta_mean = f["eta_mean"][...]
        eta_std = f["eta_std"][...]
        u_hat = f["u_hat"][...]
        v_hat = f["v_hat"][...]
        attrs = dict(f.attrs)

    with np.load(NPZ_PATH) as z:
        eta_ref = z["viscosity"]
        ux = z["ux"]
        uy = z["uy"]
        geom_ref = np.isfinite(z["surface"]) & np.isfinite(z["thickness"]) & np.isfinite(z["bed"])

    eval_mask = (
        geom
        & geom_ref
        & np.isfinite(eta_mean)
        & np.isfinite(eta_ref)
        & (eta_ref > 0)
        & (eta_mean > 0)
    )
    return {
        "x": x,
        "y": y,
        "geom": geom,
        "eta_mean": eta_mean,
        "eta_std": eta_std,
        "u_hat": u_hat,
        "v_hat": v_hat,
        "eta_ref": eta_ref,
        "ux": ux,
        "uy": uy,
        "eval_mask": eval_mask,
        "attrs": attrs,
    }


def compute_metrics(data: dict) -> dict[str, float]:
    eval_mask = data["eval_mask"]
    eta_mean = data["eta_mean"]
    eta_ref = data["eta_ref"]
    log_eta_pred = log10_safe(eta_mean[eval_mask])
    log_eta_ref = log10_safe(eta_ref[eval_mask])
    rel_eta = (eta_mean[eval_mask] - eta_ref[eval_mask]) / eta_ref[eval_mask]

    speed_hat = np.hypot(data["u_hat"], data["v_hat"])
    speed_ref = np.hypot(data["ux"], data["uy"])
    speed_mask = eval_mask & (speed_ref > 5.0)
    speed_rel = (speed_hat[speed_mask] - speed_ref[speed_mask]) / speed_ref[speed_mask]

    return {
        "log10_eta_rmse": float(np.sqrt(np.mean((log_eta_pred - log_eta_ref) ** 2))),
        "log10_eta_bias": float(np.mean(log_eta_pred - log_eta_ref)),
        "log10_eta_r": float(np.corrcoef(log_eta_pred, log_eta_ref)[0, 1]),
        "rel_eta_rmse": float(np.sqrt(np.mean(rel_eta ** 2))),
        "rel_eta_mae": float(np.mean(np.abs(rel_eta))),
        "speed_rel_rmse_gt5": float(np.sqrt(np.mean(speed_rel ** 2))),
        "speed_rel_mae_gt5": float(np.mean(np.abs(speed_rel))),
        "n_eval_pixels": int(eval_mask.sum()),
    }


def save_figures(data: dict, metrics: dict) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    s = max(int(PLOT_STRIDE), 1)
    x = data["x"][::s]
    y = data["y"][::s]
    X, Y = np.meshgrid(x, y)
    eval_mask = data["eval_mask"][::s, ::s]
    eta_ref = data["eta_ref"][::s, ::s]
    eta_mean = data["eta_mean"][::s, ::s]
    eta_std = data["eta_std"][::s, ::s]

    eta_ref_m = np.where(eval_mask, eta_ref, np.nan)
    eta_mean_m = np.where(eval_mask, eta_mean, np.nan)
    eta_std_m = np.where(eval_mask, eta_std, np.nan)
    log_err_m = np.where(
        eval_mask,
        log10_safe(eta_mean) - log10_safe(eta_ref),
        np.nan,
    )

    vals = np.concatenate([eta_ref[eval_mask], eta_mean[eval_mask]])
    norm = LogNorm(vmin=max(np.percentile(vals, 2), 1e-3), vmax=np.percentile(vals, 98))

    rng = np.random.default_rng(0)
    flat_idx = np.flatnonzero(data["eval_mask"])
    if flat_idx.size > 20_000:
        flat_idx = rng.choice(flat_idx, size=20_000, replace=False)
    scatter_ref = log10_safe(data["eta_ref"].flat[flat_idx])
    scatter_pred = log10_safe(data["eta_mean"].flat[flat_idx])

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    lim = np.nanpercentile(np.abs(log_err_m[eval_mask]), 98)
    if PLOT_MAPS:
        axes[0].pcolormesh(
            X / 1e3,
            Y / 1e3,
            log_err_m,
            shading="auto",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(-lim, 0, lim),
        )
        axes[0].set_title("log10 eta error")
    else:
        axes[0].hist(
            log_err_m[eval_mask].ravel(),
            bins=60,
            color="steelblue",
            edgecolor="white",
        )
        axes[0].set_title("log10 eta error histogram")
        axes[0].set_xlabel("log10 eta_pred - log10 eta_ref")

    axes[1].plot(scatter_ref, scatter_pred, ".", ms=1, alpha=0.3)
    lo, hi = scatter_ref.min(), scatter_ref.max()
    axes[1].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[1].set_title(f"scatter r={metrics['log10_eta_r']:.3f}")
    fig.tight_layout()
    path = FIG_DIR / "eta_error_scatter.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    if PLOT_MAPS:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, field, title, use_norm in zip(
            axes,
            [eta_ref_m, eta_mean_m, eta_std_m],
            ["Spin-up eta", "VI mean eta", "VI std eta"],
            [True, True, False],
        ):
            kw = {"shading": "auto", "cmap": "magma" if use_norm else "viridis"}
            if use_norm:
                im = ax.pcolormesh(X / 1e3, Y / 1e3, field, norm=norm, **kw)
            else:
                im = ax.pcolormesh(X / 1e3, Y / 1e3, field, **kw)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        fig.tight_layout()
        path = FIG_DIR / "eta_maps.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    gc.collect()
    return saved


def main() -> int:
    t0 = time.perf_counter()
    data = load_data()
    print(f"load: {time.perf_counter() - t0:.2f}s | grid {data['eta_mean'].shape}")
    print("HDF5 attrs:", data["attrs"])

    metrics = compute_metrics(data)
    print("\n=== eta vs spin-up reference ===")
    for key, val in metrics.items():
        if key == "n_eval_pixels":
            print(f"  {key:22s} {val:,}")
        else:
            print(f"  {key:22s} {val: .4f}")

    if "--no-plots" in sys.argv:
        return 0

    t1 = time.perf_counter()
    paths = save_figures(data, metrics)
    print(f"\nplots: {time.perf_counter() - t1:.2f}s")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
