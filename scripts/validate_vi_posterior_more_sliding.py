#!/usr/bin/env python3
"""Validate VI predict HDF5 vs spin-up NPZ (more_sliding). No Jupyter required.

Writes comparison maps under outputs/figures/vi/more_sliding/:
  - eta: truth, estimate, estimate-truth (linear + log10)
  - speed: truth, estimate, difference
  - surface / thickness / bed: truth vs PINN inference
  - scatter + error histogram for η
"""

from __future__ import annotations

import argparse
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


def log10_safe(values: np.ndarray) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    mask = np.isfinite(values) & (values > 0)
    out[mask] = np.log10(values[mask])
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true", help="Print metrics only")
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Spatial downsample for map plots (default 2; use 1 for full res)",
    )
    parser.add_argument(
        "--h5",
        type=Path,
        default=H5_PATH,
        help="VI posterior HDF5 path",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=NPZ_PATH,
        help="Spin-up reference NPZ path",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=FIG_DIR,
        help="Output directory for figures",
    )
    return parser.parse_args()


def _restore_masked_field(
    values: np.ndarray,
    grid_shape: tuple[int, ...],
    geom_mask: np.ndarray,
    name: str,
) -> np.ndarray:
    """Restore either a full-grid or geometry-masked vector to the reference grid."""
    values = np.asarray(values)
    if values.shape == grid_shape:
        return values
    if values.ndim == 1 and values.size == int(geom_mask.sum()):
        full = np.full(grid_shape, np.nan, dtype=values.dtype)
        full[geom_mask] = values
        return full
    if values.size == int(np.prod(grid_shape)):
        return values.reshape(grid_shape)
    raise ValueError(
        f"Cannot map HDF5 field {name!r} with shape {values.shape} "
        f"to grid {grid_shape} ({int(geom_mask.sum())} geometry points)"
    )


def _posterior_sample_std(
    dataset: h5py.Dataset,
    grid_shape: tuple[int, ...],
    geom_mask: np.ndarray,
    chunk_points: int = 8192,
) -> np.ndarray:
    """Compute pointwise sample std without loading the entire sample matrix."""
    if dataset.ndim != 2:
        raise ValueError(f"eta_samples must be 2-D, got {dataset.shape}")
    n_points = dataset.shape[1]
    std = np.empty(n_points, dtype=np.float64)
    for start in range(0, n_points, chunk_points):
        stop = min(start + chunk_points, n_points)
        std[start:stop] = np.std(dataset[:, start:stop], axis=0, ddof=0)
    return _restore_masked_field(std, grid_shape, geom_mask, "eta_samples std")


def load_data(h5_path: Path, npz_path: Path) -> dict:
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    with np.load(npz_path) as z:
        eta_ref = z["viscosity"]
        ux = z["ux"]
        uy = z["uy"]
        surface = z["surface"] if "surface" in z.files else None
        thickness = z["thickness"] if "thickness" in z.files else None
        bed = z["bed"] if "bed" in z.files else None
        geom_ref = np.isfinite(z["surface"]) & np.isfinite(z["thickness"]) & np.isfinite(z["bed"])
        if "X" in z.files and "Y" in z.files:
            x_ref, y_ref = z["X"], z["Y"]
        elif "x" in z.files and "y" in z.files:
            x_raw, y_raw = z["x"], z["y"]
            if x_raw.shape == eta_ref.shape and y_raw.shape == eta_ref.shape:
                x_ref, y_ref = x_raw, y_raw
            else:
                x_ref, y_ref = np.meshgrid(np.ravel(x_raw), np.ravel(y_raw))
        else:
            raise KeyError("Reference NPZ must contain X/Y or x/y coordinates")

    grid_shape = eta_ref.shape
    with h5py.File(h5_path) as f:
        attrs = dict(f.attrs)
        is_masked_vi_only = "geom_mask" not in f
        geom = f["geom_mask"][...].astype(bool) if "geom_mask" in f else geom_ref.copy()
        x = f["x"][...] if "x" in f else x_ref
        y = f["y"][...] if "y" in f else y_ref

        def field(primary: str, fallback: str | None = None) -> np.ndarray | None:
            key = primary if primary in f else fallback
            if key is None or key not in f:
                return None
            return _restore_masked_field(f[key][...], grid_shape, geom, key)

        eta_mean = field("eta_mean")
        if eta_mean is None:
            raise KeyError("HDF5 is missing eta_mean")
        eta_std = field("eta_std")
        if eta_std is None:
            if "eta_samples" in f:
                eta_std = _posterior_sample_std(f["eta_samples"], grid_shape, geom)
            elif "eta_latent_std" in f:
                # Delta-method conversion from latent log-η std to physical η std.
                latent_std = field("eta_latent_std")
                eta_std = eta_mean * latent_std
            else:
                raise KeyError("HDF5 needs eta_std, eta_samples, or eta_latent_std")

        u_hat = field("u_hat", "u")
        v_hat = field("v_hat", "v")
        if u_hat is None or v_hat is None:
            raise KeyError("HDF5 needs u_hat/v_hat or VI-only u/v fields")
        s_hat = field("s_hat", "s")
        h_hat = field("h_hat", "h")
        b_hat = field("b_hat", "b")
        attrs["source_schema"] = "vi_only_masked" if is_masked_vi_only else "joint_full_grid"

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
        "s_hat": s_hat,
        "h_hat": h_hat,
        "b_hat": b_hat,
        "eta_ref": eta_ref,
        "ux": ux,
        "uy": uy,
        "surface": surface,
        "thickness": thickness,
        "bed": bed,
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


def _masked(field: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    if field is None:
        return None
    return np.where(mask, field, np.nan)


def _add_map(ax, X_km, Y_km, field, title, *, cmap="viridis", norm=None, clim=None):
    kw = {"shading": "auto", "cmap": cmap}
    if norm is not None:
        kw["norm"] = norm
    elif clim is not None:
        kw["vmin"], kw["vmax"] = clim
    im = ax.pcolormesh(X_km, Y_km, field, **kw)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    return im


def save_figures(data: dict, metrics: dict, fig_dir: Path, stride: int) -> list[Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    s = max(int(stride), 1)
    x = np.asarray(data["x"]).ravel()
    y = np.asarray(data["y"]).ravel()
    # HDF5 may store full 2D x/y grids or 1D axes.
    if data["x"].ndim == 2:
        X = data["x"][::s, ::s]
        Y = data["y"][::s, ::s]
        x_km, y_km = X / 1e3, Y / 1e3
    else:
        x = data["x"][::s]
        y = data["y"][::s]
        X, Y = np.meshgrid(x, y)
        x_km, y_km = X / 1e3, Y / 1e3

    mask = data["eval_mask"][::s, ::s]
    eta_ref = _masked(data["eta_ref"][::s, ::s], mask)
    eta_mean = _masked(data["eta_mean"][::s, ::s], mask)
    eta_std = _masked(data["eta_std"][::s, ::s], mask)
    eta_diff = np.where(mask, data["eta_mean"][::s, ::s] - data["eta_ref"][::s, ::s], np.nan)
    log_eta_ref = _masked(log10_safe(data["eta_ref"][::s, ::s]), mask)
    log_eta_mean = _masked(log10_safe(data["eta_mean"][::s, ::s]), mask)
    log_eta_diff = np.where(mask, log_eta_mean - log_eta_ref, np.nan)

    speed_ref = _masked(np.hypot(data["ux"], data["uy"])[::s, ::s], mask)
    speed_hat = _masked(np.hypot(data["u_hat"], data["v_hat"])[::s, ::s], mask)
    speed_diff = np.where(
        mask,
        np.hypot(data["u_hat"], data["v_hat"])[::s, ::s]
        - np.hypot(data["ux"], data["uy"])[::s, ::s],
        np.nan,
    )

    # --- 1) η maps: truth | estimate | difference ---
    eta_vals = np.concatenate(
        [data["eta_ref"][data["eval_mask"]], data["eta_mean"][data["eval_mask"]]]
    )
    eta_norm = LogNorm(
        vmin=max(float(np.percentile(eta_vals, 2)), 1e-3),
        vmax=float(np.percentile(eta_vals, 98)),
    )
    # Difference panels share the color-scale magnitude of the truth/estimate maps.
    diff_lim = max(float(eta_norm.vmax), 1e-6)
    log_vals = np.concatenate(
        [log_eta_ref[mask].ravel(), log_eta_mean[mask].ravel()]
    )
    log_vals = log_vals[np.isfinite(log_vals)]
    if log_vals.size:
        log_lo, log_hi = np.percentile(log_vals, [2, 98])
    else:
        log_lo, log_hi = -1.0, 1.0
    log_diff_lim = max(abs(float(log_lo)), abs(float(log_hi)), 1e-3)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)
    im00 = _add_map(axes[0, 0], x_km, y_km, eta_ref, r"Truth $\eta$ (spin-up)", cmap="magma", norm=eta_norm)
    im01 = _add_map(axes[0, 1], x_km, y_km, eta_mean, r"Estimate $\eta$ (VI mean)", cmap="magma", norm=eta_norm)
    im02 = _add_map(
        axes[0, 2],
        x_km,
        y_km,
        eta_diff,
        r"Estimate − truth $\eta$",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-diff_lim, vmax=diff_lim),
    )
    fig.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.02, label=r"$\eta$ (MPa·yr)")
    fig.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.02, label=r"$\eta$ (MPa·yr)")
    fig.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.02, label=r"$\Delta\eta$")

    im10 = _add_map(axes[1, 0], x_km, y_km, log_eta_ref, r"Truth $\log_{10}\eta$", cmap="magma")
    im11 = _add_map(axes[1, 1], x_km, y_km, log_eta_mean, r"Estimate $\log_{10}\eta$", cmap="magma")
    im12 = _add_map(
        axes[1, 2],
        x_km,
        y_km,
        log_eta_diff,
        r"$\log_{10}$ estimate − $\log_{10}$ truth",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-log_diff_lim, vmax=log_diff_lim),
    )
    # Match log color limits
    im10.set_clim(log_lo, log_hi)
    im11.set_clim(log_lo, log_hi)
    fig.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.02)
    fig.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.02)
    fig.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.02)
    fig.suptitle(
        rf"Viscosity recovery  |  $\log_{{10}}$ bias={metrics['log10_eta_bias']:.3f}, "
        rf"RMSE={metrics['log10_eta_rmse']:.3f}, $r$={metrics['log10_eta_r']:.3f}",
        fontsize=12,
    )
    path = fig_dir / "eta_truth_estimate_diff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # --- 2) η uncertainty + scatter ---
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    im = _add_map(axes[0], x_km, y_km, eta_std, r"VI $\eta$ std", cmap="viridis")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.02)

    rng = np.random.default_rng(0)
    flat_idx = np.flatnonzero(data["eval_mask"])
    if flat_idx.size > 25_000:
        flat_idx = rng.choice(flat_idx, size=25_000, replace=False)
    scatter_ref = log10_safe(data["eta_ref"].ravel()[flat_idx])
    scatter_pred = log10_safe(data["eta_mean"].ravel()[flat_idx])
    axes[1].plot(scatter_ref, scatter_pred, ".", ms=1.2, alpha=0.25, color="tab:blue")
    lo = float(np.nanmin([scatter_ref.min(), scatter_pred.min()]))
    hi = float(np.nanmax([scatter_ref.max(), scatter_pred.max()]))
    axes[1].plot([lo, hi], [lo, hi], "r--", lw=1)
    axes[1].set_xlabel(r"truth $\log_{10}\eta$")
    axes[1].set_ylabel(r"estimate $\log_{10}\eta$")
    axes[1].set_title(rf"Scatter  $r$={metrics['log10_eta_r']:.3f}")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(
        log_eta_diff[mask].ravel(),
        bins=60,
        color="steelblue",
        edgecolor="white",
        alpha=0.9,
    )
    axes[2].axvline(0.0, color="k", ls="--", lw=1)
    axes[2].axvline(metrics["log10_eta_bias"], color="tab:red", ls="-", lw=1.2, label="mean bias")
    axes[2].set_xlabel(r"$\log_{10}\eta_{\mathrm{est}}-\log_{10}\eta_{\mathrm{truth}}$")
    axes[2].set_ylabel("count")
    axes[2].set_title("η error histogram")
    axes[2].legend(fontsize=8)
    path = fig_dir / "eta_uncertainty_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # --- 3) Speed maps ---
    speed_vals = np.concatenate(
        [
            np.hypot(data["ux"], data["uy"])[data["eval_mask"]],
            np.hypot(data["u_hat"], data["v_hat"])[data["eval_mask"]],
        ]
    )
    speed_vals = speed_vals[np.isfinite(speed_vals) & (speed_vals > 0)]
    speed_norm = LogNorm(
        vmin=max(float(np.percentile(speed_vals, 2)), 1e-2),
        vmax=float(np.percentile(speed_vals, 98)),
    )
    spd_diff_lim = float(np.nanpercentile(np.abs(speed_diff[mask]), 98))
    spd_diff_lim = max(spd_diff_lim, 1e-3)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), constrained_layout=True)
    im0 = _add_map(axes[0], x_km, y_km, speed_ref, r"Truth speed $|u|$", cmap="plasma", norm=speed_norm)
    im1 = _add_map(axes[1], x_km, y_km, speed_hat, r"PINN speed $|u|$", cmap="plasma", norm=speed_norm)
    im2 = _add_map(
        axes[2],
        x_km,
        y_km,
        speed_diff,
        r"Estimate − truth speed",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-spd_diff_lim, vmax=spd_diff_lim),
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.02, label="m/yr")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.02, label="m/yr")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.02, label="m/yr")
    fig.suptitle(
        rf"Speed  |  rel-RMSE (|u|>5)={metrics['speed_rel_rmse_gt5']:.4f}",
        fontsize=12,
    )
    path = fig_dir / "speed_truth_estimate_diff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # --- 4) Geometry: surface / thickness / bed ---
    geo_panels = []
    if data["surface"] is not None and data["s_hat"] is not None:
        geo_panels.append(
            (
                "surface",
                _masked(data["surface"][::s, ::s], mask),
                _masked(data["s_hat"][::s, ::s], mask),
                "m",
            )
        )
    if data["thickness"] is not None and data["h_hat"] is not None:
        geo_panels.append(
            (
                "thickness",
                _masked(data["thickness"][::s, ::s], mask),
                _masked(data["h_hat"][::s, ::s], mask),
                "m",
            )
        )
    if data["bed"] is not None and data["b_hat"] is not None:
        geo_panels.append(
            (
                "bed",
                _masked(data["bed"][::s, ::s], mask),
                _masked(data["b_hat"][::s, ::s], mask),
                "m",
            )
        )

    if geo_panels:
        nrows = len(geo_panels)
        fig, axes = plt.subplots(nrows, 3, figsize=(13.5, 3.4 * nrows), constrained_layout=True)
        if nrows == 1:
            axes = np.asarray([axes])
        for row, (name, truth, est, unit) in enumerate(geo_panels):
            diff = est - truth
            finite = np.isfinite(truth) & np.isfinite(est)
            vals = np.concatenate([truth[finite], est[finite]])
            clim = (float(np.percentile(vals, 2)), float(np.percentile(vals, 98)))
            dlim = float(np.nanpercentile(np.abs(diff[finite]), 98))
            dlim = max(dlim, 1e-3)
            im0 = _add_map(axes[row, 0], x_km, y_km, truth, f"Truth {name}", cmap="terrain", clim=clim)
            im1 = _add_map(axes[row, 1], x_km, y_km, est, f"PINN {name}", cmap="terrain", clim=clim)
            im2 = _add_map(
                axes[row, 2],
                x_km,
                y_km,
                diff,
                f"Estimate − truth {name}",
                cmap="RdBu_r",
                norm=TwoSlopeNorm(vcenter=0.0, vmin=-dlim, vmax=dlim),
            )
            fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.02, label=unit)
            fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.02, label=unit)
            fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.02, label=unit)
        fig.suptitle("Geometry: ground truth vs PINN inference", fontsize=12)
        path = fig_dir / "geometry_truth_estimate_diff.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # --- 5) Velocity components ---
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), constrained_layout=True)
    for row, (comp, truth, est) in enumerate(
        (
            ("u", data["ux"][::s, ::s], data["u_hat"][::s, ::s]),
            ("v", data["uy"][::s, ::s], data["v_hat"][::s, ::s]),
        )
    ):
        truth_m = _masked(truth, mask)
        est_m = _masked(est, mask)
        diff = np.where(mask, est - truth, np.nan)
        vals = np.concatenate([truth[mask], est[mask]])
        vals = vals[np.isfinite(vals)]
        lim = float(np.percentile(np.abs(vals), 98))
        lim = max(lim, 1e-3)
        dlim = float(np.nanpercentile(np.abs(diff[mask]), 98))
        dlim = max(dlim, 1e-3)
        im0 = _add_map(
            axes[row, 0], x_km, y_km, truth_m, f"Truth {comp}", cmap="RdBu_r",
            clim=(-lim, lim),
        )
        im1 = _add_map(
            axes[row, 1], x_km, y_km, est_m, f"PINN {comp}", cmap="RdBu_r",
            clim=(-lim, lim),
        )
        im2 = _add_map(
            axes[row, 2],
            x_km,
            y_km,
            diff,
            f"Estimate − truth {comp}",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-dlim, vmax=dlim),
        )
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.02, label="m/yr")
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.02, label="m/yr")
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.02, label="m/yr")
    fig.suptitle("Velocity components: ground truth vs PINN", fontsize=12)
    path = fig_dir / "velocity_truth_estimate_diff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    gc.collect()
    return saved


def main() -> int:
    args = parse_args()
    t0 = time.perf_counter()
    data = load_data(args.h5, args.npz)
    print(f"load: {time.perf_counter() - t0:.2f}s | grid {data['eta_mean'].shape}")
    print("HDF5:", args.h5)
    print("NPZ:", args.npz)
    print("HDF5 attrs:", data["attrs"])

    metrics = compute_metrics(data)
    print("\n=== eta vs spin-up reference ===")
    for key, val in metrics.items():
        if key == "n_eval_pixels":
            print(f"  {key:22s} {val:,}")
        else:
            print(f"  {key:22s} {val: .4f}")

    if args.no_plots:
        return 0

    t1 = time.perf_counter()
    paths = save_figures(data, metrics, args.fig_dir, args.stride)
    print(f"\nplots: {time.perf_counter() - t1:.2f}s → {args.fig_dir}")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
