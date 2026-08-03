#!/usr/bin/env python3
"""Exploratory geometry plots for the Totten Archive VI product."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = ROOT / "data/real/totten/totten_archive_vi_2022.npz"
DEFAULT_OUT = ROOT / "outputs/figures/real/totten/eda"
RHO_ICE = 917.0
RHO_WATER = 1028.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def height_above_flotation(surface: np.ndarray, thickness: np.ndarray) -> np.ndarray:
    """HAF = s - (1 - ρ_i/ρ_w) h; >0 grounded, ≈0 floating."""
    return surface - (1.0 - RHO_ICE / RHO_WATER) * thickness


def add_map(ax, x_km, y_km, field, title, *, cmap="viridis", norm=None):
    image = ax.pcolormesh(
        x_km, y_km, field, shading="auto", cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal")
    return image


def percentile_norm(values, lo=2, hi=98):
    import matplotlib.pyplot as plt

    finite = values[np.isfinite(values)]
    return plt.Normalize(
        vmin=float(np.percentile(finite, lo)),
        vmax=float(np.percentile(finite, hi)),
    )


def main():
    args = parse_args()
    if not args.npz.exists():
        raise FileNotFoundError(args.npz)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    with np.load(args.npz) as data:
        x = np.asarray(data["x"], dtype=float)
        y = np.asarray(data["y"], dtype=float)
        X = np.asarray(data["X"], dtype=float)
        Y = np.asarray(data["Y"], dtype=float)
        h = np.asarray(data["h"], dtype=float)
        s = np.asarray(data["s"], dtype=float)
        bed = np.asarray(data["bed"], dtype=float)
        ux = np.asarray(data["ux"], dtype=float)
        uy = np.asarray(data["uy"], dtype=float)
        ux_err = np.asarray(data["ux_err"], dtype=float)
        uy_err = np.asarray(data["uy_err"], dtype=float)
        bed_err = np.asarray(data["bed_err"], dtype=float)
        surf_err = np.asarray(data["surf_err"], dtype=float)

    speed = np.hypot(ux, uy)
    geom = np.isfinite(h) & np.isfinite(s) & np.isfinite(bed)
    uv = geom & np.isfinite(ux) & np.isfinite(uy)
    haf = np.where(geom, height_above_flotation(s, h), np.nan)
    grounded = geom & (haf > 0.0)
    floating = geom & (haf <= 0.0)
    # In this Totten product bed == s - h exactly (ice base, not ocean floor).
    draft = np.where(geom, s - h, np.nan)
    bed_is_ice_base = bool(np.allclose(bed[geom], draft[geom]))

    # Surface slope magnitude from finite differences on the 500 m grid.
    dy = float(np.abs(y[1] - y[0])) if len(y) > 1 else 500.0
    dx = float(np.abs(x[1] - x[0])) if len(x) > 1 else 500.0
    s_fill = np.where(geom, s, np.nanmedian(s[geom]))
    ds_dy, ds_dx = np.gradient(s_fill, dy, dx)
    surface_slope = np.where(geom, np.hypot(ds_dx, ds_dy), np.nan)

    x_km, y_km = X / 1e3, Y / 1e3
    h_m = np.ma.array(h, mask=~geom)
    s_m = np.ma.array(s, mask=~geom)
    bed_m = np.ma.array(bed, mask=~geom)
    speed_m = np.ma.array(speed, mask=~uv)
    haf_m = np.ma.array(haf, mask=~geom)
    draft_m = np.ma.array(draft, mask=~geom)
    slope_m = np.ma.array(surface_slope, mask=~geom)

    pos_speed = speed[uv & (speed > 0)]
    speed_norm = LogNorm(
        vmin=max(float(np.percentile(pos_speed, 2)), 1e-1),
        vmax=float(np.percentile(pos_speed, 99.5)),
    )
    haf_lim = max(
        abs(float(np.percentile(haf[geom], 2))),
        abs(float(np.percentile(haf[geom], 98))),
        1.0,
    )
    haf_norm = TwoSlopeNorm(vcenter=0.0, vmin=-haf_lim, vmax=haf_lim)

    # --- 1. Geometry overview -------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 10.5), constrained_layout=True)
    panels = (
        (axes[0, 0], h_m, "Ice thickness", "Blues", percentile_norm(h[geom]), "m"),
        (axes[0, 1], s_m, "Surface elevation", "cividis", percentile_norm(s[geom]), "m"),
        (axes[0, 2], bed_m, "Bed elevation", "terrain", percentile_norm(bed[geom]), "m"),
        (axes[1, 0], draft_m, "Ice base / bed (s − h)", "viridis", percentile_norm(draft[geom]), "m"),
        (axes[1, 1], haf_m, "Height above flotation", "RdBu_r", haf_norm, "m"),
        (
            axes[1, 2],
            slope_m,
            "Surface slope |∇s|",
            "magma",
            percentile_norm(surface_slope[geom], lo=5, hi=99),
            "m/m",
        ),
    )
    for ax, field, title, cmap, norm, unit in panels:
        image = add_map(ax, x_km, y_km, field, title, cmap=cmap, norm=norm)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label=unit)
    fig.suptitle(
        "Totten 2022 shelf grid — geometry "
        f"({int(geom.sum()):,} finite cells, {geom.mean():.1%} of domain)",
        fontsize=13,
    )
    overview_path = args.output_dir / "geometry_overview.png"
    fig.savefig(overview_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- 1b. Geometry + velocity (4×2 / 2 rows × 4 cols) ----------------------
    fig, axes = plt.subplots(2, 4, figsize=(18.5, 9.2), constrained_layout=True)
    panels_gv = (
        (axes[0, 0], h_m, "Ice thickness", "Blues", percentile_norm(h[geom]), "m"),
        (axes[0, 1], s_m, "Surface elevation", "cividis", percentile_norm(s[geom]), "m"),
        (axes[0, 2], bed_m, "Bed elevation", "terrain", percentile_norm(bed[geom]), "m"),
        (axes[0, 3], draft_m, "Ice base (s − h)", "viridis", percentile_norm(draft[geom]), "m"),
        (axes[1, 0], haf_m, "Height above flotation", "RdBu_r", haf_norm, "m"),
        (
            axes[1, 1],
            slope_m,
            "Surface slope |∇s|",
            "magma",
            percentile_norm(surface_slope[geom], lo=5, hi=99),
            "m/m",
        ),
        (axes[1, 2], speed_m, "Observed speed", "magma", speed_norm, "m/yr"),
    )
    for ax, field, title, cmap, norm, unit in panels_gv:
        image = add_map(ax, x_km, y_km, field, title, cmap=cmap, norm=norm)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label=unit)
    ax = axes[1, 3]
    image = add_map(
        ax, x_km, y_km, h_m, "Thickness + velocity",
        cmap="Blues", norm=percentile_norm(h[geom]),
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label="m")
    step_y = max(h.shape[0] // 18, 1)
    step_x = max(h.shape[1] // 12, 1)
    u_s = np.where(uv, ux, np.nan)[::step_y, ::step_x]
    v_s = np.where(uv, uy, np.nan)[::step_y, ::step_x]
    sp_s = np.hypot(u_s, v_s)
    ax.quiver(
        x_km[::step_y, ::step_x],
        y_km[::step_y, ::step_x],
        u_s / np.maximum(sp_s, 1e-12),
        v_s / np.maximum(sp_s, 1e-12),
        color="k",
        alpha=0.65,
        pivot="mid",
        scale=45,
    )
    fig.suptitle(
        "Totten 2022 shelf grid — geometry & velocity "
        f"({int(geom.sum()):,} finite cells, {geom.mean():.1%} of domain)",
        fontsize=13,
    )
    gv_path = args.output_dir / "geometry_velocity_overview.png"
    fig.savefig(gv_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- 2. Grounding / flotation mask + speed context ------------------------
    regime = np.full(h.shape, np.nan)
    regime[floating] = 0.0
    regime[grounded] = 1.0
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.8), constrained_layout=True)
    image = add_map(
        axes[0], x_km, y_km, np.ma.array(regime, mask=~geom),
        "Flotation regime\n(0=floating, 1=grounded)", cmap="coolwarm",
        norm=plt.Normalize(0, 1),
    )
    fig.colorbar(image, ax=axes[0], fraction=0.046, pad=0.02, ticks=[0, 1])
    image = add_map(
        axes[1], x_km, y_km, speed_m, "Observed speed", cmap="magma", norm=speed_norm)
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.02, label="m/yr")
    image = add_map(
        axes[2], x_km, y_km, h_m, "Thickness with velocity arrows", cmap="Blues",
        norm=percentile_norm(h[geom]),
    )
    fig.colorbar(image, ax=axes[2], fraction=0.046, pad=0.02, label="m")
    step_y = max(h.shape[0] // 18, 1)
    step_x = max(h.shape[1] // 12, 1)
    u_s = np.where(uv, ux, np.nan)[::step_y, ::step_x]
    v_s = np.where(uv, uy, np.nan)[::step_y, ::step_x]
    sp_s = np.hypot(u_s, v_s)
    axes[2].quiver(
        x_km[::step_y, ::step_x],
        y_km[::step_y, ::step_x],
        u_s / np.maximum(sp_s, 1e-12),
        v_s / np.maximum(sp_s, 1e-12),
        color="k",
        alpha=0.65,
        pivot="mid",
        scale=45,
    )
    fig.suptitle("Totten grounding vs flow", fontsize=13)
    regime_path = args.output_dir / "grounding_and_flow.png"
    fig.savefig(regime_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- 3. Uncertainty maps --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10), constrained_layout=True)
    err_panels = (
        (axes[0, 0], np.ma.array(surf_err, mask=~geom), "Surface error", "m"),
        (axes[0, 1], np.ma.array(bed_err, mask=~geom), "Bed error", "m"),
        (axes[1, 0], np.ma.array(ux_err, mask=~uv), "ux error", "m/yr"),
        (axes[1, 1], np.ma.array(uy_err, mask=~uv), "uy error", "m/yr"),
    )
    for ax, field, title, unit in err_panels:
        image = add_map(ax, x_km, y_km, field, title, cmap="YlOrRd")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label=unit)
    fig.suptitle("Totten observation uncertainties", fontsize=13)
    err_path = args.output_dir / "observation_errors.png"
    fig.savefig(err_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- 4. Profiles through fastest and thickest ice -------------------------
    # Prefer a transect along the column with max speed (typical shelf flow axis).
    speed_safe = np.where(uv, speed, -np.inf)
    i_fast, j_fast = np.unravel_index(int(np.argmax(speed_safe)), speed_safe.shape)
    i_thick, j_thick = np.unravel_index(
        int(np.argmax(np.where(geom, h, -np.inf))), h.shape)
    i_fast_row = i_fast

    def plot_transect(ax_row, row_idx, title):
        mask = geom[row_idx]
        xx = x / 1e3
        ax_row[0].plot(xx[mask], s[row_idx][mask], color="tab:orange", label="surface")
        ax_row[0].plot(
            xx[mask], draft[row_idx][mask], color="tab:brown", label="ice base (= bed)")
        ax_row[0].fill_between(
            xx[mask],
            draft[row_idx][mask],
            s[row_idx][mask],
            color="tab:blue",
            alpha=0.25,
            label="ice",
        )
        ax_row[0].axhline(0.0, color="k", lw=0.6, alpha=0.5)
        ax_row[0].set_ylabel("Elevation (m)")
        ax_row[0].set_title(title)
        ax_row[0].legend(fontsize=8, ncol=3, loc="best")
        ax_row[0].grid(True, alpha=0.25)

        ax_row[1].plot(xx[mask], h[row_idx][mask], color="tab:blue")
        ax_row[1].set_ylabel("Thickness (m)")
        ax_row[1].grid(True, alpha=0.25)

        umask = uv[row_idx]
        ax_row[2].plot(xx[umask], speed[row_idx][umask], color="tab:red")
        ax_row[2].set_ylabel("Speed (m/yr)")
        ax_row[2].set_xlabel("x (km)")
        ax_row[2].grid(True, alpha=0.25)

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex="col", constrained_layout=True)
    plot_transect(
        axes[:, 0],
        i_fast_row,
        f"Transect through max speed (row y={y[i_fast_row]/1e3:.1f} km)",
    )
    plot_transect(
        axes[:, 1],
        i_thick,
        f"Transect through max thickness (row y={y[i_thick]/1e3:.1f} km)",
    )
    profiles_path = args.output_dir / "geometry_profiles.png"
    fig.savefig(profiles_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- 5. Histograms --------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    hist_specs = (
        (h[geom], "Thickness (m)", axes[0, 0]),
        (s[geom], "Surface (m)", axes[0, 1]),
        (bed[geom], "Bed (m)", axes[0, 2]),
        (haf[geom], "HAF (m)", axes[1, 0]),
        (speed[uv], "Speed (m/yr)", axes[1, 1]),
        (surface_slope[geom], "Surface slope |∇s| (m/m)", axes[1, 2]),
    )
    for values, title, ax in hist_specs:
        ax.hist(values, bins=60, color="steelblue", alpha=0.85, edgecolor="none")
        ax.set_title(title)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
    axes[1, 0].axvline(0.0, color="k", lw=1.0)
    fig.suptitle("Totten geometry / speed distributions (finite cells)", fontsize=13)
    hist_path = args.output_dir / "geometry_histograms.png"
    fig.savefig(hist_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    def stats(arr, mask):
        v = arr[mask]
        return {
            "count": int(mask.sum()),
            "min": float(np.min(v)),
            "p05": float(np.percentile(v, 5)),
            "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
            "max": float(np.max(v)),
            "mean": float(np.mean(v)),
        }

    summary = {
        "source": str(args.npz.relative_to(ROOT)),
        "shape_ny_nx": list(h.shape),
        "domain_km": {
            "Lx": float((x.max() - x.min()) / 1e3),
            "Ly": float((y.max() - y.min()) / 1e3),
            "resolution_m": 500.0,
        },
        "geom_finite_fraction": float(geom.mean()),
        "floating_fraction_of_geom": float(floating.sum() / geom.sum()),
        "grounded_fraction_of_geom": float(grounded.sum() / geom.sum()),
        "bed_equals_ice_base": bed_is_ice_base,
        "notes": (
            "bed == s - h everywhere on finite cells (ice-base elevation, "
            "not ocean-floor bathymetry; no cavity thickness in this product). "
            "HAF uses hydrostatic freeboard from s and h only."
        ),
        "thickness_m": stats(h, geom),
        "surface_m": stats(s, geom),
        "bed_m": stats(bed, geom),
        "haf_m": stats(haf, geom),
        "speed_m_per_yr": stats(speed, uv),
        "surface_slope": stats(surface_slope, geom),
        "errors": {
            "surf_err_median_m": float(np.nanmedian(surf_err[geom])),
            "bed_err_median_m": float(np.nanmedian(bed_err[geom])),
            "ux_err_median_m_per_yr": float(np.nanmedian(ux_err[uv])),
            "uy_err_median_m_per_yr": float(np.nanmedian(uy_err[uv])),
        },
        "transects": {
            "max_speed_row_y_km": float(y[i_fast_row] / 1e3),
            "max_speed_m_per_yr": float(speed_safe[i_fast_row, j_fast]),
            "max_thickness_row_y_km": float(y[i_thick] / 1e3),
            "max_thickness_m": float(h[i_thick, j_thick]),
        },
        "figures": [
            str(p.relative_to(ROOT))
            for p in (
                overview_path,
                gv_path,
                regime_path,
                err_path,
                profiles_path,
                hist_path,
            )
        ],
    }
    summary_path = args.output_dir / "eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    for path in summary["figures"] + [str(summary_path.relative_to(ROOT))]:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
