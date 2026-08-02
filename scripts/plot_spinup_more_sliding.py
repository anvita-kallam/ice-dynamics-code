#!/usr/bin/env python3
"""Plot the production more_sliding spin-up NPZ fields and profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    ROOT
    / "outputs/spinup/production/more_sliding/"
    "SteadyState_more_sliding_10500yr_ramp4000_1refine_grid.npz"
)
DEFAULT_OUT = ROOT / "outputs/figures/production/more_sliding/npz_summary"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def add_map(ax, x_km, y_km, field, title, *, cmap="viridis", norm=None, xaxis_top=False):
    image = ax.pcolormesh(
        x_km, y_km, field, shading="auto", cmap=cmap, norm=norm)
    ax.set_ylabel("y (km)")
    if xaxis_top:
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.xaxis.set_label_position("top")
        # Keep title above the top spine; nudge slightly so it clears ticks.
        ax.set_title(title, fontsize=10, pad=18)
    else:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (km)")
    # Match axes box to the wide MISMIP domain so equal-aspect maps
    # don't leave empty vertical bands beside/above the ice strip.
    x0, x1 = float(np.nanmin(x_km)), float(np.nanmax(x_km))
    y0, y1 = float(np.nanmin(y_km)), float(np.nanmax(y_km))
    ax.set_box_aspect((y1 - y0) / max(x1 - x0, 1e-12))
    return image


def add_colorbar(fig, image, ax, label, *, pad=0.45):
    """Horizontal colorbar under each panel.

    ``pad`` is in inches (make_axes_locatable). Prefer inches over axes-fraction
    pads: with set_box_aspect on a wide domain the axes are very short, so a
    fractional pad collapses and the bar covers tick labels.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="28%", pad=pad)
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.set_label(label)
    cbar.ax.tick_params(labelsize=8)
    return cbar


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
        x = data["x"].astype(float)
        y = data["y"].astype(float)
        x_grid = data["X"].astype(float)
        y_grid = data["Y"].astype(float)
        thickness = data["thickness"].astype(float)
        surface = data["surface"].astype(float)
        bed = data["bed"].astype(float)
        ux = data["ux"].astype(float)
        uy = data["uy"].astype(float)
        speed = np.hypot(ux, uy)
        viscosity = data["viscosity"].astype(float)
        haf = data["height_above_flotation"].astype(float)
        cfg = json.loads(str(data["cfg_json"].item()))

    x_km, y_km = x_grid / 1e3, y_grid / 1e3
    positive_speed = speed[np.isfinite(speed) & (speed > 0)]
    positive_eta = viscosity[np.isfinite(viscosity) & (viscosity > 0)]
    speed_norm = LogNorm(
        vmin=max(float(np.percentile(positive_speed, 2)), 1e-2),
        vmax=float(np.percentile(positive_speed, 99.5)),
    )
    eta_norm = LogNorm(
        vmin=float(np.percentile(positive_eta, 2)),
        vmax=float(np.percentile(positive_eta, 98)),
    )

    # Surface: warm topographic ramp anchored at its own range.
    surface_norm = plt.Normalize(
        vmin=float(np.percentile(surface, 1)),
        vmax=float(np.percentile(surface, 99)),
    )
    # Bed: diverging around sea level — blue below 0, brown above.
    bed_limit = max(abs(float(bed.min())), abs(float(bed.max())))
    bed_norm = TwoSlopeNorm(vcenter=0.0, vmin=-bed_limit, vmax=bed_limit)

    # Six-field overview.
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 5.2))
    panels = (
        (thickness, "Ice thickness", "Blues", None, "m"),
        (surface, "Surface elevation", "gist_earth", surface_norm, "m"),
        (bed, "Bed elevation", "BrBG_r", bed_norm, "m"),
        (speed, "Depth-averaged speed", "plasma", speed_norm, "m/yr"),
        (viscosity, "Effective viscosity η", "magma", eta_norm, "MPa·yr"),
        (haf, "Height above flotation", "RdBu_r", None, "m"),
    )
    for row_idx, row in enumerate(axes):
        for ax, (field, title, cmap, norm, unit) in zip(row, panels[row_idx * 3:(row_idx + 1) * 3]):
            if title == "Height above flotation":
                limit = max(
                    abs(float(np.percentile(haf, 2))),
                    abs(float(np.percentile(haf, 98))),
                )
                norm = TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)
            image = add_map(
                ax, x_km, y_km, field, title,
                cmap=cmap, norm=norm, xaxis_top=(row_idx == 0))
            # Top row: x-axis is above the map, so the colorbar can sit closer.
            # Top row: x-axis above map → tighter inch pad. Bottom: room for x ticks.
            add_colorbar(fig, image, ax, unit, pad=0.18 if row_idx == 0 else 0.50)
    fig.suptitle(
        "Production more_sliding spin-up: final state"
        f"  |  C={cfg.get('C', 'unknown')}, A={cfg.get('A', 'unknown')}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    overview_path = args.output_dir / "spinup_final_fields.png"
    fig.savefig(overview_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Velocity components and flow direction.
    velocity_limit = float(np.percentile(np.abs(np.concatenate([ux.ravel(), uy.ravel()])), 99))
    velocity_limit = max(velocity_limit, 1.0)
    velocity_norm = TwoSlopeNorm(
        vcenter=0.0, vmin=-velocity_limit, vmax=velocity_limit)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 4.8))
    image = add_map(
        axes[0, 0], x_km, y_km, speed, "Speed magnitude",
        cmap="plasma", norm=speed_norm)
    add_colorbar(fig, image, axes[0, 0], "m/yr")
    for ax, field, title in (
        (axes[0, 1], ux, "Along-flow velocity u"),
        (axes[1, 0], uy, "Cross-flow velocity v"),
    ):
        image = add_map(
            ax, x_km, y_km, field, title,
            cmap="RdBu_r", norm=velocity_norm)
        add_colorbar(fig, image, ax, "m/yr")
    image = add_map(
        axes[1, 1], x_km, y_km, thickness,
        "Velocity direction over thickness", cmap="Blues")
    add_colorbar(fig, image, axes[1, 1], "m")
    step_y = max(len(y) // 16, 1)
    step_x = max(len(x) // 48, 1)
    sampled_speed = np.hypot(
        ux[::step_y, ::step_x], uy[::step_y, ::step_x])
    axes[1, 1].quiver(
        x_km[::step_y, ::step_x],
        y_km[::step_y, ::step_x],
        ux[::step_y, ::step_x] / np.maximum(sampled_speed, 1e-12),
        uy[::step_y, ::step_x] / np.maximum(sampled_speed, 1e-12),
        color="black", alpha=0.7, pivot="mid", scale=55,
    )
    fig.tight_layout()
    velocity_path = args.output_dir / "spinup_velocity.png"
    fig.savefig(velocity_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Centerline profiles.
    center_row = int(np.argmin(np.abs(y - 0.5 * (y.min() + y.max()))))
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    axes[0].plot(x / 1e3, surface[center_row], label="surface")
    axes[0].plot(x / 1e3, bed[center_row], label="bed")
    axes[0].fill_between(
        x / 1e3, bed[center_row], surface[center_row],
        color="tab:blue", alpha=0.18, label="ice")
    axes[0].set_ylabel("Elevation (m)")
    axes[0].legend(ncol=3)
    axes[0].set_title(f"Centerline profiles at y={y[center_row] / 1e3:.1f} km")

    axes[1].plot(x / 1e3, thickness[center_row], color="tab:blue")
    axes[1].set_ylabel("Thickness (m)")
    axes[2].semilogy(
        x / 1e3, np.maximum(speed[center_row], 1e-3),
        color="tab:orange")
    axes[2].set_ylabel("Speed (m/yr)")
    axes[3].plot(x / 1e3, viscosity[center_row], color="tab:red")
    axes[3].set_ylabel("η (MPa·yr)")
    axes[3].set_xlabel("x (km)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    profiles_path = args.output_dir / "spinup_centerline_profiles.png"
    fig.savefig(profiles_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "source": str(args.npz),
        "shape": list(thickness.shape),
        "C": cfg.get("C"),
        "A": cfg.get("A"),
        "mean_thickness_m": float(np.mean(thickness)),
        "min_thickness_m": float(np.min(thickness)),
        "max_thickness_m": float(np.max(thickness)),
        "mean_speed_m_per_yr": float(np.mean(speed)),
        "max_speed_m_per_yr": float(np.max(speed)),
        "mean_viscosity_mpa_yr": float(np.mean(viscosity)),
        "min_viscosity_mpa_yr": float(np.min(viscosity)),
        "max_viscosity_mpa_yr": float(np.max(viscosity)),
        "grounded_fraction_haf_positive": float(np.mean(haf > 0.0)),
    }
    summary_path = args.output_dir / "spinup_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    for path in (overview_path, velocity_path, profiles_path, summary_path):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
