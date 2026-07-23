#!/usr/bin/env python3
"""Package Totten shelf-grid NPZ for Archive sequential VI.

Writes an archive-ready observation product (no viscosity truth, no A/C in
cfg_json), QC maps, and smoke-tests Archive load_snapshot. Basal sliding is
not encoded in the NPZ — use the two Totten VI configs (friction_C).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/real/totten/totten_vel_bed_surf_grid_shelf_2022.npz"
DEFAULT_OUTPUT = ROOT / "data/real/totten/totten_archive_vi_2022.npz"
DEFAULT_SUMMARY = ROOT / "data/real/totten/totten_archive_vi_2022_summary.json"
DEFAULT_FIGDIR = ROOT / "outputs/figures/real/totten"
ARCHIVE_DIR = ROOT / "Archive"
GRID_RESOLUTION = 500.0
S_MINUS_BED_H_TOL = 1.0e-6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--figdir", type=Path, default=DEFAULT_FIGDIR)
    parser.add_argument("--skip-qc", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args()


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def build_coords(xmin: float, xmax: float, ymin: float, ymax: float, ny: int, nx: int):
    """Match Archive load_snapshot: x ascending, y descending (ymax → ymin)."""
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymax, ymin, ny)
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def process(input_path: Path, output_path: Path, summary_path: Path) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    with np.load(input_path) as raw:
        h = np.asarray(raw["h"], dtype=float)
        s = np.asarray(raw["s"], dtype=float)
        bed = np.asarray(raw["bed"], dtype=float)
        ux = np.asarray(raw["ux"], dtype=float)
        uy = np.asarray(raw["uy"], dtype=float)
        ux_err = np.asarray(raw["ux_err"], dtype=float)
        uy_err = np.asarray(raw["uy_err"], dtype=float)
        bed_err = np.asarray(raw["bed_err"], dtype=float)
        surf_err = np.asarray(raw["surf_err"], dtype=float)
        xmin = _scalar(raw["xmin"])
        xmax = _scalar(raw["xmax"])
        ymin = _scalar(raw["ymin"])
        ymax = _scalar(raw["ymax"])

    if h.shape != (318, 174):
        raise ValueError(f"Unexpected Totten shape {h.shape}; expected (318, 174)")

    ny, nx = h.shape
    x, y, X, Y = build_coords(xmin, xmax, ymin, ymax, ny, nx)
    dx = (xmax - xmin) / (nx - 1)
    dy = (ymax - ymin) / (ny - 1)
    if abs(dx - GRID_RESOLUTION) > 1.0e-6 or abs(dy - GRID_RESOLUTION) > 1.0e-6:
        raise ValueError(f"Expected 500 m spacing; got dx={dx}, dy={dy}")

    geom_finite = np.isfinite(h) & np.isfinite(s) & np.isfinite(bed)
    uv_finite = geom_finite & np.isfinite(ux) & np.isfinite(uy)
    resid = (s - (bed + h))[geom_finite]
    max_abs_resid = float(np.max(np.abs(resid))) if resid.size else float("nan")
    if max_abs_resid > S_MINUS_BED_H_TOL:
        raise AssertionError(
            f"s ≈ bed+h failed: max |s-(bed+h)| = {max_abs_resid}"
        )

    speed = np.hypot(ux, uy)
    cfg = {
        "case_id": "totten_2022_shelf",
        "source": str(input_path.relative_to(ROOT)),
        "year": 2022,
        "notes": (
            "Totten shelf-grid observations for Archive VI. "
            "No viscosity truth. Basal sliding is set via friction_C in "
            "Archive/configs/totten VI cfgs (C=100 no_sliding, C=0.001 max_sliding); "
            "A/C intentionally omitted from this cfg_json."
        ),
        "grid_resolution_m": GRID_RESOLUTION,
        "shape_ny_nx": [ny, nx],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        h=h,
        s=s,
        bed=bed,
        ux=ux,
        uy=uy,
        ux_err=ux_err,
        uy_err=uy_err,
        bed_err=bed_err,
        surf_err=surf_err,
        thickness=h,
        surface=s,
        speed=speed,
        x=x,
        y=y,
        X=X,
        Y=Y,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        grid_resolution=np.float64(GRID_RESOLUTION),
        cfg_json=np.asarray(json.dumps(cfg)),
    )

    speed_finite = speed[uv_finite]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path.relative_to(ROOT)),
        "output": str(output_path.relative_to(ROOT)),
        "shape": [ny, nx],
        "bounds_m": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
        "grid_resolution_m": GRID_RESOLUTION,
        "geom_finite_fraction": float(geom_finite.mean()),
        "uv_finite_fraction": float(uv_finite.mean()),
        "geom_finite_count": int(geom_finite.sum()),
        "uv_finite_count": int(uv_finite.sum()),
        "s_minus_bed_h_max_abs": max_abs_resid,
        "speed_m_per_yr": {
            "min": float(np.min(speed_finite)) if speed_finite.size else None,
            "max": float(np.max(speed_finite)) if speed_finite.size else None,
            "median": float(np.median(speed_finite)) if speed_finite.size else None,
        },
        "cfg_json": cfg,
        "sliding_cases": {
            "note": "Set in VI configs, not in this NPZ",
            "no_sliding": {"friction_C": 100.0},
            "max_sliding": {"friction_C": 0.001},
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {output_path}")
    print(f"Wrote {summary_path}")
    print(
        f"geom finite {summary['geom_finite_fraction']:.3%} "
        f"({summary['geom_finite_count']}); "
        f"max |s-(bed+h)| = {max_abs_resid:.2e}"
    )
    return summary


def write_qc(output_path: Path, figdir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    with np.load(output_path) as data:
        X = np.asarray(data["X"], dtype=float)
        Y = np.asarray(data["Y"], dtype=float)
        h = np.asarray(data["h"], dtype=float)
        s = np.asarray(data["s"], dtype=float)
        bed = np.asarray(data["bed"], dtype=float)
        speed = np.asarray(data["speed"], dtype=float)

    geom = np.isfinite(h) & np.isfinite(s) & np.isfinite(bed)
    x_km, y_km = X / 1e3, Y / 1e3
    figdir.mkdir(parents=True, exist_ok=True)

    def save_map(field, title, fname, *, cmap="viridis", norm=None, mask=None):
        plot = np.ma.array(field, mask=~mask if mask is not None else ~np.isfinite(field))
        fig, ax = plt.subplots(figsize=(6.5, 7.5))
        image = ax.pcolormesh(x_km, y_km, plot, shading="auto", cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        out = figdir / fname
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}")

    pos_speed = speed[np.isfinite(speed) & (speed > 0)]
    speed_norm = None
    if pos_speed.size:
        speed_norm = LogNorm(
            vmin=max(float(np.percentile(pos_speed, 2)), 1e-2),
            vmax=float(np.percentile(pos_speed, 99.5)),
        )

    save_map(speed, "Totten speed (m/yr)", "speed.png", cmap="magma", norm=speed_norm, mask=geom)
    save_map(h, "Totten thickness (m)", "thickness.png", cmap="Blues", mask=geom)
    save_map(bed, "Totten bed (m)", "bed.png", cmap="terrain", mask=geom)
    save_map(s, "Totten surface (m)", "surface.png", cmap="cividis", mask=geom)
    save_map(geom.astype(float), "Finite geometry mask", "finite_mask.png", cmap="gray", mask=None)


def _cfg_friction_C(cfg_path: Path) -> float:
    import ast
    import configparser

    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.read(cfg_path)
    return float(ast.literal_eval(cfg["prior"]["friction_C"]))


def smoke_test(output_path: Path) -> None:
    """Validate Archive-compatible fields without requiring torch.

    Mirrors the mask / coordinate construction in Archive.utilities_torch.load_snapshot
    and checks that cfg_json does not carry A/C (so VI friction_C stays cfg-driven).
    """
    with np.load(output_path, allow_pickle=True) as data:
        required = (
            "ux", "uy", "h", "s", "bed", "ux_err", "uy_err", "bed_err", "surf_err",
            "xmin", "xmax", "ymin", "ymax", "x", "y", "X", "Y", "cfg_json",
        )
        missing = [k for k in required if k not in data.files]
        if missing:
            raise AssertionError(f"archive NPZ missing keys: {missing}")
        if "viscosity" in data.files:
            raise AssertionError("Totten product must not include viscosity truth")

        cfg = json.loads(str(data["cfg_json"]))
        if "A" in cfg or "C" in cfg:
            raise AssertionError(
                f"cfg_json must omit A/C so VI friction_C wins; got keys {sorted(cfg)}"
            )

        u = np.asarray(data["ux"], dtype=float)
        v = np.asarray(data["uy"], dtype=float)
        s = np.asarray(data["s"], dtype=float)
        h = np.asarray(data["h"], dtype=float)
        b = np.asarray(data["bed"], dtype=float)
        xmin, xmax = _scalar(data["xmin"]), _scalar(data["xmax"])
        ymin, ymax = _scalar(data["ymin"]), _scalar(data["ymax"])
        nx, ny = u.shape[1], u.shape[0]
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymax, ymin, ny)
        if not np.allclose(x, np.asarray(data["x"], dtype=float)):
            raise AssertionError("stored x does not match load_snapshot linspace")
        if not np.allclose(y, np.asarray(data["y"], dtype=float)):
            raise AssertionError("stored y does not match load_snapshot linspace")

        geom_mask = np.isfinite(s) & np.isfinite(h) & np.isfinite(b)
        uv_mask = geom_mask & np.isfinite(u) & np.isfinite(v)
        u_finite = u[uv_mask]
        print(
            "smoke archive fields: "
            f"shape={(ny, nx)}, geom_mask={int(geom_mask.sum())}, "
            f"uv_mask={int(uv_mask.sum())}, "
            f"u range=[{float(np.min(u_finite)):.3g}, {float(np.max(u_finite)):.3g}], "
            "viscosity=False"
        )

    # Prefer full Archive loader when torch is available (e.g. DSI pytorch env).
    try:
        sys.path.insert(0, str(ARCHIVE_DIR))
        from utilities_torch import ParameterClass, load_snapshot  # noqa: WPS433
    except ImportError as exc:
        print(f"smoke skip utilities_torch import ({exc})")
    else:
        pars = ParameterClass(str(ARCHIVE_DIR / "configs/totten/run_torch_pretrain_totten.cfg"))
        pars.data.h5file = str(output_path)
        before_C = float(pars.prior.friction_C)
        snap = load_snapshot(pars.data.h5file, pars)
        after_C = float(pars.prior.friction_C)
        if abs(after_C - before_C) > 1.0e-12:
            raise AssertionError(
                f"cfg_json must not override friction_C; before={before_C}, after={after_C}"
            )
        print(
            "smoke load_snapshot: "
            f"shape={snap.shape}, geom_mask={int(snap.geom_mask.sum())}, "
            f"friction_C={after_C}"
        )

    for name, expected_C in (
        ("run_torch_vi_only_totten_no_sliding.cfg", 100.0),
        ("run_torch_vi_only_totten_max_sliding.cfg", 0.001),
    ):
        path = ARCHIVE_DIR / "configs/totten" / name
        if not path.is_file():
            print(f"smoke skip missing cfg: {path}")
            continue
        got = _cfg_friction_C(path)
        if abs(got - expected_C) > 1.0e-12:
            raise AssertionError(f"{name}: friction_C={got}, expected {expected_C}")
        print(f"smoke cfg {name}: friction_C={got}")


def main():
    args = parse_args()
    process(args.input, args.output, args.summary)
    if not args.skip_qc:
        write_qc(args.output, args.figdir)
    if not args.skip_smoke:
        smoke_test(args.output)


if __name__ == "__main__":
    main()
