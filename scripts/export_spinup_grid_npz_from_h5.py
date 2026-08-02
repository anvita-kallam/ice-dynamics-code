#!/usr/bin/env python3
"""Export *_grid.npz from an existing Firedrake steady-state checkpoint.

Usage:
  export PATH="$HOME/firedrake-env/bin:$PATH"
  python scripts/export_spinup_grid_npz_from_h5.py \\
    outputs/spinup/production/more_sliding_A40/SteadyState_more_sliding_A40_10500yr_ramp4000_1refine.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import firedrake
from firedrake import Constant, PointNotInDomainError, grad, inner, sqrt, sym
from icepack.constants import glen_flow_law as n


def effective_viscosity(velocity, fluidity):
    eps = sym(grad(velocity))
    eps_eff = sqrt(
        0.5 * (inner(eps, eps) + firedrake.tr(eps) ** 2) + Constant(1e-30)
    )
    return 0.5 * fluidity ** (-1 / n) * eps_eff ** (1 / n - 1)


def evaluate_function_on_grid(function, pts, values, *, block=50_000):
    for i0 in range(0, pts.shape[0], block):
        i1 = min(i0 + block, pts.shape[0])
        p = pts[i0:i1]
        try:
            values[i0:i1] = function.at(p)
        except PointNotInDomainError:
            for j, point in enumerate(p):
                try:
                    values[i0 + j] = function.at(point)
                except PointNotInDomainError:
                    pass
    return values


def export_grid_npz(json_path: Path) -> Path:
    json_path = json_path.resolve()
    with json_path.open() as f:
        cfg = json.load(f)
    cfg["save_dir"] = str(json_path.parent)
    stem = cfg.get("output_stem", json_path.stem)
    h5_path = json_path.with_suffix(".h5")
    npz_path = json_path.parent / f"{stem}_grid.npz"

    with firedrake.CheckpointFile(str(h5_path), "r") as checkpoint:
        mesh = checkpoint.load_mesh()
        u = checkpoint.load_function(mesh, name="velocity")
        h = checkpoint.load_function(mesh, name="thickness")
        s = checkpoint.load_function(mesh, name="surface")
        b = checkpoint.load_function(mesh, name="bed")
        A_field = checkpoint.load_function(mesh, name="A")

    Q = h.function_space()
    mesh_coords = mesh.coordinates.dat.data_ro
    xmin, ymin = mesh_coords.min(axis=0)
    xmax, ymax = mesh_coords.max(axis=0)
    resolution = float(cfg["grid_resolution"])
    x = np.arange(xmin, xmax + 0.5 * resolution, resolution)
    y = np.arange(ymin, ymax + 0.5 * resolution, resolution)
    x = x[x <= xmax + 1e-8]
    y = y[y <= ymax + 1e-8]
    X, Y = np.meshgrid(x, y, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    n_pts = pts.shape[0]

    arrays = {
        name: np.full(n_pts if name != "u" else (n_pts, 2), np.nan, dtype=float)
        for name in ("h", "s", "b", "A", "speed", "haf", "viscosity", "u")
    }
    speed = firedrake.Function(Q, name="speed").interpolate(sqrt(inner(u, u)))
    haf = firedrake.Function(Q, name="height_above_flotation").interpolate(
        s - (1 - firedrake.Constant(917.0) / firedrake.Constant(1028.0)) * h
    )
    eta = firedrake.project(effective_viscosity(velocity=u, fluidity=A_field), Q)

    evaluate_function_on_grid(h, pts, arrays["h"])
    evaluate_function_on_grid(s, pts, arrays["s"])
    evaluate_function_on_grid(b, pts, arrays["b"])
    evaluate_function_on_grid(A_field, pts, arrays["A"])
    evaluate_function_on_grid(speed, pts, arrays["speed"])
    evaluate_function_on_grid(haf, pts, arrays["haf"])
    evaluate_function_on_grid(eta, pts, arrays["viscosity"])
    evaluate_function_on_grid(u, pts, arrays["u"])

    ny, nx = len(y), len(x)
    U = arrays["u"].reshape(ny, nx, 2)
    np.savez_compressed(
        npz_path,
        x=x,
        y=y,
        X=X,
        Y=Y,
        h=arrays["h"].reshape(ny, nx),
        thickness=arrays["h"].reshape(ny, nx),
        s=arrays["s"].reshape(ny, nx),
        surface=arrays["s"].reshape(ny, nx),
        bed=arrays["b"].reshape(ny, nx),
        ux=U[..., 0],
        uy=U[..., 1],
        velocity=U,
        speed=arrays["speed"].reshape(ny, nx),
        A=arrays["A"].reshape(ny, nx),
        A_inv=arrays["A"].reshape(ny, nx),
        viscosity=arrays["viscosity"].reshape(ny, nx),
        height_above_flotation=arrays["haf"].reshape(ny, nx),
        xmin=float(xmin),
        xmax=float(xmax),
        ymin=float(ymin),
        ymax=float(ymax),
        grid_resolution=resolution,
        cfg_json=json.dumps(cfg, sort_keys=True),
    )
    print(f"Saved 2D gridded steady-state NPZ: {npz_path} (ny={ny}, nx={nx})")
    return npz_path


def main(argv):
    if not argv:
        print(__doc__, file=sys.stderr)
        return 1
    export_grid_npz(Path(argv[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
