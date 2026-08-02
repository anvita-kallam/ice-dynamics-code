#!/usr/bin/env python3
"""Create a MISMIP A=20 velocity-holdout copy of the spin-up grid NPZ.

Copies the full A=20 more_sliding grid, randomly holds out a fraction of valid
velocity pixels (seeded), sets those ux/uy to NaN so Archive VI drops them via
uv_mask, and writes the holdout mask for later evaluation.

Does not modify the original dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "outputs/spinup/production/more_sliding"
    / "SteadyState_more_sliding_10500yr_ramp4000_1refine_grid.npz"
)
DEFAULT_OUT_DIR = ROOT / "outputs/heldout_velocity_validation/dataset"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--holdout-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--tag",
        default=None,
        help="Optional filename tag; default encodes fraction and seed.",
    )
    return p.parse_args()


def valid_velocity_mask(ux: np.ndarray, uy: np.ndarray, geom: np.ndarray) -> np.ndarray:
    return geom & np.isfinite(ux) & np.isfinite(uy)


def geometry_mask(data: np.lib.npyio.NpzFile) -> np.ndarray:
    s = np.asarray(data["s"] if "s" in data.files else data["surface"], dtype=float)
    h = np.asarray(data["h"] if "h" in data.files else data["thickness"], dtype=float)
    b = np.asarray(data["bed"], dtype=float)
    return np.isfinite(s) & np.isfinite(h) & np.isfinite(b)


def main():
    args = parse_args()
    if not (0.0 < args.holdout_fraction < 1.0):
        raise ValueError(f"holdout_fraction must be in (0,1), got {args.holdout_fraction}")
    if not args.source.is_file():
        raise FileNotFoundError(args.source)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"f{args.holdout_fraction:g}_s{args.seed}"
    out_npz = args.output_dir / f"grid_velocity_holdout_{tag}.npz"
    out_mask = args.output_dir / f"holdout_mask_{tag}.npz"
    out_meta = args.output_dir / f"holdout_meta_{tag}.json"

    with np.load(args.source, allow_pickle=False) as src:
        arrays = {k: src[k] for k in src.files}
        geom = geometry_mask(src)
        ux = np.array(src["ux"], dtype=float, copy=True)
        uy = np.array(src["uy"], dtype=float, copy=True)

    valid = valid_velocity_mask(ux, uy, geom)
    valid_idx = np.flatnonzero(valid.ravel())
    n_valid = int(valid_idx.size)
    n_holdout = int(round(args.holdout_fraction * n_valid))
    n_holdout = min(max(n_holdout, 1), n_valid - 1)

    rng = np.random.default_rng(args.seed)
    holdout_flat = rng.choice(valid_idx, size=n_holdout, replace=False)
    holdout_flat.sort()

    holdout_mask = np.zeros(valid.shape, dtype=bool)
    holdout_mask.ravel()[holdout_flat] = True
    train_mask = valid & ~holdout_mask

    ux_h = ux.copy()
    uy_h = uy.copy()
    ux_h[holdout_mask] = np.nan
    uy_h[holdout_mask] = np.nan
    arrays["ux"] = ux_h
    arrays["uy"] = uy_h
    if "velocity" in arrays:
        vel = np.array(arrays["velocity"], dtype=float, copy=True)
        vel[holdout_mask, 0] = np.nan
        vel[holdout_mask, 1] = np.nan
        arrays["velocity"] = vel
    if "speed" in arrays:
        speed = np.array(arrays["speed"], dtype=float, copy=True)
        speed[holdout_mask] = np.nan
        arrays["speed"] = speed

    np.savez_compressed(out_npz, **arrays)
    np.savez_compressed(
        out_mask,
        holdout_mask=holdout_mask,
        train_mask=train_mask,
        valid_mask=valid,
        geom_mask=geom,
        holdout_flat_indices=holdout_flat.astype(np.int64),
        source_npz=str(args.source),
        holdout_fraction=np.array(args.holdout_fraction),
        seed=np.array(args.seed, dtype=np.int64),
    )
    meta = {
        "source_npz": str(args.source),
        "holdout_npz": str(out_npz),
        "holdout_mask_npz": str(out_mask),
        "holdout_fraction": float(args.holdout_fraction),
        "seed": int(args.seed),
        "n_valid_velocity": n_valid,
        "n_holdout": int(holdout_mask.sum()),
        "n_train": int(train_mask.sum()),
        "holdout_fraction_realized": float(holdout_mask.sum() / n_valid),
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))
    print(f"wrote {out_npz}")
    print(f"wrote {out_mask}")
    print(f"wrote {out_meta}")


if __name__ == "__main__":
    main()
