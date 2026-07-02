#!/usr/bin/env python3
"""Build VI-ready datasets from production spin-up NPZ ground-truth files.

Reads the two production `_grid.npz` outputs (more sliding / no sliding),
adds masks, log-viscosity targets, and synthetic noisy velocity observations,
then writes per-case bundles plus a manifest JSON for the VI framework.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from project_paths import PRODUCTION_CASES, PROJECT_ROOT, VI_DIR

DEFAULT_CASES = {
    case_id: {
        "source_npz": spec["grid_npz"],
        "sliding_regime": spec["sliding_regime"],
    }
    for case_id, spec in PRODUCTION_CASES.items()
}

VISCOSITY_FLOOR = 1.0e-3


@dataclass(frozen=True)
class NoiseConfig:
    relative_speed_sigma: float = 0.02
    seed: int = 42


def read_jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): read_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [read_jsonable(v) for v in value]
    return value


def load_source_case(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source NPZ: {path}")

    data = np.load(path)
    cfg = json.loads(str(data["cfg_json"]))

    speed = np.asarray(data["speed"], dtype=float)
    ux = np.asarray(data["ux"], dtype=float)
    uy = np.asarray(data["uy"], dtype=float)
    viscosity = np.asarray(data["viscosity"], dtype=float)
    haf = np.asarray(data["height_above_flotation"], dtype=float)

    valid_mask = (
        np.isfinite(speed)
        & np.isfinite(ux)
        & np.isfinite(uy)
        & np.isfinite(viscosity)
        & np.isfinite(haf)
    )
    grounded_mask = valid_mask & (haf > 0.0)
    floating_mask = valid_mask & (haf <= 0.0)

    viscosity_true = np.maximum(viscosity, VISCOSITY_FLOOR)
    log_viscosity_true = np.log(viscosity_true)

    return {
        "source_npz": path,
        "cfg": cfg,
        "x": np.asarray(data["x"], dtype=float),
        "y": np.asarray(data["y"], dtype=float),
        "X": np.asarray(data["X"], dtype=float),
        "Y": np.asarray(data["Y"], dtype=float),
        "bed": np.asarray(data["bed"], dtype=float),
        "thickness_true": np.asarray(data["thickness"], dtype=float),
        "surface_true": np.asarray(data["surface"], dtype=float),
        "speed_true": speed,
        "ux_true": ux,
        "uy_true": uy,
        "viscosity_true": viscosity_true,
        "log_viscosity_true": log_viscosity_true,
        "height_above_flotation": haf,
        "A_true": np.asarray(data["A"], dtype=float),
        "valid_mask": valid_mask.astype(np.uint8),
        "grounded_mask": grounded_mask.astype(np.uint8),
        "floating_mask": floating_mask.astype(np.uint8),
        "grid_resolution": float(data["grid_resolution"]),
    }


def add_velocity_noise(fields: dict, noise: NoiseConfig, rng: np.random.Generator) -> dict:
    """Gaussian noise on speed observations, applied consistently to ux/uy."""
    speed = fields["speed_true"]
    # Relative noise scale avoids blowing up noise in nearly stagnant regions.
    sigma = noise.relative_speed_sigma * np.maximum(speed, 1.0e-3)
    noisy_speed = speed + rng.normal(0.0, sigma)

    scale = np.ones_like(speed)
    moving = speed > 1.0e-6
    scale[moving] = noisy_speed[moving] / speed[moving]

    fields = dict(fields)
    fields["speed_obs"] = np.maximum(noisy_speed, 0.0)
    fields["ux_obs"] = fields["ux_true"] * scale
    fields["uy_obs"] = fields["uy_true"] * scale
    fields["speed_noise_sigma"] = sigma.astype(np.float32)
    return fields


def summarize_array(name: str, array: np.ndarray, mask: np.ndarray | None = None) -> dict:
    values = array[mask] if mask is not None else array.reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"name": name, "count": 0}
    return {
        "name": name,
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def compute_normalization(cases: dict[str, dict]) -> dict:
    """Pooled stats over grounded, valid cells for VI input scaling."""
    speed_vals = []
    log_visc_vals = []
    bed_vals = []

    for fields in cases.values():
        mask = fields["grounded_mask"].astype(bool)
        speed_vals.append(fields["speed_obs"][mask])
        log_visc_vals.append(fields["log_viscosity_true"][mask])
        bed_vals.append(fields["bed"][mask])

    speed_all = np.concatenate(speed_vals)
    log_visc_all = np.concatenate(log_visc_vals)
    bed_all = np.concatenate(bed_vals)

    return {
        "speed_obs": {
            "mean": float(speed_all.mean()),
            "std": float(max(speed_all.std(), 1.0e-8)),
        },
        "log_viscosity_true": {
            "mean": float(log_visc_all.mean()),
            "std": float(max(log_visc_all.std(), 1.0e-8)),
        },
        "bed": {
            "min": float(bed_all.min()),
            "max": float(bed_all.max()),
        },
    }


def save_bundle(path: Path, fields: dict, case_id: str, noise: NoiseConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = fields["cfg"]

    np.savez_compressed(
        path,
        case_id=case_id,
        x=fields["x"],
        y=fields["y"],
        X=fields["X"],
        Y=fields["Y"],
        bed=fields["bed"],
        thickness_true=fields["thickness_true"],
        surface_true=fields["surface_true"],
        height_above_flotation=fields["height_above_flotation"],
        A_true=fields["A_true"],
        valid_mask=fields["valid_mask"],
        grounded_mask=fields["grounded_mask"],
        floating_mask=fields["floating_mask"],
        speed_true=fields["speed_true"],
        ux_true=fields["ux_true"],
        uy_true=fields["uy_true"],
        speed_obs=fields["speed_obs"],
        ux_obs=fields["ux_obs"],
        uy_obs=fields["uy_obs"],
        speed_noise_sigma=fields["speed_noise_sigma"],
        viscosity_true=fields["viscosity_true"],
        log_viscosity_true=fields["log_viscosity_true"],
        C=float(cfg["C"]),
        a=float(cfg["a"]),
        grid_resolution=fields["grid_resolution"],
        source_npz=str(fields["source_npz"]),
        cfg_json=json.dumps(read_jsonable(cfg), sort_keys=True),
    )


def build_manifest(
    entries: list[dict],
    output_dir: Path,
    noise: NoiseConfig,
    normalization: dict,
) -> dict:
    return {
        "version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": (
            "VI-ready bundles derived from production MISMIP+ spin-up NPZ ground truth. "
            "Observations: noisy surface speed (and components). "
            "Primary inference target: log_viscosity_true (or viscosity_true)."
        ),
        "output_dir": str(output_dir.resolve()),
        "noise": {
            "relative_speed_sigma": noise.relative_speed_sigma,
            "seed": noise.seed,
            "model": "Gaussian on speed with sigma = relative_speed_sigma * max(speed, 1e-3)",
        },
        "masks": {
            "valid_mask": "finite values for speed, velocity, viscosity, HAF",
            "grounded_mask": "valid and height_above_flotation > 0",
            "floating_mask": "valid and height_above_flotation <= 0",
        },
        "fields": {
            "observations": ["speed_obs", "ux_obs", "uy_obs"],
            "truth_targets": ["viscosity_true", "log_viscosity_true"],
            "fixed_geometry": ["bed", "x", "y", "thickness_true", "surface_true"],
            "fixed_physics": ["C", "a", "A_true"],
        },
        "normalization": normalization,
        "cases": entries,
    }


def prepare_case(
    case_id: str,
    source_npz: Path,
    sliding_regime: str,
    output_dir: Path,
    noise: NoiseConfig,
    rng: np.random.Generator,
) -> tuple[dict, dict]:
    fields = load_source_case(source_npz)
    fields = add_velocity_noise(fields, noise, rng)

    bundle_name = f"vi_case_{case_id}.npz"
    bundle_path = output_dir / bundle_name
    save_bundle(bundle_path, fields, case_id, noise)

    cfg = fields["cfg"]
    grounded = fields["grounded_mask"].astype(bool)
    entry = {
        "case_id": case_id,
        "sliding_regime": sliding_regime,
        "source_npz": str(source_npz.resolve()),
        "vi_bundle": str(bundle_path.resolve()),
        "C": float(cfg["C"]),
        "C_start": float(cfg.get("C_start", cfg["C"])),
        "a": float(cfg["a"]),
        "test_mode": bool(cfg.get("test_mode", False)),
        "grid_resolution_m": fields["grid_resolution"],
        "shape_yx": [int(fields["speed_true"].shape[0]), int(fields["speed_true"].shape[1])],
        "fraction_grounded": float(grounded.mean()),
        "summary": {
            "speed_obs": summarize_array("speed_obs", fields["speed_obs"], grounded),
            "viscosity_true": summarize_array(
                "viscosity_true", fields["viscosity_true"], grounded
            ),
            "log_viscosity_true": summarize_array(
                "log_viscosity_true", fields["log_viscosity_true"], grounded
            ),
        },
    }
    return entry, fields


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VI_DIR,
        help="Directory for VI bundles and manifest.json (default: outputs/vi)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.02,
        help="Relative Gaussian noise level on speed observations (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for synthetic observation noise (default: 42)",
    )
    parser.add_argument(
        "--more-npz",
        type=Path,
        default=DEFAULT_CASES["more_sliding"]["source_npz"],
        help="Production NPZ for the more-sliding case",
    )
    parser.add_argument(
        "--less-npz",
        type=Path,
        default=DEFAULT_CASES["no_sliding"]["source_npz"],
        help="Production NPZ for the no-sliding case",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    noise = NoiseConfig(relative_speed_sigma=float(args.noise), seed=int(args.seed))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_specs = {
        "more_sliding": {
            "source_npz": args.more_npz.resolve(),
            "sliding_regime": "high",
        },
        "no_sliding": {
            "source_npz": args.less_npz.resolve(),
            "sliding_regime": "low",
        },
    }

    rng = np.random.default_rng(noise.seed)
    entries = []
    prepared_fields = {}

    for case_id, spec in case_specs.items():
        entry, fields = prepare_case(
            case_id,
            spec["source_npz"],
            spec["sliding_regime"],
            output_dir,
            noise,
            rng,
        )
        entries.append(entry)
        prepared_fields[case_id] = fields
        print(f"Wrote {entry['vi_bundle']}")

    normalization = compute_normalization(prepared_fields)
    manifest = build_manifest(entries, output_dir, noise, normalization)
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(read_jsonable(manifest), stream, indent=2, sort_keys=True)

    print(f"Wrote {manifest_path}")
    print(
        f"Prepared {len(entries)} VI cases | noise={noise.relative_speed_sigma:.3g} | "
        f"seed={noise.seed}"
    )
    return manifest_path


if __name__ == "__main__":
    main()
