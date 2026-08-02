#!/usr/bin/env python3
"""Evaluate held-out velocity prediction after sequential VI on MISMIP A=20.

Loads the VI-only checkpoint (frozen PINN + inferred η), predicts velocity over
the full glacier from the PINN, and scores train (80%) vs holdout (20%) pixels
against the original (unmasked) velocity observations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Archive"
DEFAULT_CFG = ARCHIVE / "configs/heldout_velocity_validation/run_torch_vi_only_a20_holdout.cfg"
DEFAULT_OUT = ROOT / "outputs/heldout_velocity_validation"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--checkpoint", default="best", help="best | latest | path")
    p.add_argument("--device", default=None, help="cuda | cpu | auto")
    return p.parse_args()


def _resolve_path(path: str | Path, base: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    cand = (base / path).resolve()
    if cand.exists():
        return cand
    return (ROOT / path).resolve()


def _metrics(pred: np.ndarray, obs: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=float).ravel()
    obs = np.asarray(obs, dtype=float).ravel()
    valid = np.isfinite(pred) & np.isfinite(obs)
    p = pred[valid]
    o = obs[valid]
    if p.size == 0:
        return {
            "n": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "pred_mean": float("nan"),
            "obs_mean": float("nan"),
            "mean_bias": float("nan"),
        }
    resid = p - o
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((o - np.mean(o)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "n": int(p.size),
        "rmse": float(np.sqrt(np.mean(resid**2))),
        "mae": float(np.mean(np.abs(resid))),
        "r2": r2,
        "pred_mean": float(np.mean(p)),
        "obs_mean": float(np.mean(o)),
        "mean_bias": float(np.mean(resid)),
    }


def _interpretation(train: dict, val: dict) -> str:
    tr = train["rmse"]
    vr = val["rmse"]
    ratio = vr / tr if tr and np.isfinite(tr) and tr > 0 else float("nan")
    r2_gap = train["r2"] - val["r2"] if np.isfinite(train["r2"]) and np.isfinite(val["r2"]) else float("nan")

    lines = []
    if np.isfinite(vr) and np.isfinite(val["r2"]) and val["r2"] >= 0.9 and (not np.isfinite(ratio) or ratio < 1.5):
        lines.append(
            "Yes — held-out velocities are predicted accurately: validation R² is high "
            f"({val['r2']:.3f}) and validation RMSE ({vr:.3g}) is close to training RMSE ({tr:.3g})."
        )
    elif np.isfinite(vr) and np.isfinite(val["r2"]) and val["r2"] >= 0.7:
        lines.append(
            "Partially — the inferred model predicts held-out velocities reasonably well "
            f"(validation R²={val['r2']:.3f}), but residuals are larger than on training pixels."
        )
    else:
        lines.append(
            "No — held-out velocity predictions are weak relative to the validation goal "
            f"(validation R²={val.get('r2', float('nan')):.3f}, RMSE={vr:.3g})."
        )

    if np.isfinite(ratio):
        lines.append(
            f"Validation RMSE is {ratio:.2f}× training RMSE "
            f"(ΔR² train−val = {r2_gap:.3f})."
        )
    if np.isfinite(ratio) and ratio < 1.25 and np.isfinite(r2_gap) and r2_gap < 0.05:
        lines.append(
            "Little evidence of overfitting: train and held-out scores are similar, "
            "suggesting good generalization to unseen velocity observations."
        )
    elif np.isfinite(ratio) and ratio > 2.0:
        lines.append(
            "Evidence of overfitting / poor generalization: held-out error is much larger "
            "than training error."
        )
    else:
        lines.append(
            "Train–validation gap is moderate; some generalization with limited overfitting."
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    sys.path.insert(0, str(ARCHIVE))

    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    import predict_torch as _predict
    from models_torch import JointModel, MeanNetwork, make_sparse_vgp
    from train_vi_only_torch import VI_ONLY_ARCHITECTURE
    from utilities_torch import (
        ParameterClass,
        checkpoint_path,
        flatten_snapshot,
        load_snapshot,
        make_normalizers,
        resolve_np_dtype,
        resolve_torch_dtype,
        torch_load_checkpoint,
    )

    cfg_path = args.cfg.resolve()
    pars = ParameterClass(str(cfg_path))
    archive_cwd = ARCHIVE

    source = _resolve_path(
        getattr(pars.data, "velocity_holdout_source", args.cfg),
        archive_cwd,
    )
    if not source.is_file():
        # Fallback: sibling of holdout file if source key missing/broken.
        source = ROOT / (
            "outputs/spinup/production/more_sliding/"
            "SteadyState_more_sliding_10500yr_ramp4000_1refine_grid.npz"
        )
    mask_path = _resolve_path(getattr(pars.data, "velocity_holdout_mask"), archive_cwd)
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing holdout mask: {mask_path}")

    torch_dtype = resolve_torch_dtype(pars.runtime.dtype)
    if args.device and args.device != "auto":
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if pars.torch.device != "cpu" and torch.cuda.is_available() else "cpu"
        )

    # Full unmasked observations for scoring.
    full_pars = ParameterClass(str(cfg_path))
    full_pars.data.h5file = str(source)
    full_snap = load_snapshot(str(source), full_pars)

    # Holdout training snapshot (PINN norms / geom from the VI data file).
    train_snap = load_snapshot(pars.data.h5file, pars)
    norms = make_normalizers(train_snap)
    mean_net = MeanNetwork(norms, resnet=pars.pretrain.resnet, dtype=torch_dtype)
    x_ref = train_snap.x[train_snap.geom_mask]
    y_ref = train_snap.y[train_snap.geom_mask]
    model = JointModel(
        mean_net,
        make_sparse_vgp(x_ref, y_ref, norms, pars, "eta", torch_dtype),
        make_sparse_vgp(x_ref, y_ref, norms, pars, "lambda", torch_dtype),
        dtype=torch_dtype,
    ).to(device)

    if args.checkpoint == "best":
        ckpt = checkpoint_path(
            pars.train.checkdir,
            str(getattr(pars.train, "checkname_best", "model_best") or "model_best"),
        )
        if not Path(ckpt).is_file():
            ckpt = checkpoint_path(pars.train.checkdir, pars.train.checkname_new)
    elif args.checkpoint in ("latest", "last"):
        ckpt = checkpoint_path(pars.train.checkdir, pars.train.checkname_new)
    else:
        ckpt = args.checkpoint
    # Resolve relative checkpoint paths from Archive/
    ckpt_path = Path(ckpt)
    if not ckpt_path.is_file():
        ckpt_path = (ARCHIVE / ckpt).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"VI checkpoint not found: {ckpt}")

    state = torch_load_checkpoint(str(ckpt_path), map_location=device)
    if state.get("architecture") != VI_ONLY_ARCHITECTURE:
        raise RuntimeError(
            f"Expected {VI_ONLY_ARCHITECTURE}, got {state.get('architecture')!r}"
        )
    model.load_state_dict(
        {k: v for k, v in state["model"].items() if not k.startswith("mean_net_ref.")},
        strict=False,
    )
    model.eval()

    arrays = flatten_snapshot(
        full_snap,
        norms,
        pars.prior.thickness_min,
        np_dtype=resolve_np_dtype(pars.runtime.dtype),
    )
    preds = _predict._batched_mean_predictions(
        model.mean_net,
        arrays,
        pars.predict.batch_size,
        pars.prior.thickness_min,
        device,
        torch_dtype,
    )

    # Reconstruct full-grid predicted velocity.
    geom = full_snap.geom_mask
    u_pred = np.full(geom.shape, np.nan, dtype=float)
    v_pred = np.full(geom.shape, np.nan, dtype=float)
    u_pred[geom] = np.asarray(preds["u"]).reshape(-1)
    v_pred[geom] = np.asarray(preds["v"]).reshape(-1)
    speed_pred = np.hypot(u_pred, v_pred)
    speed_obs = np.hypot(full_snap.u, full_snap.v)

    with np.load(mask_path) as m:
        holdout_mask = np.asarray(m["holdout_mask"], dtype=bool)
        train_mask = np.asarray(m["train_mask"], dtype=bool)
        valid_mask = np.asarray(m["valid_mask"], dtype=bool)

    train_m = _metrics(speed_pred[train_mask], speed_obs[train_mask])
    val_m = _metrics(speed_pred[holdout_mask], speed_obs[holdout_mask])
    residual = speed_pred - speed_obs

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": str(cfg_path),
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "source_npz": str(source),
        "holdout_mask": str(mask_path),
        "velocity_holdout_fraction": float(
            getattr(pars.data, "velocity_holdout_fraction", 0.2)
        ),
        "velocity_holdout_seed": int(getattr(pars.data, "velocity_holdout_seed", 42)),
        "n_valid": int(valid_mask.sum()),
        "n_train": int(train_mask.sum()),
        "n_holdout": int(holdout_mask.sum()),
        "training": train_m,
        "heldout": val_m,
        "interpretation": _interpretation(train_m, val_m),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "n",
                "rmse",
                "mae",
                "r2",
                "pred_mean",
                "obs_mean",
                "mean_bias",
            ]
        )
        for name, m in (("training", train_m), ("heldout", val_m)):
            writer.writerow(
                [
                    name,
                    m["n"],
                    m["rmse"],
                    m["mae"],
                    m["r2"],
                    m["pred_mean"],
                    m["obs_mean"],
                    m["mean_bias"],
                ]
            )

    x_km = full_snap.x / 1e3
    y_km = full_snap.y / 1e3

    # Scatter
    fig, ax = plt.subplots(figsize=(6.5, 6.2))
    ax.scatter(
        speed_obs[train_mask],
        speed_pred[train_mask],
        s=4,
        alpha=0.25,
        c="tab:blue",
        label="training (80%)",
        rasterized=True,
    )
    ax.scatter(
        speed_obs[holdout_mask],
        speed_pred[holdout_mask],
        s=8,
        alpha=0.45,
        c="tab:orange",
        label="held-out (20%)",
        rasterized=True,
    )
    lims = [
        float(np.nanmin(speed_obs[valid_mask])),
        float(np.nanmax(speed_obs[valid_mask])),
    ]
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.set_xlabel("observed speed (m/yr)")
    ax.set_ylabel("predicted speed (m/yr)")
    ax.set_title("Predicted vs observed velocity")
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "velocity_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Residual histogram
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.hist(
        residual[train_mask][np.isfinite(residual[train_mask])],
        bins=60,
        density=True,
        alpha=0.55,
        color="tab:blue",
        label="training",
    )
    ax.hist(
        residual[holdout_mask][np.isfinite(residual[holdout_mask])],
        bins=60,
        density=True,
        alpha=0.55,
        color="tab:orange",
        label="held-out",
    )
    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel("residual = pred − obs (m/yr)")
    ax.set_ylabel("density")
    ax.set_title("Velocity residual histogram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "velocity_residuals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Holdout mask map
    split = np.full(geom.shape, np.nan, dtype=float)
    split[train_mask] = 0.0
    split[holdout_mask] = 1.0
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.pcolormesh(x_km, y_km, split, shading="auto", cmap="coolwarm", vmin=0, vmax=1)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Holdout mask (blue=train, red=held-out)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["train", "holdout"])
    fig.tight_layout()
    fig.savefig(out_dir / "holdout_mask.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Predicted speed
    fig, ax = plt.subplots(figsize=(10, 3.8))
    im = ax.pcolormesh(x_km, y_km, speed_pred, shading="auto", cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Predicted velocity speed")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="m/yr")
    fig.tight_layout()
    fig.savefig(out_dir / "predicted_velocity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Residual map
    fig, ax = plt.subplots(figsize=(10, 3.8))
    vmax = float(np.nanpercentile(np.abs(residual[valid_mask]), 98))
    vmax = max(vmax, 1e-6)
    im = ax.pcolormesh(
        x_km, y_km, residual, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Velocity residual map (pred − obs)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="m/yr")
    fig.tight_layout()
    fig.savefig(out_dir / "residual_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print()
    print(f"Training RMSE:   {train_m['rmse']:.6g}")
    print(f"Validation RMSE: {val_m['rmse']:.6g}")
    print()
    print(f"Training MAE:    {train_m['mae']:.6g}")
    print(f"Validation MAE:  {val_m['mae']:.6g}")
    print()
    print(f"Training R²:     {train_m['r2']:.6g}")
    print(f"Validation R²:   {val_m['r2']:.6g}")
    print()
    print("Interpretation:")
    print(summary["interpretation"])
    print()
    print(f"wrote metrics + figures under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
