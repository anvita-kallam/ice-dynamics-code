#!/usr/bin/env python3
"""Compare Totten VI η posteriors: no_sliding (C=100) vs max_sliding (C=0.001).

No viscosity truth — plots posterior mean / latent std and their differences.

Usage (from Archive/ on DSI after training):

  python plot_totten_sliding_comparison.py \\
    configs/totten/run_torch_vi_only_totten_no_sliding.cfg \\
    configs/totten/run_torch_vi_only_totten_max_sliding.cfg \\
    --checkpoint latest

Figures + summary land under outputs/figures/vi/totten_sliding_comparison/ by default.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm, TwoSlopeNorm

from models_torch import JointModel, MeanNetwork, make_sparse_vgp, normalize_tensor
from train_vi_only_torch import VI_ONLY_ARCHITECTURE
from utilities_torch import (
    ParameterClass,
    checkpoint_path,
    load_snapshot,
    make_normalizers,
    resolve_torch_dtype,
    torch_load_checkpoint,
)

DEFAULT_OUT = Path("outputs/figures/vi/totten_sliding_comparison")


def prior_get(pars, name: str, default=None):
    """ConfigParser historically lowercased keys; accept either case."""
    if hasattr(pars.prior, name):
        return getattr(pars.prior, name)
    lower = name.lower()
    if hasattr(pars.prior, lower):
        return getattr(pars.prior, lower)
    return default


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "no_sliding_cfg",
        nargs="?",
        default="configs/totten/run_torch_vi_only_totten_no_sliding.cfg",
    )
    parser.add_argument(
        "max_sliding_cfg",
        nargs="?",
        default="configs/totten/run_torch_vi_only_totten_max_sliding.cfg",
    )
    parser.add_argument(
        "--checkpoint",
        default="latest",
        help="best | latest | path to .pt (Totten runs typically only have model.pt)",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", default=None, help="cuda | cpu | auto (default)")
    return parser.parse_args()


def resolve_device(pars, override: str | None) -> torch.device:
    if override and override != "auto":
        return torch.device(override)
    if pars.torch.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_ckpt(pars, choice: str) -> str:
    if choice == "best":
        name = str(getattr(pars.train, "checkname_best", "model_best") or "model_best")
        path = checkpoint_path(pars.train.checkdir, name)
        if Path(path).is_file():
            return path
        # Totten runs disable early-stop, so fall back to final checkpoint.
        fallback = checkpoint_path(pars.train.checkdir, pars.train.checkname_new)
        if Path(fallback).is_file():
            print(f"note: {path} missing; using {fallback}")
            return fallback
        return path
    if choice in ("latest", "last"):
        return checkpoint_path(pars.train.checkdir, pars.train.checkname_new)
    return choice


def load_eta_maps(cfg_path: str, checkpoint_choice: str, device_override: str | None):
    pars = ParameterClass(cfg_path)
    torch_dtype = resolve_torch_dtype(pars.runtime.dtype)
    device = resolve_device(pars, device_override)
    snapshot = load_snapshot(pars.data.h5file, pars)
    norms = make_normalizers(snapshot)

    mean_net = MeanNetwork(norms, resnet=pars.pretrain.resnet, dtype=torch_dtype)
    x_ref = snapshot.x[snapshot.geom_mask]
    y_ref = snapshot.y[snapshot.geom_mask]
    model = JointModel(
        mean_net,
        make_sparse_vgp(x_ref, y_ref, norms, pars, "eta", torch_dtype),
        make_sparse_vgp(x_ref, y_ref, norms, pars, "lambda", torch_dtype),
        dtype=torch_dtype,
    ).to(device)

    ckpt = resolve_ckpt(pars, checkpoint_choice)
    if not Path(ckpt).is_file():
        raise FileNotFoundError(ckpt)
    state = torch_load_checkpoint(ckpt, map_location=device)
    if state.get("architecture") != VI_ONLY_ARCHITECTURE:
        raise RuntimeError(
            f"Expected VI-only architecture, got {state.get('architecture')!r}"
        )
    model.load_state_dict(
        {k: v for k, v in state["model"].items() if not k.startswith("mean_net_ref.")},
        strict=False,
    )
    model.eval()

    geom = snapshot.geom_mask
    ys, xs = np.where(geom)
    x = torch.as_tensor(snapshot.x[ys, xs], dtype=torch_dtype, device=device).reshape(-1, 1)
    y = torch.as_tensor(snapshot.y[ys, xs], dtype=torch_dtype, device=device).reshape(-1, 1)
    Xn = normalize_tensor(
        torch.cat([x, y], dim=1), model.mean_net.iW_coord, model.mean_net.b_coord
    )
    with torch.no_grad():
        theta, var, _, _, _ = model.vgp_eta.posterior_stats(Xn)
        theta = theta.cpu().numpy().reshape(-1)
        std = torch.sqrt(var).cpu().numpy().reshape(-1)

    eta_init = float(pars.prior.eta_init)
    shift = float(model.eta_log_shift.detach().cpu().item())
    eta_log = np.clip(
        math.log(eta_init) + shift + theta,
        math.log(float(pars.prior.eta_min)),
        math.log(float(pars.prior.eta_max)),
    )
    eta_mean = np.exp(eta_log)
    log10_std = std / math.log(10.0)

    eta_map = np.full(snapshot.x.shape, np.nan)
    std_map = np.full(snapshot.x.shape, np.nan)
    log10_eta_map = np.full(snapshot.x.shape, np.nan)
    log10_std_map = np.full(snapshot.x.shape, np.nan)
    eta_map[ys, xs] = eta_mean
    std_map[ys, xs] = std
    log10_eta_map[ys, xs] = np.log10(eta_mean)
    log10_std_map[ys, xs] = log10_std

    return {
        "cfg": cfg_path,
        "checkpoint": ckpt,
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "friction_C": float(prior_get(pars, "friction_C", 1.0)),
        "eta_init": eta_init,
        "eta_log_shift": shift,
        "x_km": snapshot.x / 1e3,
        "y_km": snapshot.y / 1e3,
        "geom": geom,
        "eta_map": eta_map,
        "std_map": std_map,
        "log10_eta_map": log10_eta_map,
        "log10_std_map": log10_std_map,
        "eta_flat": eta_mean,
        "std_flat": std,
        "log10_std_flat": log10_std,
        "kernel": model.vgp_eta.kernel_diagnostics(),
    }


def add_map(ax, x_km, y_km, field, title, *, cmap, norm=None, vmin=None, vmax=None):
    image = ax.pcolormesh(
        x_km, y_km, field, shading="auto", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax
    )
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    return image


def shared_log_norm(*fields, lo=5, hi=95):
    vals = np.concatenate([f[np.isfinite(f) & (f > 0)].ravel() for f in fields])
    return LogNorm(
        vmin=max(float(np.percentile(vals, lo)), 1e-3),
        vmax=float(np.percentile(vals, hi)),
    )


def shared_linear_limits(*fields, lo=5, hi=95):
    vals = np.concatenate([f[np.isfinite(f)].ravel() for f in fields])
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading no_sliding: {args.no_sliding_cfg}")
    no = load_eta_maps(args.no_sliding_cfg, args.checkpoint, args.device)
    print(f"loading max_sliding: {args.max_sliding_cfg}")
    mx = load_eta_maps(args.max_sliding_cfg, args.checkpoint, args.device)

    if no["eta_map"].shape != mx["eta_map"].shape:
        raise RuntimeError("Grid shape mismatch between cases")

    both = no["geom"] & mx["geom"]
    d_log10 = np.where(both, no["log10_eta_map"] - mx["log10_eta_map"], np.nan)
    d_std = np.where(both, no["log10_std_map"] - mx["log10_std_map"], np.nan)
    ratio = np.where(both, no["eta_map"] / np.maximum(mx["eta_map"], 1e-12), np.nan)

    # --- Main 2x3 comparison -------------------------------------------------
    eta_norm = shared_log_norm(no["eta_map"], mx["eta_map"])
    std_lo, std_hi = shared_linear_limits(no["log10_std_map"], mx["log10_std_map"])
    d_lim = max(
        abs(float(np.nanpercentile(d_log10, 2))),
        abs(float(np.nanpercentile(d_log10, 98))),
        1e-3,
    )
    d_norm = TwoSlopeNorm(vcenter=0.0, vmin=-d_lim, vmax=d_lim)
    ds_lim = max(
        abs(float(np.nanpercentile(d_std, 2))),
        abs(float(np.nanpercentile(d_std, 98))),
        1e-4,
    )
    ds_norm = TwoSlopeNorm(vcenter=0.0, vmin=-ds_lim, vmax=ds_lim)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.5), constrained_layout=True)
    panels = (
        (axes[0, 0], no["eta_map"], f"no_sliding η mean\n(C={no['friction_C']:g})", "viridis", eta_norm, None, None, "η (MPa·yr)"),
        (axes[0, 1], mx["eta_map"], f"max_sliding η mean\n(C={mx['friction_C']:g})", "viridis", eta_norm, None, None, "η (MPa·yr)"),
        (axes[0, 2], d_log10, "Δ log10 η\n(no − max)", "RdBu_r", d_norm, None, None, "log10"),
        (axes[1, 0], no["log10_std_map"], "no_sliding log10 η std", "magma", None, std_lo, std_hi, "log10"),
        (axes[1, 1], mx["log10_std_map"], "max_sliding log10 η std", "magma", None, std_lo, std_hi, "log10"),
        (axes[1, 2], d_std, "Δ log10 η std\n(no − max)", "RdBu_r", ds_norm, None, None, "log10"),
    )
    for ax, field, title, cmap, norm, vmin, vmax, unit in panels:
        image = add_map(
            ax, no["x_km"], no["y_km"], field, title,
            cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label=unit)
    fig.suptitle(
        "Totten sequential VI — basal-sliding end-member comparison",
        fontsize=13,
    )
    maps_path = args.output_dir / "eta_mean_std_comparison.png"
    fig.savefig(maps_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- Scatter / histograms ------------------------------------------------
    no_eta = no["eta_flat"]
    mx_eta = mx["eta_flat"]
    no_s = no["log10_std_flat"]
    mx_s = mx["log10_std_flat"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    axes[0].scatter(np.log10(mx_eta), np.log10(no_eta), s=3, alpha=0.25, c="steelblue")
    lims = [
        min(axes[0].get_xlim()[0], axes[0].get_ylim()[0]),
        max(axes[0].get_xlim()[1], axes[0].get_ylim()[1]),
    ]
    axes[0].plot(lims, lims, "k--", lw=1)
    axes[0].set_xlabel("log10 η max_sliding")
    axes[0].set_ylabel("log10 η no_sliding")
    r = float(np.corrcoef(np.log10(mx_eta), np.log10(no_eta))[0, 1])
    axes[0].set_title(f"η mean agreement (r={r:.3f})")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(np.log10(no_eta), bins=60, alpha=0.55, label="no_sliding", color="C0")
    axes[1].hist(np.log10(mx_eta), bins=60, alpha=0.55, label="max_sliding", color="C1")
    axes[1].set_xlabel("log10 η")
    axes[1].set_ylabel("count")
    axes[1].legend(fontsize=8)
    axes[1].set_title("η mean distributions")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(no_s, bins=60, alpha=0.55, label="no_sliding", color="C0")
    axes[2].hist(mx_s, bins=60, alpha=0.55, label="max_sliding", color="C1")
    axes[2].set_xlabel("posterior std (log10 η)")
    axes[2].set_ylabel("count")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Uncertainty distributions")
    axes[2].grid(True, alpha=0.3)
    diag_path = args.output_dir / "eta_sliding_diagnostics.png"
    fig.savefig(diag_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- Ratio map -----------------------------------------------------------
    ratio_lim = max(
        abs(float(np.nanpercentile(np.log10(ratio), 2))),
        abs(float(np.nanpercentile(np.log10(ratio), 98))),
        1e-3,
    )
    ratio_norm = TwoSlopeNorm(vcenter=0.0, vmin=-ratio_lim, vmax=ratio_lim)
    fig, ax = plt.subplots(figsize=(6.2, 7.2), constrained_layout=True)
    image = add_map(
        ax, no["x_km"], no["y_km"], np.log10(ratio),
        "log10(η_no / η_max)",
        cmap="RdBu_r", norm=ratio_norm,
    )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02, label="log10 ratio")
    ratio_path = args.output_dir / "eta_mean_log_ratio.png"
    fig.savefig(ratio_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    def flat_stats(name, arr):
        v = arr[np.isfinite(arr)]
        return {
            "name": name,
            "count": int(v.size),
            "min": float(np.min(v)),
            "p05": float(np.percentile(v, 5)),
            "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)),
            "max": float(np.max(v)),
            "mean": float(np.mean(v)),
        }

    summary = {
        "no_sliding": {
            "cfg": no["cfg"],
            "checkpoint": no["checkpoint"],
            "epoch": no["checkpoint_epoch"],
            "friction_C": no["friction_C"],
            "eta_init": no["eta_init"],
            "eta_log_shift": no["eta_log_shift"],
            "eta_mean": flat_stats("eta", no["eta_map"]),
            "log10_std": flat_stats("log10_std", no["log10_std_map"]),
            "kernel": no["kernel"],
        },
        "max_sliding": {
            "cfg": mx["cfg"],
            "checkpoint": mx["checkpoint"],
            "epoch": mx["checkpoint_epoch"],
            "friction_C": mx["friction_C"],
            "eta_init": mx["eta_init"],
            "eta_log_shift": mx["eta_log_shift"],
            "eta_mean": flat_stats("eta", mx["eta_map"]),
            "log10_std": flat_stats("log10_std", mx["log10_std_map"]),
            "kernel": mx["kernel"],
        },
        "comparison": {
            "log10_eta_corr": r,
            "median_log10_eta_diff_no_minus_max": float(np.nanmedian(d_log10)),
            "median_eta_ratio_no_over_max": float(np.nanmedian(ratio)),
            "median_log10_std_diff_no_minus_max": float(np.nanmedian(d_std)),
            "mean_abs_log10_eta_diff": float(np.nanmean(np.abs(d_log10))),
        },
        "figures": [str(p) for p in (maps_path, diag_path, ratio_path)],
    }
    summary_path = args.output_dir / "sliding_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["comparison"], indent=2))
    print(f"wrote {maps_path}")
    print(f"wrote {diag_path}")
    print(f"wrote {ratio_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
