#!/usr/bin/env python3
"""Totten unfreeze C-sensitivity: velocity / thickness observation misfits (no vs max).

Run from Archive/ on the cluster (needs torch + checkpoints):

  python scripts/plot_totten_state_misfits_unfreeze.py [tag]
  # default tag: unfreeze_state_reg
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ARCHIVE = Path(__file__).resolve().parents[1]
ROOT = ARCHIVE.parent
if str(ARCHIVE) not in sys.path:
    sys.path.insert(0, str(ARCHIVE))

from models_torch import JointModel, MeanNetwork, make_sparse_vgp, normalize_tensor
from plot_totten_sliding_comparison import prior_get, resolve_ckpt, resolve_device
from train_vi_only_torch import VI_ONLY_ARCHITECTURES
from utilities_torch import (
    ParameterClass,
    load_snapshot,
    make_normalizers,
    resolve_torch_dtype,
    torch_load_checkpoint,
)

TAG = sys.argv[1] if len(sys.argv) > 1 else "unfreeze_state_reg"
NO_CFG = f"configs/totten/c_sensitivity/{TAG}/run_torch_vi_only_totten_no_sliding.cfg"
MAX_CFG = f"configs/totten/c_sensitivity/{TAG}/run_torch_vi_only_totten_max_sliding.cfg"
OUT = ROOT / f"outputs/figures/vi/totten_c_sensitivity/{TAG}"


def predict_state(cfg_path: str, checkpoint_choice: str = "latest"):
    pars = ParameterClass(cfg_path)
    torch_dtype = resolve_torch_dtype(pars.runtime.dtype)
    device = resolve_device(pars, None)
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
    state = torch_load_checkpoint(ckpt, map_location=device)
    if state.get("architecture") not in VI_ONLY_ARCHITECTURES:
        raise RuntimeError(f"bad architecture in {ckpt}: {state.get('architecture')!r}")
    model.load_state_dict(
        {k: v for k, v in state["model"].items() if not k.startswith("mean_net_ref.")},
        strict=False,
    )
    model.eval()

    geom = snapshot.geom_mask
    ys, xs = np.where(geom)
    x = torch.as_tensor(snapshot.x[ys, xs], dtype=torch_dtype, device=device).reshape(-1, 1)
    y = torch.as_tensor(snapshot.y[ys, xs], dtype=torch_dtype, device=device).reshape(-1, 1)
    with torch.no_grad():
        u, v, s, H = model.mean_net(x, y, inverse_norm=True)
        u = u.cpu().numpy().reshape(-1)
        v = v.cpu().numpy().reshape(-1)
        s = s.cpu().numpy().reshape(-1)
        H = H.cpu().numpy().reshape(-1)

    shape = snapshot.x.shape
    u_hat = np.full(shape, np.nan)
    v_hat = np.full(shape, np.nan)
    s_hat = np.full(shape, np.nan)
    h_hat = np.full(shape, np.nan)
    u_hat[ys, xs] = u
    v_hat[ys, xs] = v
    s_hat[ys, xs] = s
    h_hat[ys, xs] = H

    return {
        "cfg": cfg_path,
        "checkpoint": ckpt,
        "epoch": int(state.get("epoch", -1)),
        "friction_C": float(prior_get(pars, "friction_C", 1.0)),
        "eta_log_shift": float(model.eta_log_shift.detach().cpu().item()),
        "x_km": snapshot.x / 1e3,
        "y_km": snapshot.y / 1e3,
        "geom": geom,
        "uv_mask": snapshot.uv_mask,
        "u_obs": np.asarray(snapshot.u, dtype=np.float64),
        "v_obs": np.asarray(snapshot.v, dtype=np.float64),
        "h_obs": np.asarray(snapshot.h, dtype=np.float64),
        "s_obs": np.asarray(snapshot.s, dtype=np.float64),
        "u_hat": u_hat,
        "v_hat": v_hat,
        "h_hat": h_hat,
        "s_hat": s_hat,
    }


def _stats(diff, mask):
    d = diff[mask]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {}
    return {
        "n": int(d.size),
        "rmse": float(np.sqrt(np.mean(d * d))),
        "mae": float(np.mean(np.abs(d))),
        "bias": float(np.mean(d)),
        "p50_abs": float(np.median(np.abs(d))),
        "p90_abs": float(np.percentile(np.abs(d), 90)),
    }


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    OUT.mkdir(parents=True, exist_ok=True)
    print("loading no_sliding…")
    no = predict_state(NO_CFG)
    print("loading max_sliding…")
    mx = predict_state(MAX_CFG)

    geom = no["geom"] & mx["geom"]
    uv = no["uv_mask"] & mx["uv_mask"] & geom
    # Thickness / surface: geometry mask
    hmask = geom & np.isfinite(no["h_obs"]) & np.isfinite(no["h_hat"]) & np.isfinite(mx["h_hat"])

    speed_obs = np.hypot(no["u_obs"], no["v_obs"])
    speed_no = np.hypot(no["u_hat"], no["v_hat"])
    speed_mx = np.hypot(mx["u_hat"], mx["v_hat"])

    du_no = np.where(uv, speed_no - speed_obs, np.nan)
    du_mx = np.where(uv, speed_mx - speed_obs, np.nan)
    dh_no = np.where(hmask, no["h_hat"] - no["h_obs"], np.nan)
    dh_mx = np.where(hmask, mx["h_hat"] - no["h_obs"], np.nan)

    summary = {
        "no_sliding": {
            "friction_C": no["friction_C"],
            "eta_log_shift": no["eta_log_shift"],
            "epoch": no["epoch"],
            "speed_misfit": _stats(du_no, uv),
            "thickness_misfit": _stats(dh_no, hmask),
            "speed_pred_mean": float(np.nanmean(speed_no[uv])),
            "speed_obs_mean": float(np.nanmean(speed_obs[uv])),
        },
        "max_sliding": {
            "friction_C": mx["friction_C"],
            "eta_log_shift": mx["eta_log_shift"],
            "epoch": mx["epoch"],
            "speed_misfit": _stats(du_mx, uv),
            "thickness_misfit": _stats(dh_mx, hmask),
            "speed_pred_mean": float(np.nanmean(speed_mx[uv])),
            "speed_obs_mean": float(np.nanmean(speed_obs[uv])),
        },
    }
    (OUT / "state_misfit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    np.savez_compressed(
        OUT / "state_misfits.npz",
        x_km=no["x_km"],
        y_km=no["y_km"],
        geom=geom,
        uv_mask=uv,
        speed_obs=np.where(uv, speed_obs, np.nan),
        speed_no=np.where(uv, speed_no, np.nan),
        speed_max=np.where(uv, speed_mx, np.nan),
        dspeed_no=du_no,
        dspeed_max=du_mx,
        h_obs=np.where(hmask, no["h_obs"], np.nan),
        h_no=np.where(hmask, no["h_hat"], np.nan),
        h_max=np.where(hmask, mx["h_hat"], np.nan),
        dh_no=dh_no,
        dh_max=dh_mx,
    )

    x_km, y_km = no["x_km"], no["y_km"]

    def add_map(ax, field, title, *, cmap, norm=None, vmin=None, vmax=None):
        kw = {"shading": "auto", "cmap": cmap}
        if norm is not None:
            kw["norm"] = norm
        else:
            kw["vmin"] = vmin
            kw["vmax"] = vmax
        im = ax.pcolormesh(x_km, y_km, field, **kw)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        return im

    # Shared speed scale
    sp_vals = speed_obs[uv]
    sp_vmax = float(np.nanpercentile(sp_vals[np.isfinite(sp_vals)], 98))
    dspeed_lim = max(
        abs(float(np.nanpercentile(du_no, 2))),
        abs(float(np.nanpercentile(du_no, 98))),
        abs(float(np.nanpercentile(du_mx, 2))),
        abs(float(np.nanpercentile(du_mx, 98))),
        1.0,
    )
    dh_lim = max(
        abs(float(np.nanpercentile(dh_no, 2))),
        abs(float(np.nanpercentile(dh_no, 98))),
        abs(float(np.nanpercentile(dh_mx, 2))),
        abs(float(np.nanpercentile(dh_mx, 98))),
        1.0,
    )

    sn = summary["no_sliding"]["speed_misfit"]
    sm = summary["max_sliding"]["speed_misfit"]
    hn = summary["no_sliding"]["thickness_misfit"]
    hm = summary["max_sliding"]["thickness_misfit"]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.0), constrained_layout=True)
    im00 = add_map(axes[0, 0], np.where(uv, speed_obs, np.nan), "Observed speed", cmap="viridis", vmin=0, vmax=sp_vmax)
    im01 = add_map(
        axes[0, 1],
        np.where(uv, speed_no, np.nan),
        rf"no_sliding pred  (RMSE={sn['rmse']:.1f} m/yr)",
        cmap="viridis",
        vmin=0,
        vmax=sp_vmax,
    )
    im02 = add_map(
        axes[0, 2],
        np.where(uv, speed_mx, np.nan),
        rf"max_sliding pred  (RMSE={sm['rmse']:.1f} m/yr)",
        cmap="viridis",
        vmin=0,
        vmax=sp_vmax,
    )
    fig.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.02, label="m/yr")
    fig.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.02, label="m/yr")
    fig.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.02, label="m/yr")

    dnorm = TwoSlopeNorm(vcenter=0.0, vmin=-dspeed_lim, vmax=dspeed_lim)
    im10 = add_map(axes[1, 0], du_no, rf"no_sliding − obs speed  (bias={sn['bias']:.1f})", cmap="RdBu_r", norm=dnorm)
    im11 = add_map(axes[1, 1], du_mx, rf"max_sliding − obs speed  (bias={sm['bias']:.1f})", cmap="RdBu_r", norm=dnorm)
    # thickness misfit side-by-side in third panel as difference of |errors|? Better: show both as overlay hist in text
    # Use third panel for thickness RMSE comparison via dh for no_sliding
    hnorm = TwoSlopeNorm(vcenter=0.0, vmin=-dh_lim, vmax=dh_lim)
    im12 = add_map(
        axes[1, 2],
        dh_no,
        rf"no_sliding − obs H  (RMSE={hn['rmse']:.1f} m)",
        cmap="RdBu_r",
        norm=hnorm,
    )
    fig.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.02, label="m/yr")
    fig.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.02, label="m/yr")
    fig.colorbar(im12, ax=axes[1, 2], fraction=0.046, pad=0.02, label="m")

    fig.suptitle(
        f"Totten {TAG} — velocity / thickness observation misfits",
        fontsize=12,
    )
    path = OUT / "state_misfit_speed.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)

    # Dedicated thickness figure: obs | no | max | dh_no | dh_max | table-ish titles
    h_vals = no["h_obs"][hmask]
    h_vmax = float(np.nanpercentile(h_vals[np.isfinite(h_vals)], 98))
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.0), constrained_layout=True)
    im00 = add_map(axes[0, 0], np.where(hmask, no["h_obs"], np.nan), "Observed thickness", cmap="magma", vmin=0, vmax=h_vmax)
    im01 = add_map(
        axes[0, 1],
        np.where(hmask, no["h_hat"], np.nan),
        rf"no_sliding pred  (RMSE={hn['rmse']:.1f} m)",
        cmap="magma",
        vmin=0,
        vmax=h_vmax,
    )
    im02 = add_map(
        axes[0, 2],
        np.where(hmask, mx["h_hat"], np.nan),
        rf"max_sliding pred  (RMSE={hm['rmse']:.1f} m)",
        cmap="magma",
        vmin=0,
        vmax=h_vmax,
    )
    fig.colorbar(im00, ax=axes[0, 0], fraction=0.046, pad=0.02, label="m")
    fig.colorbar(im01, ax=axes[0, 1], fraction=0.046, pad=0.02, label="m")
    fig.colorbar(im02, ax=axes[0, 2], fraction=0.046, pad=0.02, label="m")

    im10 = add_map(axes[1, 0], dh_no, rf"no_sliding − obs H  (bias={hn['bias']:.1f} m)", cmap="RdBu_r", norm=hnorm)
    im11 = add_map(axes[1, 1], dh_mx, rf"max_sliding − obs H  (bias={hm['bias']:.1f} m)", cmap="RdBu_r", norm=hnorm)
    axes[1, 2].axis("off")
    table = axes[1, 2].table(
        cellText=[
            ["speed RMSE (m/yr)", f"{sn['rmse']:.2f}", f"{sm['rmse']:.2f}"],
            ["speed bias (m/yr)", f"{sn['bias']:.2f}", f"{sm['bias']:.2f}"],
            ["speed mean pred", f"{summary['no_sliding']['speed_pred_mean']:.1f}", f"{summary['max_sliding']['speed_pred_mean']:.1f}"],
            ["H RMSE (m)", f"{hn['rmse']:.2f}", f"{hm['rmse']:.2f}"],
            ["H bias (m)", f"{hn['bias']:.2f}", f"{hm['bias']:.2f}"],
            ["η mean (MPa·yr)", "4.89", "2.09"],
            ["η_log_shift", f"{no['eta_log_shift']:.3f}", f"{mx['eta_log_shift']:.3f}"],
        ],
        colLabels=["", "no_sliding\n(C=100)", "max_sliding\n(C=0.001)"],
        loc="center",
        cellLoc="center",
        bbox=[0.05, 0.15, 0.90, 0.75],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)
    axes[1, 2].set_title("misfit summary", fontsize=11)
    fig.colorbar(im10, ax=axes[1, 0], fraction=0.046, pad=0.02, label="m")
    fig.colorbar(im11, ax=axes[1, 1], fraction=0.046, pad=0.02, label="m")
    fig.suptitle(
        f"Totten {TAG} — thickness misfits (+ velocity/η summary)",
        fontsize=12,
    )
    path = OUT / "state_misfit_thickness.png"
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("wrote", path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
