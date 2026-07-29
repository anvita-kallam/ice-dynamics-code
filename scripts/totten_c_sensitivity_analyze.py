#!/usr/bin/env python3
"""Analyze one Totten no/max sliding experiment pair (Exps 2, 5, 6).

Computes full-domain and grounded-only η diagnostics, Δη histograms,
SSA residual decomposition maps, and writes a per-experiment summary JSON.

Usage (from Archive/, after both VI jobs finish):

  python ../scripts/totten_c_sensitivity_analyze.py \\
    configs/totten/c_sensitivity/phys_w5/run_torch_vi_only_totten_no_sliding.cfg \\
    configs/totten/c_sensitivity/phys_w5/run_torch_vi_only_totten_max_sliding.cfg \\
    --tag phys_w5 \\
    --output-dir ../outputs/figures/vi/totten_c_sensitivity/phys_w5
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
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Archive"
sys.path.insert(0, str(ARCHIVE))

from models_torch import (  # noqa: E402
    JointModel,
    MeanNetwork,
    icepack_ssa_constants,
    make_sparse_vgp,
    materialize_eta_numpy,
    normalize_tensor,
    spinup_plastic_basal_drag,
)
from plot_totten_sliding_comparison import (  # noqa: E402
    add_map,
    load_eta_maps,
    prior_get,
    resolve_ckpt,
    resolve_device,
)
from train_vi_only_torch import (  # noqa: E402
    VI_ONLY_ARCHITECTURES,
    grounded_mask_from_snapshot,
)
from utilities_torch import (  # noqa: E402
    ParameterClass,
    load_snapshot,
    make_normalizers,
    resolve_torch_dtype,
    torch_load_checkpoint,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("no_sliding_cfg")
    p.add_argument("max_sliding_cfg")
    p.add_argument("--tag", default="experiment")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--device", default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: outputs/figures/vi/totten_c_sensitivity/<tag>",
    )
    p.add_argument("--residual-stride", type=int, default=2)
    return p.parse_args()


def _eta_stats(eta: np.ndarray) -> dict:
    v = eta[np.isfinite(eta) & (eta > 0)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "max_over_min": float(np.max(v) / max(np.min(v), 1e-30)),
        "log10_range": float(np.log10(np.max(v)) - np.log10(np.min(v))),
    }


def _corr_log10(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    aa = a[mask]
    bb = b[mask]
    ok = np.isfinite(aa) & np.isfinite(bb) & (aa > 0) & (bb > 0)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(np.log10(aa[ok]), np.log10(bb[ok]))[0, 1])


def _final_losses(logfile: Path) -> dict:
    """Parse last epoch train_total / train_phys from metrics CSV if present."""
    csv_path = logfile.parent / f"metrics_vi_only_{logfile.name}.csv"
    if not csv_path.is_file():
        return {"metrics_csv": None}
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        row = df.iloc[-1]
        return {
            "metrics_csv": str(csv_path),
            "epoch": int(row.get("epoch", -1)),
            "train_total": float(row["train_total"]),
            "train_phys": float(row["train_phys"]),
            "train_data": float(row["train_data"]),
            "train_kl": float(row["train_kl"]),
            "test_total": float(row["test_total"]) if "test_total" in df else float("nan"),
            "test_phys": float(row["test_phys"]) if "test_phys" in df else float("nan"),
        }
    except Exception as exc:
        return {"metrics_csv": str(csv_path), "error": str(exc)}


def _ssa_component_maps(cfg_path: str, checkpoint_choice: str, device_override, stride: int):
    """Membrane / basal / driving / residual magnitude maps on a strided ice grid."""
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
    if stride > 1:
        keep = (ys % stride == 0) & (xs % stride == 0)
        ys, xs = ys[keep], xs[keep]

    x = torch.as_tensor(snapshot.x[ys, xs], dtype=torch_dtype, device=device).reshape(-1, 1)
    y = torch.as_tensor(snapshot.y[ys, xs], dtype=torch_dtype, device=device).reshape(-1, 1)
    Xn = normalize_tensor(
        torch.cat([x, y], dim=1), model.mean_net.iW_coord, model.mean_net.b_coord
    )

    def _grad(out, inp):
        return torch.autograd.grad(
            out, inp, grad_outputs=torch.ones_like(out), create_graph=False, retain_graph=True
        )[0]

    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        u, v, s, H = model.mean_net(x, y, inverse_norm=True)
        s_x, s_y = _grad(s, x), _grad(s, y)
        u_x, u_y = _grad(u, x), _grad(u, y)
        v_x, v_y = _grad(v, x), _grad(v, y)
        ice = icepack_ssa_constants(pars, torch_dtype, device)
        friction_C = ice["friction_C"]
        tau_dx = ice["rho_ice"] * ice["g"] * H * s_x
        tau_dy = ice["rho_ice"] * ice["g"] * H * s_y
        water_depth = torch.clamp(-(s - H), min=0.0)
        p_water = ice["rho_water"] * ice["g"] * water_depth
        p_ice = ice["rho_ice"] * ice["g"] * H
        tau_c = 0.5 * torch.clamp(p_ice - p_water, min=0.0)
        speed_eps = torch.tensor(
            float(getattr(pars.prior, "speed_epsilon", 1.0)), dtype=torch_dtype, device=device
        )
        speed = torch.sqrt(u.square() + v.square() + speed_eps.square())
        bx, by = spinup_plastic_basal_drag(
            u, v, speed, tau_c, friction_C, ice["weertman_m"], speed_eps
        )

        with torch.no_grad():
            theta, _, _, _, _ = model.vgp_eta.posterior_stats(Xn)
            theta_np = theta.detach().cpu().numpy().reshape(-1)
        shift = float(model.eta_log_shift.detach().cpu().item())
        eta_np = materialize_eta_numpy(
            math.log(float(pars.prior.eta_init)) + shift + theta_np,
            float(pars.prior.eta_min),
            float(pars.prior.eta_max),
            str(getattr(pars.prior, "eta_bound_mode", "log_clamp") or "log_clamp"),
        )
        eta = torch.as_tensor(eta_np, dtype=torch_dtype, device=device).reshape_as(H)

        membrane_xx = 2.0 * H * eta * (2.0 * u_x + v_y)
        membrane_xy = H * eta * (u_y + v_x)
        membrane_yy = 2.0 * H * eta * (u_x + 2.0 * v_y)
        membrane_div_x = _grad(membrane_xx, x) + _grad(membrane_xy, y)
        membrane_div_y = _grad(membrane_xy, x) + _grad(membrane_yy, y)
        rux = membrane_div_x + tau_dx - bx
        rvy = membrane_div_y + tau_dy - by

    shape = snapshot.x.shape

    def _to_map(t):
        arr = np.full(shape, np.nan, dtype=np.float64)
        arr[ys, xs] = t.detach().cpu().numpy().reshape(-1)
        return arr

    membrane = _to_map(torch.sqrt(0.5 * (membrane_div_x.square() + membrane_div_y.square())))
    basal = _to_map(torch.sqrt(0.5 * (bx.square() + by.square())))
    driving = _to_map(torch.sqrt(0.5 * (tau_dx.square() + tau_dy.square())))
    residual = _to_map(torch.sqrt(0.5 * (rux.square() + rvy.square())))
    grounded = _to_map(tau_c) > 0

    return {
        "x_km": snapshot.x / 1e3,
        "y_km": snapshot.y / 1e3,
        "geom": geom,
        "grounded": grounded,
        "membrane": membrane,
        "basal": basal,
        "driving": driving,
        "residual": residual,
        "friction_C": float(prior_get(pars, "friction_C", 1.0)),
        "eta_log_shift": shift,
        "logfile": Path(pars.train.logfile),
    }


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir or (
        ROOT / "outputs/figures/vi/totten_c_sensitivity" / args.tag
    )
    out_dir = out_dir if out_dir.is_absolute() else (Path.cwd() / out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading η maps: {args.tag}")
    no = load_eta_maps(args.no_sliding_cfg, args.checkpoint, args.device)
    mx = load_eta_maps(args.max_sliding_cfg, args.checkpoint, args.device)
    both = no["geom"] & mx["geom"]
    pars_no = ParameterClass(args.no_sliding_cfg)
    # Grounded mask from snapshot geometry (obs s,h) — shared for both cases.
    snapshot = load_snapshot(pars_no.data.h5file, pars_no)
    grounded = grounded_mask_from_snapshot(snapshot, pars_no) & both
    floating = both & ~grounded

    d_eta = np.where(both, mx["eta_map"] - no["eta_map"], np.nan)
    abs_d = np.abs(d_eta)

    metrics = {
        "tag": args.tag,
        "no_sliding_cfg": args.no_sliding_cfg,
        "max_sliding_cfg": args.max_sliding_cfg,
        "frac_grounded": float(grounded.sum() / max(both.sum(), 1)),
        "eta_stats_full": {
            "no": _eta_stats(no["eta_map"][both]),
            "max": _eta_stats(mx["eta_map"][both]),
        },
        "eta_stats_grounded": {
            "no": _eta_stats(no["eta_map"][grounded]),
            "max": _eta_stats(mx["eta_map"][grounded]),
        },
        "eta_stats_floating": {
            "no": _eta_stats(no["eta_map"][floating]),
            "max": _eta_stats(mx["eta_map"][floating]),
        },
        "log10_eta_corr_full": _corr_log10(no["eta_map"], mx["eta_map"], both),
        "log10_eta_corr_grounded": _corr_log10(no["eta_map"], mx["eta_map"], grounded),
        "log10_eta_corr_floating": _corr_log10(no["eta_map"], mx["eta_map"], floating),
        "delta_eta": {
            "mean_abs_full": float(np.nanmean(abs_d[both])),
            "mean_abs_grounded": float(np.nanmean(abs_d[grounded])),
            "mean_abs_floating": float(np.nanmean(abs_d[floating])),
            "max_abs_full": float(np.nanmax(abs_d[both])),
            "max_abs_grounded": float(np.nanmax(abs_d[grounded])),
            "median_abs_grounded": float(np.nanmedian(abs_d[grounded])),
        },
        "eta_log_shift": {
            "no": no["eta_log_shift"],
            "max": mx["eta_log_shift"],
        },
    }

    # Losses from metrics CSV
    metrics["losses"] = {
        "no": _final_losses(Path(ParameterClass(args.no_sliding_cfg).train.logfile)),
        "max": _final_losses(Path(ParameterClass(args.max_sliding_cfg).train.logfile)),
    }

    # --- Figures: Δη + grounded histogram ------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    lim = max(abs(float(np.nanpercentile(d_eta, 2))), abs(float(np.nanpercentile(d_eta, 98))), 1e-4)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    add_map(axes[0], no["x_km"], no["y_km"], d_eta, "Δη = max − no (full)", cmap="RdBu_r", norm=norm)
    d_g = np.where(grounded, d_eta, np.nan)
    add_map(axes[1], no["x_km"], no["y_km"], d_g, "Δη grounded only", cmap="RdBu_r", norm=norm)
    for ax in axes[:2]:
        fig.colorbar(
            ax.collections[0], ax=ax, fraction=0.046, pad=0.02, label="η (MPa·yr)"
        )
    dg = d_eta[grounded]
    dg = dg[np.isfinite(dg)]
    axes[2].hist(dg, bins=60, color="steelblue", alpha=0.85)
    axes[2].axvline(0.0, color="k", ls="--", lw=1)
    axes[2].set_xlabel("Δη (max − no) on grounded ice")
    axes[2].set_ylabel("count")
    axes[2].set_title(
        f"grounded Δη\n"
        f"corr_full={metrics['log10_eta_corr_full']:.3f}  "
        f"corr_g={metrics['log10_eta_corr_grounded']:.3f}"
    )
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"Totten C-sensitivity — {args.tag}", fontsize=12)
    delta_path = out_dir / "delta_eta_grounded.png"
    fig.savefig(delta_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Scatter full vs grounded
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    for ax, mask, title in (
        (axes[0], both, f"full domain r={metrics['log10_eta_corr_full']:.3f}"),
        (axes[1], grounded, f"grounded only r={metrics['log10_eta_corr_grounded']:.3f}"),
    ):
        ax.scatter(
            np.log10(mx["eta_map"][mask]),
            np.log10(no["eta_map"][mask]),
            s=3,
            alpha=0.25,
            c="C0",
        )
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel("log10 η max_sliding")
        ax.set_ylabel("log10 η no_sliding")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    scatter_path = out_dir / "eta_corr_full_vs_grounded.png"
    fig.savefig(scatter_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- Residual decomposition (Exp 5) --------------------------------------
    print("computing SSA residual components…")
    res_no = _ssa_component_maps(
        args.no_sliding_cfg, args.checkpoint, args.device, args.residual_stride
    )
    res_mx = _ssa_component_maps(
        args.max_sliding_cfg, args.checkpoint, args.device, args.residual_stride
    )
    np.savez_compressed(
        out_dir / "ssa_residual_components.npz",
        x_km=res_no["x_km"],
        y_km=res_no["y_km"],
        geom=res_no["geom"],
        grounded=res_no["grounded"],
        membrane_no=res_no["membrane"],
        membrane_max=res_mx["membrane"],
        basal_no=res_no["basal"],
        basal_max=res_mx["basal"],
        driving_no=res_no["driving"],
        driving_max=res_mx["driving"],
        residual_no=res_no["residual"],
        residual_max=res_mx["residual"],
        d_membrane=res_no["membrane"] - res_mx["membrane"],
        d_basal=res_no["basal"] - res_mx["basal"],
        d_driving=res_no["driving"] - res_mx["driving"],
        d_residual=res_no["residual"] - res_mx["residual"],
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), constrained_layout=True)
    pairs = (
        ("basal |τ_b|", res_no["basal"], res_mx["basal"], res_no["basal"] - res_mx["basal"]),
        ("membrane |div M|", res_no["membrane"], res_mx["membrane"], res_no["membrane"] - res_mx["membrane"]),
        ("driving |τ_d|", res_no["driving"], res_mx["driving"], res_no["driving"] - res_mx["driving"]),
        ("|SSA residual|", res_no["residual"], res_mx["residual"], res_no["residual"] - res_mx["residual"]),
    )
    for col, (title, a, b, d) in enumerate(pairs):
        for ax, field, ttl in (
            (axes[0, col], a, f"no · {title}"),
            (axes[1, col], d, f"Δ (no−max) {title}"),
        ):
            if "Δ" in ttl:
                lim = max(
                    abs(float(np.nanpercentile(d, 2))),
                    abs(float(np.nanpercentile(d, 98))),
                    1e-6,
                )
                norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
                img = add_map(ax, res_no["x_km"], res_no["y_km"], field, ttl, cmap="RdBu_r", norm=norm)
            else:
                img = add_map(ax, res_no["x_km"], res_no["y_km"], field, ttl, cmap="viridis")
            fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"SSA residual decomposition — {args.tag}", fontsize=12)
    resid_path = out_dir / "ssa_residual_decomposition.png"
    fig.savefig(resid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics["figures"] = [str(delta_path), str(scatter_path), str(resid_path)]
    summary_path = out_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({
        "corr_full": metrics["log10_eta_corr_full"],
        "corr_grounded": metrics["log10_eta_corr_grounded"],
        "mean_abs_deta_grounded": metrics["delta_eta"]["mean_abs_grounded"],
        "max_abs_deta_grounded": metrics["delta_eta"]["max_abs_grounded"],
    }, indent=2))
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
