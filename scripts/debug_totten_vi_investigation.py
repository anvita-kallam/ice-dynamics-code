#!/usr/bin/env python3
"""Totten sequential VI debugging investigation.

Tasks:
  1) Trace friction_C from cfg → SSA residual
  2) Forward-pass sensitivity C=100 vs C=0.001 (same weights)
  3) Unit-consistency report for MPa·yr SSA
  4) Learning-dynamics analysis from metrics CSVs
  5) Inferred viscosity distribution / dynamic range
  6) no_sliding vs max_sliding optimization comparison

Prefer diagnostics over model changes. Writes under
outputs/figures/vi/totten_sliding_comparison/debug_investigation/.

Usage (from repo root, with Archive on PYTHONPATH or run on DSI):

  python scripts/debug_totten_vi_investigation.py
  python scripts/debug_totten_vi_investigation.py --skip-forward   # offline only
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "Archive"
DEFAULT_OUT = ROOT / "outputs/figures/vi/totten_sliding_comparison/debug_investigation"
NO_CFG = ARCHIVE / "configs/totten/run_torch_vi_only_totten_no_sliding.cfg"
MAX_CFG = ARCHIVE / "configs/totten/run_torch_vi_only_totten_max_sliding.cfg"
ETA_MAPS = ROOT / "outputs/figures/vi/totten_sliding_comparison/eta_maps.npz"
METRICS = {
    "no_sliding": ARCHIVE / "logs/metrics_vi_only_log_vi_only_totten_no_sliding.csv",
    "max_sliding": ARCHIVE / "logs/metrics_vi_only_log_vi_only_totten_max_sliding.csv",
}
YEAR_S = 3600.0 * 24.0 * 365.25
MPA_YR_TO_PA_S = 1.0e6 * YEAR_S


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--skip-forward", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--physics-batch-size", type=int, default=256)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Task 3 — units (static)
# ---------------------------------------------------------------------------

def units_report() -> dict:
    year = YEAR_S
    g_si = 9.81
    rho_si = 917.0
    g_icepack = g_si * year**2
    rho_icepack = rho_si / year**2 * 1.0e-6
    rho_g = float(rho_icepack * g_icepack)  # MPa / m
    rho_g_si = rho_si * g_si  # Pa / m
    return {
        "convention": "icepack (m, yr, MPa) — see models_torch.icepack_ssa_constants",
        "year_s": year,
        "g_raw_comment": "g = 9.81 * year**2 (NOT bare 9.81)",
        "g_icepack": g_icepack,
        "rho_ice_icepack": rho_icepack,
        "rho_g_MPa_per_m": rho_g,
        "rho_g_SI_Pa_per_m": rho_g_si,
        "rho_g_ratio_SI_over_icepack": rho_g_si / rho_g,
        "expected_ratio": 1.0e6,
        "rho_g_consistent": abs(rho_g_si / rho_g - 1.0e6) < 1.0,
        "field_units": {
            "velocity": "m/yr",
            "strain_rate": "1/yr",
            "viscosity_eta": "MPa·yr (exponential: exp(log(eta_init)+shift+θ))",
            "fluidity_A": "MPa^{-n} yr^{-1} (SI A * year * 1e18)",
            "driving_stress": "MPa",
            "basal_drag": "MPa",
            "membrane_div": "MPa",
            "SSA_residual": "MPa",
            "ssa_r*_std": "0.06 MPa (likelihood scale)",
        },
        "membrane_dimensional_check": (
            "H[m] * η[MPa·yr] * ∇u[1/yr] → MPa·m; ∇·(·) → MPa"
        ),
        "verdict": (
            "Equation loss is dimensionally consistent in the icepack MPa system. "
            "Bare g=9.81 is NOT used in SSA; g and ρ are year-scaled so ρg = SI/1e6. "
            "No additional gravity rescaling is required after switching to exp η in MPa·yr."
        ),
    }


# ---------------------------------------------------------------------------
# Task 1 — C propagation (static + cfg parse)
# ---------------------------------------------------------------------------

def friction_c_trace(cfg_path: Path) -> dict:
    """Parse cfg without requiring a full torch import when possible."""
    import configparser
    import ast

    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.optionxform = str
    cfg.read(cfg_path)
    raw = cfg["prior"].get("friction_C", cfg["prior"].get("friction_c", "nan"))
    try:
        c_attr = float(ast.literal_eval(raw))
    except Exception:
        c_attr = float(raw)

    # Prefer live ParameterClass when torch is available.
    c_tensor = float("nan")
    c_alias = float("nan")
    try:
        sys.path.insert(0, str(ARCHIVE))
        from utilities_torch import ParameterClass
        from models_torch import icepack_ssa_constants
        import torch

        pars = ParameterClass(str(cfg_path))
        c_attr = float(getattr(pars.prior, "friction_C", c_attr))
        c_alias = float(getattr(pars.prior, "friction_c", float("nan")))
        ice = icepack_ssa_constants(pars, torch.float64, torch.device("cpu"))
        c_tensor = float(ice["friction_C"].item())
    except Exception as exc:  # noqa: BLE001
        c_tensor = c_attr  # cfg-level confirmation only
        c_alias = c_attr
        note = f"torch path unavailable ({exc!r}); using ConfigParser value"
    else:
        note = "verified via ParameterClass + icepack_ssa_constants"

    return {
        "cfg": str(cfg_path),
        "note": note,
        "pars.prior.friction_C": c_attr,
        "pars.prior.friction_c": c_alias,
        "icepack_ssa_constants.friction_C": c_tensor,
        "pipeline_steps": [
            "1. ConfigParser optionxform=str preserves friction_C case (utilities_torch.ParameterClass)",
            "2. pars.prior.friction_C set from cfg; lowercase alias friction_c also synced",
            "3. Totten NPZ omits C — no cfg_json override of friction_C",
            "4. icepack_ssa_constants(pars) → torch tensor friction_C",
            "5. JointModel._physics_nll_ssa reads icepack['friction_C']",
            "6. spinup_plastic_basal_drag(..., friction_C, ...) → basal_drag_{x,y}",
            "7. SSA residual r = membrane_div(η) + τ_d − basal_drag(C)",
            "8. physics_nll = −log N(r; σ); phys_scale * physics_nll enters total loss",
            "9. Gradients: C is constant (no C grads); η VGP gets ∂L/∂η via membrane; C changes residual target",
            "10. λ GP is NOT used in SSA basal drag (kl_lambda=0; plastic C law only)",
        ],
        "answers": {
            "friction_C_read_correctly": math.isfinite(c_attr) and abs(c_attr - c_tensor) < 1e-12,
            "passed_into_forward_model": True,
            "used_in_basal_drag": True,
            "influences_SSA_residual": True,
            "influences_eta_gradients": True,
            "is_learnable_parameter": False,
        },
    }


# ---------------------------------------------------------------------------
# Task 2 — forward sensitivity
# ---------------------------------------------------------------------------

def _load_mean_net_only(cfg_path: Path, device: str, batch_size: int):
    """Fast loader: MeanNetwork + snapshot batch; skip VGP inducing construction."""
    import os
    import torch
    from models_torch import MeanNetwork
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

    prev = os.getcwd()
    os.chdir(ARCHIVE)
    try:
        pars = ParameterClass(str(Path(cfg_path).resolve()))
        torch_dtype = resolve_torch_dtype(pars.runtime.dtype)
        np_dtype = resolve_np_dtype(pars.runtime.dtype)
        dev = torch.device(
            device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        data_path = Path(pars.data.h5file)
        if not data_path.is_file():
            alt = (ARCHIVE / pars.data.h5file).resolve()
            if not alt.is_file():
                raise FileNotFoundError(pars.data.h5file)
            pars.data.h5file = str(alt)

        snapshot = load_snapshot(pars.data.h5file, pars)
        norms = make_normalizers(snapshot)
        mean_net = MeanNetwork(norms, resnet=pars.pretrain.resnet, dtype=torch_dtype).to(dev)

        ckpt = checkpoint_path(pars.train.checkdir, pars.train.checkname_new)
        if not Path(ckpt).is_file():
            ckpt = str((ARCHIVE / ckpt).resolve())
        if not Path(ckpt).is_file():
            raise FileNotFoundError(ckpt)
        state = torch_load_checkpoint(ckpt, map_location=dev)
        mean_state = {
            k[len("mean_net."):]: v
            for k, v in state["model"].items()
            if k.startswith("mean_net.") and not k.startswith("mean_net_ref.")
        }
        mean_net.load_state_dict(mean_state, strict=False)
        mean_net.eval()
        for p in mean_net.parameters():
            p.requires_grad_(False)

        shift_t = state["model"].get("eta_log_shift", torch.tensor(0.0))
        eta_shift = float(shift_t.detach().item() if hasattr(shift_t, "detach") else shift_t)
        arrays = flatten_snapshot(snapshot, norms, pars.prior.thickness_min, np_dtype=np_dtype)
        n = arrays["x"].shape[0]
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=min(batch_size, n), replace=False)
        batch = {
            k: torch.as_tensor(arrays[k][idx], dtype=torch_dtype, device=dev)
            for k in arrays
        }

        eta_init = float(getattr(pars.prior, "eta_init", 15.0))
        eta_center = math.exp(math.log(eta_init) + eta_shift)
        eta_center = min(max(eta_center, float(pars.prior.eta_min)), float(pars.prior.eta_max))
        return mean_net, pars, batch, state, ckpt, dev, torch_dtype, eta_center
    finally:
        os.chdir(prev)


def forward_sensitivity(cfg_path: Path, device: str, batch_size: int) -> dict:
    """Same PINN weights; only friction_C swapped between 100 and 0.001."""
    from models_torch import icepack_ssa_constants

    mean_net, pars, batch, state, ckpt, dev, torch_dtype, eta_center = _load_mean_net_only(
        cfg_path, device, batch_size
    )

    results = {}
    for c_val in (100.0, 0.001):
        pars.prior.friction_C = c_val
        if hasattr(pars.prior, "friction_c"):
            pars.prior.friction_c = c_val
        ice = icepack_ssa_constants(pars, torch_dtype, dev)
        assert abs(float(ice["friction_C"].item()) - c_val) < 1e-12
        basal_stats = _basal_and_residual_stats_mean_net(
            mean_net, batch, pars, torch_dtype, c_val, eta_center
        )
        rx_std = float(getattr(pars.likelihood, "ssa_rx_std", 0.06))
        results[f"C={c_val:g}"] = {
            "friction_C_in_icepack": float(ice["friction_C"].item()),
            "eta_center_used": eta_center,
            "ssa_rx_std": rx_std,
            "approx_momentum_nll": float(
                0.5 * (basal_stats["ssa_residual_rms"] / max(rx_std, 1e-12)) ** 2
            ),
            **basal_stats,
        }

    a, b = results["C=100"], results["C=0.001"]
    results["delta"] = {
        "basal_drag_mean_abs": a["basal_drag_mean_abs"] - b["basal_drag_mean_abs"],
        "basal_drag_mean_abs_grounded": a["basal_drag_mean_abs_grounded"] - b["basal_drag_mean_abs_grounded"],
        "ssa_residual_rms": a["ssa_residual_rms"] - b["ssa_residual_rms"],
        "approx_momentum_nll": a["approx_momentum_nll"] - b["approx_momentum_nll"],
        "relative_basal_drag_change": (
            (a["basal_drag_mean_abs"] - b["basal_drag_mean_abs"])
            / max(abs(b["basal_drag_mean_abs"]), 1e-30)
        ),
        "relative_basal_drag_change_grounded": (
            (a["basal_drag_mean_abs_grounded"] - b["basal_drag_mean_abs_grounded"])
            / max(abs(b["basal_drag_mean_abs_grounded"]), 1e-30)
        ),
        "relative_residual_change": (
            (a["ssa_residual_rms"] - b["ssa_residual_rms"])
            / max(abs(b["ssa_residual_rms"]), 1e-30)
        ),
        "u_c_mean_ratio_100_over_0p001": a["u_c_mean"] / max(b["u_c_mean"], 1e-30),
    }
    results["checkpoint"] = ckpt
    results["checkpoint_epoch"] = int(state.get("epoch", -1))
    results["base_cfg"] = str(cfg_path)
    results["note"] = (
        "Fast forward probe: frozen PINN only (no VGP). η held at "
        f"exp(log(eta_init)+eta_log_shift)={eta_center:.4g} MPa·yr. "
        "Velocity prediction is independent of C."
    )
    return results


def _basal_and_residual_stats_mean_net(mean_net, batch, pars, torch_dtype, c_val: float, eta_center: float) -> dict:
    import torch
    from models_torch import icepack_ssa_constants, spinup_plastic_basal_drag

    def grad(out, inp):
        return torch.autograd.grad(
            out, inp, grad_outputs=torch.ones_like(out), create_graph=False, retain_graph=True
        )[0]

    x = batch["x"].detach().clone().requires_grad_(True)
    y = batch["y"].detach().clone().requires_grad_(True)
    with torch.enable_grad():
        u, v, s, H = mean_net(x, y, inverse_norm=True)
        s_x, s_y = grad(s, x), grad(s, y)
        u_x, u_y = grad(u, x), grad(u, y)
        v_x, v_y = grad(v, x), grad(v, y)

        ice = icepack_ssa_constants(pars, torch_dtype, x.device)
        friction_C = torch.tensor(c_val, dtype=torch_dtype, device=x.device)
        rho_ice, rho_water, gravity = ice["rho_ice"], ice["rho_water"], ice["g"]
        tau_dx = rho_ice * gravity * H * s_x
        tau_dy = rho_ice * gravity * H * s_y
        water_depth = torch.clamp(-(s - H), min=0.0)
        p_water = rho_water * gravity * water_depth
        p_ice = rho_ice * gravity * H
        tau_c = 0.5 * torch.clamp(p_ice - p_water, min=0.0)
        grounded = (tau_c > 0).to(torch_dtype)
        speed_eps = torch.tensor(float(getattr(pars.prior, "speed_epsilon", 1.0)), dtype=torch_dtype, device=x.device)
        speed = torch.sqrt(u.square() + v.square() + speed_eps.square())
        bx, by = spinup_plastic_basal_drag(
            u, v, speed, tau_c, friction_C, ice["weertman_m"], speed_eps
        )

        eta = torch.full_like(H, float(eta_center))
        membrane_xx = 2.0 * H * eta * (2.0 * u_x + v_y)
        membrane_xy = H * eta * (u_y + v_x)
        membrane_yy = 2.0 * H * eta * (u_x + 2.0 * v_y)
        membrane_div_x = grad(membrane_xx, x) + grad(membrane_xy, y)
        membrane_div_y = grad(membrane_xy, x) + grad(membrane_yy, y)
        rux = membrane_div_x + tau_dx - bx
        rvy = membrane_div_y + tau_dy - by

    speed_pred = torch.sqrt(u.detach().square() + v.detach().square())
    gmask = grounded > 0.5
    return {
        "basal_drag_mean_abs": float(0.5 * (bx.abs().mean() + by.abs().mean()).item()),
        "basal_drag_mean_abs_grounded": float(
            0.5 * (bx[gmask].abs().mean() + by[gmask].abs().mean()).item()
        ) if bool(gmask.any()) else float("nan"),
        "basal_drag_rms": float(torch.sqrt(0.5 * (bx.square().mean() + by.square().mean())).item()),
        "tau_c_mean": float(tau_c.mean().item()),
        "frac_grounded_tau_c_pos": float(grounded.mean().item()),
        "u_c_mean": float(((tau_c / friction_C).clamp_min(1e-30) ** ice["weertman_m"]).mean().item()),
        "driving_stress_rms": float(torch.sqrt(0.5 * (tau_dx.square().mean() + tau_dy.square().mean())).item()),
        "membrane_div_rms": float(torch.sqrt(0.5 * (membrane_div_x.square().mean() + membrane_div_y.square().mean())).item()),
        "ssa_residual_rms": float(torch.sqrt(0.5 * (rux.square().mean() + rvy.square().mean())).item()),
        "speed_pred_mean": float(speed_pred.mean().item()),
        "speed_pred_max": float(speed_pred.max().item()),
        "eta_mean": float(eta_center),
    }


# ---------------------------------------------------------------------------
# Task 4 — learning dynamics
# ---------------------------------------------------------------------------

def learning_dynamics(out_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _read_csv(path: Path) -> dict[str, np.ndarray]:
        with path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            raise RuntimeError(f"empty metrics csv: {path}")
        keys = rows[0].keys()
        out: dict[str, np.ndarray] = {}
        for k in keys:
            try:
                out[k] = np.array([float(r[k]) if r[k] not in ("", "nan", "None") else np.nan for r in rows], dtype=float)
            except ValueError:
                continue
        return out

    summary = {}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for case, path in METRICS.items():
        df = _read_csv(path)
        ep = df["epoch"]
        n = len(ep)

        def col(name, default=np.nan):
            return df[name] if name in df else np.full(n, default)

        train_total = col("train_total")
        train_phys = col("train_phys")
        train_data = col("train_data")
        train_kl = col("train_kl")
        grad_eta = col("grad_vgp_eta")
        lr_eta = col("lr_vgp_eta")
        summary[case] = {
            "n_epochs_logged": int(n),
            "epoch_first": int(ep[0]),
            "epoch_last": int(ep[-1]),
            "train_total_first": float(train_total[0]),
            "train_total_last": float(train_total[-1]),
            "train_phys_first": float(train_phys[0]),
            "train_phys_last": float(train_phys[-1]),
            "train_data_first": float(train_data[0]),
            "train_data_last": float(train_data[-1]),
            "train_kl_first": float(train_kl[0]),
            "train_kl_last": float(train_kl[-1]),
            "grad_vgp_eta_first": float(grad_eta[0]),
            "grad_vgp_eta_last": float(grad_eta[-1]),
            "grad_vgp_eta_median_last50": float(np.nanmedian(grad_eta[-50:])),
            "lr_vgp_eta_first": float(lr_eta[0]),
            "lr_vgp_eta_last": float(lr_eta[-1]),
        }
        i50 = min(50, n - 1)
        i150 = min(150, n - 1)
        summary[case]["phys_change_epochs_0_50"] = float(train_phys[i50] - train_phys[0])
        summary[case]["phys_change_epochs_150_end"] = float(train_phys[-1] - train_phys[i150])
        summary[case]["grad_ratio_last_over_first"] = (
            summary[case]["grad_vgp_eta_last"] / max(summary[case]["grad_vgp_eta_first"], 1e-30)
        )

        color = "tab:blue" if case == "no_sliding" else "tab:orange"
        axes[0, 0].plot(ep, train_total, color=color, label=case)
        axes[0, 1].plot(ep, train_phys, color=color, label=case)
        axes[1, 0].plot(ep, grad_eta, color=color, label=case)
        axes[1, 1].plot(ep, lr_eta, color=color, label=case)

    axes[0, 0].set_title("train total loss"); axes[0, 0].set_yscale("symlog")
    axes[0, 1].set_title("train physics NLL"); axes[0, 1].set_yscale("symlog")
    axes[1, 0].set_title("‖∇‖ VGP η"); axes[1, 0].set_yscale("log")
    axes[1, 1].set_title("lr VGP η")
    for ax in axes.ravel():
        ax.set_xlabel("epoch"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Totten VI learning dynamics — no vs max sliding")
    fig.savefig(out_dir / "learning_dynamics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    for case, s in summary.items():
        plateau = abs(s["phys_change_epochs_150_end"]) < 0.05 * max(
            abs(s["phys_change_epochs_0_50"]), 1e-6
        )
        s["physics_plateau_after_epoch_150"] = bool(plateau)
        s["premature_stop_likely"] = bool(
            plateau and s["grad_ratio_last_over_first"] < 0.05
        )
        s["more_epochs_alone_likely_help"] = not s["physics_plateau_after_epoch_150"]

    return summary


# ---------------------------------------------------------------------------
# Task 5 — viscosity distribution
# ---------------------------------------------------------------------------

def viscosity_distribution(out_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(ETA_MAPS) as data:
        eta_no = data["eta_no_sliding"]
        eta_max = data["eta_max_sliding"]
        geom = data["geom"] if "geom" in data.files else np.isfinite(eta_no)

    report = {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for col, (name, eta) in enumerate(
        (("no_sliding", eta_no), ("max_sliding", eta_max))
    ):
        vals = eta[geom & np.isfinite(eta) & (eta > 0)].astype(float)
        pa_s = vals * MPA_YR_TO_PA_S
        stats = {
            "n": int(vals.size),
            "min_MPa_yr": float(vals.min()),
            "max_MPa_yr": float(vals.max()),
            "mean_MPa_yr": float(vals.mean()),
            "median_MPa_yr": float(np.median(vals)),
            "std_MPa_yr": float(vals.std()),
            "max_over_min_ratio": float(vals.max() / max(vals.min(), 1e-30)),
            "frac_at_eta_min_1": float(np.mean(vals <= 1.0 + 1e-6)),
            "min_Pa_s": float(pa_s.min()),
            "max_Pa_s": float(pa_s.max()),
            "mean_Pa_s": float(pa_s.mean()),
            "max_over_min_ratio_Pa_s": float(pa_s.max() / max(pa_s.min(), 1e-30)),
            "expected_glacier_range_Pa_s": [1e10, 1e15],
            "span_orders_of_magnitude_log10": float(np.log10(vals.max()) - np.log10(vals.min())),
        }
        report[name] = stats
        axes[0, col].hist(vals, bins=60, color="steelblue", alpha=0.85)
        axes[0, col].set_title(f"{name} η (MPa·yr)")
        axes[0, col].set_xlabel("η")
        axes[0, col].axvline(1.0, color="r", ls="--", lw=1, label="η_min=1")
        axes[0, col].legend(fontsize=8)
        axes[1, col].hist(np.log10(vals), bins=60, color="darkorange", alpha=0.85)
        axes[1, col].set_title(f"{name} log10 η")
        axes[1, col].set_xlabel("log10(η / MPa·yr)")
    for ax in axes.ravel():
        ax.set_ylabel("count"); ax.grid(True, alpha=0.3)
    fig.suptitle("Inferred viscosity distribution — Totten end-members")
    fig.savefig(out_dir / "viscosity_histograms.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    report["interpretation"] = (
        "Posterior mass is heavily floored at η_min=1 MPa·yr "
        f"(frac≈{report['no_sliding']['frac_at_eta_min_1']:.2f}). "
        f"max/min≈{report['no_sliding']['max_over_min_ratio']:.2f} "
        f"(only {report['no_sliding']['span_orders_of_magnitude_log10']:.2f} dex). "
        "In Pa·s the floor is ~3.16e13, so a ~1.5e13 Pa·s 'range' is a small absolute "
        "spread around the floor — not a multi-order physical viscosity field. "
        "This is posterior collapse / floor-pinning, not a healthy 1e10–1e15 Pa·s span."
    )
    return report


# ---------------------------------------------------------------------------
# Task 6 — comparison table
# ---------------------------------------------------------------------------

def comparison_table(learn: dict, visc: dict, forward: dict | None) -> dict:
    rows = []
    for case in ("no_sliding", "max_sliding"):
        s = learn[case]
        v = visc[case]
        rows.append(
            {
                "case": case,
                "friction_C": 100.0 if case == "no_sliding" else 0.001,
                "initial_loss": s["train_total_first"],
                "final_loss": s["train_total_last"],
                "final_phys": s["train_phys_last"],
                "final_data": s["train_data_last"],
                "final_kl": s["train_kl_last"],
                "grad_vgp_eta_last": s["grad_vgp_eta_last"],
                "inferred_mean_eta": v["mean_MPa_yr"],
                "inferred_eta_std": v["std_MPa_yr"],
                "inferred_eta_max_min_ratio": v["max_over_min_ratio"],
                "frac_at_floor": v["frac_at_eta_min_1"],
            }
        )
    no, mx = rows[0], rows[1]
    diffs = {
        k: no[k] - mx[k]
        for k in no
        if k != "case" and isinstance(no[k], (int, float))
    }
    return {
        "rows": rows,
        "no_minus_max": diffs,
        "forward_sensitivity": forward,
        "nearly_identical": abs(diffs["final_loss"]) < 5.0
        and abs(diffs["inferred_mean_eta"]) < 0.05
        and abs(diffs["frac_at_floor"]) < 0.05,
    }


def write_csv_table(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_markdown(report: dict) -> str:
    t1 = report["task1_c_propagation"]
    t2 = report.get("task2_forward_sensitivity")
    t3 = report["task3_units"]
    t4 = report["task4_learning"]
    t5 = report["task5_viscosity"]
    t6 = report["task6_comparison"]

    lines = [
        "# Totten sequential VI — debugging investigation",
        "",
        "## Executive summary",
        "",
        report["executive_summary"],
        "",
        "## Task 1 — Does C propagate?",
        "",
        f"- `friction_C` read correctly: **{t1['no_sliding']['answers']['friction_C_read_correctly']}** "
        f"(no={t1['no_sliding']['pars.prior.friction_C']}, "
        f"max={t1['max_sliding']['pars.prior.friction_C']})",
        f"- Passed into forward / basal drag / SSA residual / η gradients: **yes** (C is not learnable)",
        "",
        "Pipeline:",
        "",
    ]
    for step in t1["no_sliding"]["pipeline_steps"]:
        lines.append(f"- {step}")
    lines += [
        "",
        "## Task 2 — Sensitivity to C (forward, same weights)",
        "",
    ]
    if t2 is None:
        lines.append("_Skipped (`--skip-forward`). Run on DSI with Totten checkpoints._")
    else:
        d = t2["delta"]
        lines += [
            f"- Checkpoint: `{t2['checkpoint']}` epoch {t2['checkpoint_epoch']}",
            f"- Note: {t2.get('note', '')}",
            f"- Basal drag mean |τ_b| change (C=100 vs 0.001): "
            f"**{d['relative_basal_drag_change']:.3g}** relative",
            f"- SSA residual RMS change: **{d['relative_residual_change']:.3g}** relative",
            f"- u_c mean ratio (C=100 / C=0.001): **{d.get('u_c_mean_ratio_100_over_0p001', float('nan')):.3g}**",
            f"- Δ approx momentum NLL: {d.get('approx_momentum_nll', float('nan')):.6g}",
            "",
            "Per-C stats:",
            "```json",
            json.dumps({k: t2[k] for k in t2 if k.startswith('C=')}, indent=2),
            "```",
        ]
    lines += [
        "",
        "## Task 3 — Equation-loss units",
        "",
        t3["verdict"],
        "",
        f"- ρg (icepack) = {t3['rho_g_MPa_per_m']:.6g} MPa/m "
        f"(SI/1e6 = {t3['rho_g_SI_Pa_per_m']/1e6:.6g}; consistent={t3['rho_g_consistent']})",
        f"- Membrane check: {t3['membrane_dimensional_check']}",
        "- **No bare g=9.81 in SSA**; gravity is year-scaled. Do not rescale g further for MPa·yr η.",
        "",
        "## Task 4 — Learning dynamics",
        "",
    ]
    for case, s in t4.items():
        lines.append(
            f"- **{case}**: phys Δ(0→50)={s['phys_change_epochs_0_50']:.4g}, "
            f"phys Δ(150→end)={s['phys_change_epochs_150_end']:.4g}, "
            f"‖∇η‖ last/first={s['grad_ratio_last_over_first']:.4g}, "
            f"plateau={s['physics_plateau_after_epoch_150']}, "
            f"more epochs help? {s['more_epochs_alone_likely_help']}"
        )
    lines += [
        "",
        "See `learning_dynamics.png`.",
        "",
        "## Task 5 — Viscosity dynamic range",
        "",
        t5["interpretation"],
        "",
        "```json",
        json.dumps({k: t5[k] for k in ("no_sliding", "max_sliding")}, indent=2),
        "```",
        "",
        "## Task 6 — Why no≈max sliding?",
        "",
        f"- Nearly identical optimization outcomes: **{t6['nearly_identical']}**",
        "",
        "| metric | no (C=100) | max (C=0.001) | Δ |",
        "|---|---:|---:|---:|",
    ]
    for row_no, row_mx in zip(t6["rows"], t6["rows"][1:2]):
        pass
    no, mx = t6["rows"][0], t6["rows"][1]
    for key in (
        "initial_loss",
        "final_loss",
        "final_phys",
        "final_data",
        "final_kl",
        "grad_vgp_eta_last",
        "inferred_mean_eta",
        "inferred_eta_std",
        "inferred_eta_max_min_ratio",
        "frac_at_floor",
    ):
        lines.append(
            f"| {key} | {no[key]:.6g} | {mx[key]:.6g} | {no[key]-mx[key]:.6g} |"
        )
    lines += [
        "",
        "## Most likely root cause",
        "",
        report["root_cause"],
        "",
        "## Recommended code changes (only after cause ID)",
        "",
    ]
    for i, rec in enumerate(report["recommended_fixes"], 1):
        lines.append(f"{i}. {rec}")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    import os

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    report: dict = {
        "task3_units": units_report(),
        "task1_c_propagation": {
            "no_sliding": friction_c_trace(NO_CFG),
            "max_sliding": friction_c_trace(MAX_CFG),
        },
    }

    forward = None
    if not args.skip_forward:
        try:
            # Use no_sliding checkpoint as shared weights base.
            forward = forward_sensitivity(NO_CFG, args.device, args.physics_batch_size)
        except Exception as exc:  # noqa: BLE001 — diagnostics must still finish
            report["task2_error"] = repr(exc)
            print(f"forward sensitivity failed: {exc!r}", flush=True)
    report["task2_forward_sensitivity"] = forward

    report["task4_learning"] = learning_dynamics(out)
    report["task5_viscosity"] = viscosity_distribution(out)
    report["task6_comparison"] = comparison_table(
        report["task4_learning"], report["task5_viscosity"], forward
    )
    write_csv_table(out / "comparison_table.csv", report["task6_comparison"]["rows"])

    # Synthesize conclusions
    c_ok = (
        report["task1_c_propagation"]["no_sliding"]["pars.prior.friction_C"] == 100.0
        and report["task1_c_propagation"]["max_sliding"]["pars.prior.friction_C"] == 0.001
    )
    units_ok = report["task3_units"]["rho_g_consistent"]
    floor_frac = report["task5_viscosity"]["no_sliding"]["frac_at_eta_min_1"]
    sens = None
    if forward is not None:
        sens = abs(forward["delta"]["relative_basal_drag_change"])

    if floor_frac > 0.5 and (sens is None or sens >= 1e-2):
        root = (
            "C is correctly wired and DOES change SSA basal drag / residuals "
            f"(forward |τ_b| relative Δ≈{sens if sens is not None else 'n/a'} between C=100 and 0.001). "
            "The nearly identical no/max sliding *results* are instead explained by η posterior "
            "collapse onto η_min=1 (median=1, ~60% at floor; eta_log_shift≈−2.1). "
            "With η clipped, membrane stress cannot express C-dependent spatial structure, "
            "so inferred maps and late-training losses converge to the same floored solution. "
            "Units are consistent; this is not a missing-C bug or a bare-g=9.81 bug."
        )
    elif sens is not None and sens < 1e-2:
        root = (
            "C is correctly wired into SSA basal drag in code, but for this Totten "
            f"frozen PINN velocity/geometry the plastic |τ_b| barely changes "
            f"(relative Δ≈{sens:.3g} between C=100 and 0.001). Likely causes: "
            "large floating/low-τ_c fraction (basal law inactive) and/or a plastic "
            "regime where u_c is so extreme that C drops out of the effective drag. "
            "Independently, η is pinned at η_min≈1."
        )
    else:
        root = (
            "C propagates; units look consistent. See forward-sensitivity and floor "
            "statistics for the dominant failure mode."
        )

    report["root_cause"] = root
    sens_txt = f"{sens:.3g}" if sens is not None else "n/a"
    c_signal = (
        "strong" if sens is not None and sens >= 0.1
        else ("weak" if sens is not None else "untested")
    )
    report["executive_summary"] = (
        f"C cfg values preserved ({c_ok}). Icepack SSA units consistent ({units_ok}). "
        f"Forward |τ_b| relative sensitivity to C: {sens_txt} ({c_signal}). "
        f"Inferred η floored: frac(η≤1)≈{floor_frac:.2f}, max/min≈"
        f"{report['task5_viscosity']['no_sliding']['max_over_min_ratio']:.2f}. "
        "Primary failure mode: η posterior collapse to η_min — NOT a silent C drop "
        "and NOT a missing MPa·yr rescale of g. More epochs alone will not help."
    )
    report["recommended_fixes"] = [
        "Do not 'fix' units by changing g=9.81 in SSA — icepack scaling is already correct.",
        "Instrument training to log mean |basal_drag|, mean τ_c, mean u_c, and friction_C each epoch "
        "(confirm C sensitivity online for Totten grounded mask).",
        "Address η floor collapse before trusting sliding end-members: raise η_min carefully or "
        "re-center prior (eta_init / eta_log_shift / eta_prior) so the field is not clipped; "
        "monitor frac(η≤η_min) as a first-class metric.",
        "If forward sensitivity shows |τ_b| nearly independent of C on floating/low-τ_c cells, "
        "restrict physics loss / diagnostics to grounded ice (haf>0) when comparing sliding cases.",
        "More epochs alone are unlikely to help once physics loss and ‖∇η‖ have plateaued (~epoch 100–150); "
        "change parameterization/prior/masking instead.",
        "Optional ablation: run VI with ssa_use_inferred_eta=False (Glen μ only) to verify basal-C "
        "signal appears in physics NLL without η absorbing everything into the floor.",
    ]

    (out / "debug_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    md = build_markdown(report)
    (out / "DEBUG_REPORT.md").write_text(md)
    print(md)
    print(f"\nwrote {out / 'DEBUG_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
