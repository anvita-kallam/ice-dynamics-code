#!/usr/bin/env python3
"""Compare latest joint and optimized sequential VI recovery on more_sliding."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np

from validate_vi_posterior_more_sliding import (
    _restore_masked_field,
    load_data,
    log10_safe,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NPZ = (
    ROOT
    / "outputs/spinup/production/more_sliding/"
    "SteadyState_more_sliding_10500yr_ramp4000_1refine_grid.npz"
)
DEFAULT_JOINT = ROOT / "Archive/outputs/more_sliding_joint_epoch749_posterior_samples_torch.h5"
DEFAULT_SEQUENTIAL = (
    ROOT / "Archive/outputs/more_sliding_vi_only_optimized_posterior_samples_torch.h5"
)
DEFAULT_FIG_DIR = ROOT / "outputs/figures/vi/joint_vs_sequential_optimized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-h5", type=Path, default=DEFAULT_JOINT)
    parser.add_argument("--sequential-h5", type=Path, default=DEFAULT_SEQUENTIAL)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--stride", type=int, default=2)
    return parser.parse_args()


def eta_metrics(data: dict, mask: np.ndarray) -> dict[str, float]:
    pred = data["eta_mean"][mask]
    truth = data["eta_ref"][mask]
    log_error = log10_safe(pred) - log10_safe(truth)
    log_std = data["eta_log10_std"][mask]
    return {
        "log10_eta_rmse": float(np.sqrt(np.mean(log_error**2))),
        "log10_eta_bias": float(np.mean(log_error)),
        "log10_eta_r": float(np.corrcoef(log10_safe(pred), log10_safe(truth))[0, 1]),
        "rel_eta_rmse": float(np.sqrt(np.mean(((pred - truth) / truth) ** 2))),
        "eta_pred_mean": float(np.mean(pred)),
        "eta_ref_mean": float(np.mean(truth)),
        "calibration_within_1sigma": float(np.mean(np.abs(log_error) <= log_std)),
        "calibration_within_2sigma": float(np.mean(np.abs(log_error) <= 2.0 * log_std)),
    }


def use_log_space_estimator(data: dict, h5_path: Path) -> None:
    """Use exp(E[log η]) and latent std so both pipelines use the same estimator."""
    grid_shape = data["eta_ref"].shape
    geom = data["geom"]
    with h5py.File(h5_path) as h5:
        if "eta_log_mean" in h5:
            eta_log_mean = _restore_masked_field(
                h5["eta_log_mean"][...], grid_shape, geom, "eta_log_mean"
            )
            data["eta_mean"] = np.exp(eta_log_mean)
        if "theta_eta_std" in h5:
            latent_std = _restore_masked_field(
                h5["theta_eta_std"][...], grid_shape, geom, "theta_eta_std"
            )
        elif "eta_latent_std" in h5:
            latent_std = _restore_masked_field(
                h5["eta_latent_std"][...], grid_shape, geom, "eta_latent_std"
            )
        else:
            # Exact for an unclipped lognormal when eta_mean is arithmetic mean.
            cv = data["eta_std"] / np.maximum(data["eta_mean"], 1.0e-12)
            latent_std = np.sqrt(np.log1p(cv**2))
    data["eta_log10_std"] = latent_std / math.log(10.0)


def rmse(values: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(values) & np.isfinite(truth)
    return float(np.sqrt(np.mean((values[valid] - truth[valid]) ** 2)))


def shared_coordinates(data: dict, stride: int) -> tuple[np.ndarray, np.ndarray]:
    s = max(stride, 1)
    x, y = np.asarray(data["x"]), np.asarray(data["y"])
    if x.ndim == 2:
        return x[::s, ::s] / 1e3, y[::s, ::s] / 1e3
    xx, yy = np.meshgrid(x[::s], y[::s])
    return xx / 1e3, yy / 1e3


def add_map(ax, x, y, values, title, *, cmap="viridis", norm=None, limits=None):
    kwargs = {"shading": "auto", "cmap": cmap}
    if norm is not None:
        kwargs["norm"] = norm
    elif limits is not None:
        kwargs["vmin"], kwargs["vmax"] = limits
    image = ax.pcolormesh(x, y, values, **kwargs)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    return image


def finite_symmetric_limit(*fields: np.ndarray, percentile: float = 98.0) -> float:
    values = np.concatenate([np.abs(f[np.isfinite(f)]) for f in fields])
    return max(float(np.percentile(values, percentile)), 1.0e-6)


def save_eta_maps(joint, sequential, mask, fig_dir, stride, metrics):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    s = max(stride, 1)
    x, y = shared_coordinates(sequential, s)
    m = mask[::s, ::s]
    truth = np.where(m, log10_safe(sequential["eta_ref"][::s, ::s]), np.nan)
    joint_eta = np.where(m, log10_safe(joint["eta_mean"][::s, ::s]), np.nan)
    seq_eta = np.where(m, log10_safe(sequential["eta_mean"][::s, ::s]), np.nan)
    joint_error = joint_eta - truth
    seq_error = seq_eta - truth
    estimate_delta = seq_eta - joint_eta
    improvement = np.abs(joint_error) - np.abs(seq_error)

    estimate_values = np.concatenate(
        [truth[m], joint_eta[m], seq_eta[m]]
    )
    estimate_limits = tuple(np.percentile(estimate_values, [2, 98]))
    error_limit = finite_symmetric_limit(joint_error, seq_error)
    delta_limit = finite_symmetric_limit(estimate_delta)
    improvement_limit = finite_symmetric_limit(improvement)

    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), constrained_layout=True)
    top = (
        (truth, r"Truth $\log_{10}\eta$"),
        (joint_eta, r"Joint $\log_{10}\eta$"),
        (seq_eta, r"Sequential $\log_{10}\eta$"),
    )
    for col, (field, title) in enumerate(top):
        image = add_map(axes[0, col], x, y, field, title, cmap="magma", limits=estimate_limits)
        fig.colorbar(image, ax=axes[0, col], fraction=0.046, pad=0.02)
    image = add_map(
        axes[0, 3], x, y, estimate_delta,
        r"Sequential − joint $\log_{10}\eta$", cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-delta_limit, vmax=delta_limit),
    )
    fig.colorbar(image, ax=axes[0, 3], fraction=0.046, pad=0.02)

    for col, (field, title) in enumerate(
        (
            (joint_error, "Joint error"),
            (seq_error, "Sequential error"),
        )
    ):
        image = add_map(
            axes[1, col], x, y, field, title, cmap="RdBu_r",
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-error_limit, vmax=error_limit),
        )
        fig.colorbar(image, ax=axes[1, col], fraction=0.046, pad=0.02)
    image = add_map(
        axes[1, 2], x, y, improvement,
        "Absolute-error reduction\n(positive = sequential better)", cmap="RdBu",
        norm=TwoSlopeNorm(
            vcenter=0.0, vmin=-improvement_limit, vmax=improvement_limit
        ),
    )
    fig.colorbar(image, ax=axes[1, 2], fraction=0.046, pad=0.02)
    seq_better = np.where(m, np.abs(seq_error) < np.abs(joint_error), np.nan)
    image = add_map(
        axes[1, 3], x, y, seq_better,
        "Sequential lower absolute error", cmap="RdYlGn", limits=(0.0, 1.0),
    )
    fig.colorbar(
        image, ax=axes[1, 3], fraction=0.046, pad=0.02,
        ticks=[0, 1], label="0 = joint, 1 = sequential",
    )
    fig.suptitle(
        "Viscosity recovery: joint epoch 749 vs sequential best epoch 54\n"
        f"joint r={metrics['joint']['log10_eta_r']:.3f}, "
        f"sequential r={metrics['sequential']['log10_eta_r']:.3f}",
        fontsize=13,
    )
    path = fig_dir / "eta_joint_vs_sequential_maps.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_eta_diagnostics(joint, sequential, mask, fig_dir, metrics):
    import matplotlib.pyplot as plt

    truth = log10_safe(sequential["eta_ref"][mask])
    joint_eta = log10_safe(joint["eta_mean"][mask])
    seq_eta = log10_safe(sequential["eta_mean"][mask])
    joint_error = joint_eta - truth
    seq_error = seq_eta - truth
    joint_log_std = joint["eta_log10_std"][mask]
    seq_log_std = sequential["eta_log10_std"][mask]

    rng = np.random.default_rng(0)
    indices = np.arange(truth.size)
    if indices.size > 30_000:
        indices = rng.choice(indices, 30_000, replace=False)
    lo = float(min(truth[indices].min(), joint_eta[indices].min(), seq_eta[indices].min()))
    hi = float(max(truth[indices].max(), joint_eta[indices].max(), seq_eta[indices].max()))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), constrained_layout=True)
    axes[0, 0].scatter(truth[indices], joint_eta[indices], s=2, alpha=0.2, label="Joint")
    axes[0, 0].scatter(
        truth[indices], seq_eta[indices], s=2, alpha=0.2, label="Sequential"
    )
    axes[0, 0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0, 0].set(
        xlabel=r"Truth $\log_{10}\eta$",
        ylabel=r"Posterior mean $\log_{10}\eta$",
        title="Truth vs posterior mean",
    )
    axes[0, 0].legend()

    bins = np.linspace(
        min(joint_error.min(), seq_error.min()),
        max(joint_error.max(), seq_error.max()),
        80,
    )
    axes[0, 1].hist(joint_error, bins=bins, density=True, alpha=0.55, label="Joint")
    axes[0, 1].hist(seq_error, bins=bins, density=True, alpha=0.55, label="Sequential")
    axes[0, 1].axvline(0.0, color="k", ls="--", lw=1)
    axes[0, 1].set(
        xlabel=r"$\log_{10}\eta_{\rm estimate}-\log_{10}\eta_{\rm truth}$",
        ylabel="Density",
        title="Error distributions",
    )
    axes[0, 1].legend()

    axes[1, 0].scatter(
        joint_log_std[indices], np.abs(joint_error[indices]),
        s=2, alpha=0.2, label="Joint",
    )
    axes[1, 0].scatter(
        seq_log_std[indices], np.abs(seq_error[indices]),
        s=2, alpha=0.2, label="Sequential",
    )
    axes[1, 0].plot([0, 1], [0, 1], "k--", lw=1, transform=axes[1, 0].transAxes)
    axes[1, 0].set(
        xlabel=r"Posterior std of $\log_{10}\eta$",
        ylabel=r"Absolute $\log_{10}\eta$ error",
        title="Uncertainty vs error",
    )
    axes[1, 0].legend()

    labels = ["RMSE", "|bias|", "1 − r"]
    joint_values = [
        metrics["joint"]["log10_eta_rmse"],
        abs(metrics["joint"]["log10_eta_bias"]),
        1.0 - metrics["joint"]["log10_eta_r"],
    ]
    seq_values = [
        metrics["sequential"]["log10_eta_rmse"],
        abs(metrics["sequential"]["log10_eta_bias"]),
        1.0 - metrics["sequential"]["log10_eta_r"],
    ]
    positions = np.arange(len(labels))
    width = 0.36
    axes[1, 1].bar(positions - width / 2, joint_values, width, label="Joint")
    axes[1, 1].bar(positions + width / 2, seq_values, width, label="Sequential")
    axes[1, 1].set(
        xticks=positions,
        xticklabels=labels,
        ylabel="Metric value (lower is better)",
        title="Recovery error summary",
    )
    axes[1, 1].legend()
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    path = fig_dir / "eta_joint_vs_sequential_diagnostics.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_state_comparison(joint, sequential, mask, fig_dir, stride):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    s = max(stride, 1)
    x, y = shared_coordinates(sequential, s)
    m = mask[::s, ::s]
    fields = [
        (
            "Speed (m/yr)",
            np.hypot(sequential["ux"], sequential["uy"]),
            np.hypot(joint["u_hat"], joint["v_hat"]),
            np.hypot(sequential["u_hat"], sequential["v_hat"]),
        )
    ]
    for label, truth_key, pred_key in (
        ("Surface (m)", "surface", "s_hat"),
        ("Thickness (m)", "thickness", "h_hat"),
        ("Bed (m)", "bed", "b_hat"),
    ):
        if (
            sequential[truth_key] is not None
            and joint[pred_key] is not None
            and sequential[pred_key] is not None
        ):
            fields.append(
                (label, sequential[truth_key], joint[pred_key], sequential[pred_key])
            )

    fig, axes = plt.subplots(
        len(fields), 3, figsize=(13.5, 3.3 * len(fields)), constrained_layout=True
    )
    if len(fields) == 1:
        axes = np.asarray([axes])
    for row, (label, truth, joint_estimate, seq_estimate) in enumerate(fields):
        truth_s = np.where(m, truth[::s, ::s], np.nan)
        joint_error = np.where(m, joint_estimate[::s, ::s] - truth[::s, ::s], np.nan)
        seq_error = np.where(m, seq_estimate[::s, ::s] - truth[::s, ::s], np.nan)
        truth_limits = tuple(np.nanpercentile(truth_s, [2, 98]))
        error_limit = finite_symmetric_limit(joint_error, seq_error)
        image = add_map(
            axes[row, 0], x, y, truth_s, f"Truth {label}", cmap="viridis",
            limits=truth_limits,
        )
        fig.colorbar(image, ax=axes[row, 0], fraction=0.046, pad=0.02)
        for col, (error, title) in enumerate(
            ((joint_error, "Joint error"), (seq_error, "Sequential error")), start=1
        ):
            image = add_map(
                axes[row, col], x, y, error, f"{title}: {label}", cmap="RdBu_r",
                norm=TwoSlopeNorm(
                    vcenter=0.0, vmin=-error_limit, vmax=error_limit
                ),
            )
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.02)
    fig.suptitle("State recovery: joint vs frozen-PINN sequential VI", fontsize=13)
    path = fig_dir / "state_joint_vs_sequential.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    joint = load_data(args.joint_h5, args.npz)
    sequential = load_data(args.sequential_h5, args.npz)
    use_log_space_estimator(joint, args.joint_h5)
    use_log_space_estimator(sequential, args.sequential_h5)
    mask = joint["eval_mask"] & sequential["eval_mask"]

    metrics = {
        "estimator": "exp(E[log eta]); latent posterior std in log10 eta",
        "joint": eta_metrics(joint, mask),
        "sequential": eta_metrics(sequential, mask),
    }
    joint_error = np.abs(
        log10_safe(joint["eta_mean"][mask]) - log10_safe(joint["eta_ref"][mask])
    )
    seq_error = np.abs(
        log10_safe(sequential["eta_mean"][mask])
        - log10_safe(sequential["eta_ref"][mask])
    )
    metrics["comparison"] = {
        "n_common_points": int(mask.sum()),
        "fraction_sequential_lower_abs_log_error": float(np.mean(seq_error < joint_error)),
        "rmse_reduction_fraction": float(
            1.0
            - metrics["sequential"]["log10_eta_rmse"]
            / metrics["joint"]["log10_eta_rmse"]
        ),
        "correlation_gain": float(
            metrics["sequential"]["log10_eta_r"] - metrics["joint"]["log10_eta_r"]
        ),
    }
    metrics["state_rmse"] = {}
    for label, truth_key, pred_key in (
        ("u", "ux", "u_hat"),
        ("v", "uy", "v_hat"),
        ("surface", "surface", "s_hat"),
        ("thickness", "thickness", "h_hat"),
        ("bed", "bed", "b_hat"),
    ):
        truth = sequential[truth_key]
        if truth is None or joint[pred_key] is None or sequential[pred_key] is None:
            continue
        metrics["state_rmse"][label] = {
            "joint": rmse(joint[pred_key], truth, mask),
            "sequential": rmse(sequential[pred_key], truth, mask),
        }

    paths = [
        save_eta_maps(joint, sequential, mask, args.fig_dir, args.stride, metrics),
        save_eta_diagnostics(joint, sequential, mask, args.fig_dir, metrics),
        save_state_comparison(joint, sequential, mask, args.fig_dir, args.stride),
    ]
    summary_path = args.fig_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    for path in [*paths, summary_path]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
