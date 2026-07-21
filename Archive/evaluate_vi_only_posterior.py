#!/usr/bin/env python3
"""Evaluate VI-only η posterior quality vs spin-up viscosity.

Writes maps + summary JSON under outputs/figures/vi_only/<tag>/.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import predict_torch as _predict
from models_torch import JointModel, MeanNetwork, make_sparse_vgp, normalize_tensor
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

usage = """
Usage: evaluate_vi_only_posterior.py CONFIG [--tag NAME] [--output-dir DIR]
       [--checkpoint best|latest|PATH] [--no-plots]
"""


def _rank_correlation(x, y):
    """Dependency-free Spearman correlation (ties are negligible for these fields)."""
    x_rank = np.empty(len(x), dtype=float)
    y_rank = np.empty(len(y), dtype=float)
    x_rank[np.argsort(x, kind='mergesort')] = np.arange(len(x), dtype=float)
    y_rank[np.argsort(y, kind='mergesort')] = np.arange(len(y), dtype=float)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _rmse(pred, truth, mask):
    valid = np.asarray(mask, dtype=bool) & np.isfinite(pred) & np.isfinite(truth)
    return float(np.sqrt(np.mean((pred[valid] - truth[valid]) ** 2)))


def main(argv):
    tag = 'more_sliding'
    args = list(argv)
    cfg = None
    output_dir = None
    checkpoint_choice = 'best'
    make_plots = True
    while args:
        tok = args.pop(0)
        if tok == '--tag' and args:
            tag = args.pop(0)
        elif tok == '--output-dir' and args:
            output_dir = Path(args.pop(0))
        elif tok == '--checkpoint' and args:
            checkpoint_choice = args.pop(0)
        elif tok == '--no-plots':
            make_plots = False
        elif cfg is None and not tok.startswith('-'):
            cfg = tok
        else:
            print(usage)
            return 1
    if cfg is None:
        print(usage)
        return 1

    pars = ParameterClass(cfg)
    torch_dtype = resolve_torch_dtype(pars.runtime.dtype)
    device = torch.device(
        'cuda' if pars.torch.device != 'cpu' and torch.cuda.is_available() else 'cpu')
    snapshot = load_snapshot(pars.data.h5file, pars)
    if snapshot.viscosity is None:
        raise RuntimeError('Snapshot has no viscosity field for evaluation')
    norms = make_normalizers(snapshot)
    mean_net = MeanNetwork(norms, resnet=pars.pretrain.resnet, dtype=torch_dtype)
    x_ref = snapshot.x[snapshot.geom_mask]
    y_ref = snapshot.y[snapshot.geom_mask]
    model = JointModel(
        mean_net,
        make_sparse_vgp(x_ref, y_ref, norms, pars, 'eta', torch_dtype),
        make_sparse_vgp(x_ref, y_ref, norms, pars, 'lambda', torch_dtype),
        dtype=torch_dtype,
    ).to(device)
    if checkpoint_choice == 'best':
        ckpt_name = str(getattr(pars.train, 'checkname_best', 'model_best') or 'model_best')
        ckpt = checkpoint_path(pars.train.checkdir, ckpt_name)
    elif checkpoint_choice == 'latest':
        ckpt = checkpoint_path(pars.train.checkdir, pars.train.checkname_new)
    else:
        ckpt = checkpoint_choice
    state = torch_load_checkpoint(ckpt, map_location=device)
    if state.get('architecture') != VI_ONLY_ARCHITECTURE:
        raise RuntimeError(f'Expected VI-only architecture, got {state.get("architecture")!r}')
    model.load_state_dict(
        {k: v for k, v in state['model'].items() if not k.startswith('mean_net_ref.')},
        strict=False)

    geom = snapshot.geom_mask & np.isfinite(snapshot.viscosity) & (snapshot.viscosity > 0)
    ys, xs = np.where(geom)
    x = torch.as_tensor(snapshot.x[ys, xs], dtype=torch_dtype, device=device).reshape(-1, 1)
    y = torch.as_tensor(snapshot.y[ys, xs], dtype=torch_dtype, device=device).reshape(-1, 1)
    Xn = normalize_tensor(torch.cat([x, y], dim=1), model.mean_net.iW_coord, model.mean_net.b_coord)
    with torch.no_grad():
        theta, var, _, _, _ = model.vgp_eta.posterior_stats(Xn)
        theta = theta.cpu().numpy()
        std = torch.sqrt(var).cpu().numpy()

    eta_init = float(pars.prior.eta_init)
    shift = float(model.eta_log_shift.detach().cpu().item())
    eta_log = np.clip(
        math.log(eta_init) + shift + theta,
        math.log(float(pars.prior.eta_min)),
        math.log(float(pars.prior.eta_max)))
    eta_mean = np.exp(eta_log)
    eta_ref = snapshot.viscosity[ys, xs].astype(float)
    log_err = np.log10(eta_mean) - np.log10(eta_ref)
    log10_std = std / math.log(10.0)
    abs_log_err = np.abs(log_err)
    rel_err = (eta_mean - eta_ref) / eta_ref

    arrays = flatten_snapshot(
        snapshot, norms, pars.prior.thickness_min,
        np_dtype=resolve_np_dtype(pars.runtime.dtype))
    state_pred = _predict._batched_mean_predictions(
        model.mean_net, arrays, pars.predict.batch_size,
        pars.prior.thickness_min, device, torch_dtype)
    geom_flat = snapshot.geom_mask
    uv_flat = snapshot.uv_mask[geom_flat]
    state_truth = {
        'u': snapshot.u[geom_flat],
        'v': snapshot.v[geom_flat],
        's': snapshot.s[geom_flat],
        'h': snapshot.h[geom_flat],
        'b': snapshot.b[geom_flat],
    }
    state_masks = {
        'u': uv_flat,
        'v': uv_flat,
        's': np.ones(int(geom_flat.sum()), dtype=bool),
        'h': np.ones(int(geom_flat.sum()), dtype=bool),
        'b': np.ones(int(geom_flat.sum()), dtype=bool),
    }
    state_rmse = {}
    state_nrmse_terms = []
    for name in ('u', 'v', 's', 'h', 'b'):
        pred = np.asarray(state_pred[name]).reshape(-1)
        truth = np.asarray(state_truth[name]).reshape(-1)
        valid = state_masks[name] & np.isfinite(pred) & np.isfinite(truth)
        value = _rmse(pred, truth, valid)
        state_rmse[name] = value
        ref_std = float(np.std(truth[valid]))
        if ref_std > 0.0:
            state_nrmse_terms.append(value / ref_std)

    summary = {
        'config': str(cfg),
        'checkpoint': ckpt,
        'checkpoint_epoch': int(state.get('epoch', -1)),
        'n_points': int(eta_mean.size),
        'log10_eta_rmse': float(np.sqrt(np.mean(log_err ** 2))),
        'log10_eta_bias': float(np.mean(log_err)),
        'log10_eta_r': float(np.corrcoef(np.log10(eta_mean), np.log10(eta_ref))[0, 1]),
        'rel_eta_rmse': float(np.sqrt(np.mean(rel_err ** 2))),
        'eta_pred_mean': float(np.mean(eta_mean)),
        'eta_ref_mean': float(np.mean(eta_ref)),
        'eta_mean_ratio': float(np.mean(eta_mean) / np.mean(eta_ref)),
        'eta_post_var_mean': float(np.mean(std ** 2)),
        'eta_post_std_mean': float(np.mean(std)),
        'eta_post_std_median': float(np.median(std)),
        'eta_post_std_p90': float(np.percentile(std, 90)),
        'log10_eta_post_std_mean': float(np.mean(log10_std)),
        'uncertainty_abs_error_spearman': _rank_correlation(log10_std, abs_log_err),
        'calibration_within_1sigma': float(np.mean(np.abs(log_err) <= log10_std)),
        'calibration_within_2sigma': float(np.mean(np.abs(log_err) <= 2.0 * log10_std)),
        'state_rmse': state_rmse,
        'state_nrmse': float(np.mean(state_nrmse_terms)),
        'kernel': model.vgp_eta.kernel_diagnostics(),
        'mean_equals_map_note': 'Gaussian variational posterior: mean(θ) used as MAP proxy',
    }

    configured_evaldir = getattr(pars.train, 'evaldir', None)
    out_dir = output_dir or (
        Path(str(configured_evaldir)) if configured_evaldir
        else Path('outputs/figures/vi_only') / tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / 'posterior_summary.json'
    tmp_summary = summary_path.with_suffix('.json.tmp')
    tmp_summary.write_text(json.dumps(summary, indent=2))
    tmp_summary.replace(summary_path)

    if not make_plots:
        print(json.dumps(summary, indent=2))
        return 0

    # Scatter: uncertainty vs |error|
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(log10_std, np.abs(log_err), s=4, alpha=0.3)
    axes[0].set_xlabel('posterior std (log10 η)')
    axes[0].set_ylabel('|log10 error|')
    axes[0].set_title('Uncertainty vs reconstruction error')
    axes[0].grid(True, alpha=0.3)
    axes[1].scatter(np.log10(eta_ref), np.log10(eta_mean), s=4, alpha=0.3)
    lims = [
        min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]),
        max(axes[1].get_xlim()[1], axes[1].get_ylim()[1]),
    ]
    axes[1].plot(lims, lims, 'k--', lw=1)
    axes[1].set_xlabel('log10 η truth')
    axes[1].set_ylabel('log10 η posterior mean')
    axes[1].set_title(f"r={summary['log10_eta_r']:.3f}")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / 'uncertainty_vs_error.png', dpi=150)
    plt.close(fig)

    # Full-grid maps
    eta_map = np.full(snapshot.x.shape, np.nan)
    std_map = np.full(snapshot.x.shape, np.nan)
    err_map = np.full(snapshot.x.shape, np.nan)
    eta_map[ys, xs] = eta_mean
    std_map[ys, xs] = std
    err_map[ys, xs] = log_err
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, field, title, cmap in (
        (axes[0], np.log10(np.clip(eta_map, 1e-12, None)), 'log10 η mean', 'viridis'),
        (axes[1], std_map, 'latent θ std', 'magma'),
        (axes[2], err_map, 'log10 η error', 'coolwarm'),
    ):
        im = ax.imshow(field, origin='lower', cmap=cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_dir / 'posterior_maps.png', dpi=150)
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print(f'figures -> {out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
