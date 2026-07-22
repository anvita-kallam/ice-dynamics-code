#!/usr/bin/env python3
"""One truth + four-model η estimate panel for completed eta-bias suite runs."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'Archive'
DEFAULT_NPZ = (
    ROOT
    / 'outputs/spinup/production/more_sliding/'
    'SteadyState_more_sliding_10500yr_ramp4000_1refine_grid.npz'
)
DEFAULT_OUT = ROOT / 'outputs/figures/vi/eta_bias_v1_truth_vs_four_estimates.png'

MODELS = (
    ('control_adamw', 'Control\n(η_init=12)'),
    ('raised_prior_center', 'Raised center\n(η_init=15)'),
    ('strong_physics', 'Strong physics'),
    ('weak_prior', 'Weak prior'),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npz', type=Path, default=DEFAULT_NPZ)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--stride', type=int, default=2)
    return parser.parse_args()


def load_eta_mean(archive: Path, run_id: str, device: torch.device):
    import sys

    sys.path.insert(0, str(archive))
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

    cfg = archive / 'configs' / 'vi_only_eta_bias_suite' / f'{run_id}.cfg'
    cwd = Path.cwd()
    os.chdir(archive)
    try:
        pars = ParameterClass(str(cfg.relative_to(archive)))
        torch_dtype = resolve_torch_dtype(pars.runtime.dtype)
        snapshot = load_snapshot(pars.data.h5file, pars)
        norms = make_normalizers(snapshot)
        model = JointModel(
            MeanNetwork(norms, resnet=pars.pretrain.resnet, dtype=torch_dtype),
            make_sparse_vgp(
                snapshot.x[snapshot.geom_mask], snapshot.y[snapshot.geom_mask],
                norms, pars, 'eta', torch_dtype),
            make_sparse_vgp(
                snapshot.x[snapshot.geom_mask], snapshot.y[snapshot.geom_mask],
                norms, pars, 'lambda', torch_dtype),
            dtype=torch_dtype,
        ).to(device)
        ckpt = checkpoint_path(pars.train.checkdir, 'model_best')
        state = torch_load_checkpoint(ckpt, map_location=device)
        if state.get('architecture') != VI_ONLY_ARCHITECTURE:
            raise RuntimeError(f'{ckpt}: expected VI-only architecture')
        model.load_state_dict(
            {k: v for k, v in state['model'].items() if not k.startswith('mean_net_ref.')},
            strict=False)

        geom = snapshot.geom_mask & np.isfinite(snapshot.viscosity) & (snapshot.viscosity > 0)
        ys, xs = np.where(geom)
        x = torch.as_tensor(snapshot.x[ys, xs], dtype=torch_dtype, device=device).reshape(-1, 1)
        y = torch.as_tensor(snapshot.y[ys, xs], dtype=torch_dtype, device=device).reshape(-1, 1)
        Xn = normalize_tensor(torch.cat([x, y], dim=1), model.mean_net.iW_coord, model.mean_net.b_coord)
        with torch.no_grad():
            theta, _, _, _, _ = model.vgp_eta.posterior_stats(Xn)
            theta = theta.cpu().numpy()
        eta_log = np.clip(
            math.log(float(pars.prior.eta_init))
            + float(model.eta_log_shift.detach().cpu().item())
            + theta,
            math.log(float(pars.prior.eta_min)),
            math.log(float(pars.prior.eta_max)))
        eta = np.full(snapshot.x.shape, np.nan, dtype=np.float64)
        eta[ys, xs] = np.exp(eta_log)
        return eta, int(state.get('epoch', -1))
    finally:
        os.chdir(cwd)


def main():
    args = parse_args()
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib_cache'))
    Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    with np.load(args.npz) as data:
        x = data['X'].astype(float)
        y = data['Y'].astype(float)
        truth = data['viscosity'].astype(float)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    estimates = []
    for run_id, label in MODELS:
        eta, epoch = load_eta_mean(args.archive, run_id, device)
        estimates.append((label, eta, epoch))
        print(f'loaded {run_id}: epoch={epoch}')

    s = max(int(args.stride), 1)
    x_km = x[::s, ::s] / 1e3
    y_km = y[::s, ::s] / 1e3
    fields = [truth[::s, ::s]] + [eta[::s, ::s] for _, eta, _ in estimates]
    titles = [r'Truth $\eta$'] + [
        f'{label}\nepoch {epoch}' for label, _, epoch in estimates]
    vals = np.concatenate([
        truth[np.isfinite(truth) & (truth > 0)],
        *[eta[np.isfinite(eta) & (eta > 0)] for _, eta, _ in estimates],
    ])
    norm = LogNorm(
        vmin=max(float(np.percentile(vals, 2)), 1e-3),
        vmax=float(np.percentile(vals, 98)),
    )

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2), constrained_layout=True)
    axes = axes.ravel()
    images = []
    for ax, field, title in zip(axes[:5], fields, titles):
        im = ax.pcolormesh(x_km, y_km, field, shading='auto', cmap='magma', norm=norm)
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('y (km)')
        images.append(im)
    axes[5].axis('off')
    cbar = fig.colorbar(images[0], ax=axes[:5], fraction=0.025, pad=0.02)
    cbar.set_label(r'$\eta$ (MPa·yr)')
    fig.suptitle(
        'Viscosity estimates: truth vs completed eta-bias suite models',
        fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(args.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
