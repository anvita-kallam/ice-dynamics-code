#!/usr/bin/env python3
"""Posterior sampling from a VI-only (frozen PINN) checkpoint."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import h5py
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

_predict.JOINT_ARCHITECTURE = VI_ONLY_ARCHITECTURE

usage = """
Usage: predict_vi_only_torch.py run_torch_vi_only.cfg [num_samples]
"""


def main(pars, num_samples=None):
    torch_dtype = resolve_torch_dtype(pars.runtime.dtype)
    np_dtype = resolve_np_dtype(pars.runtime.dtype)
    device = torch.device(
        'cuda' if pars.torch.device != 'cpu' and torch.cuda.is_available() else 'cpu')
    snapshot = load_snapshot(pars.data.h5file, pars)
    norms = make_normalizers(snapshot)
    arrays = flatten_snapshot(snapshot, norms, pars.prior.thickness_min, np_dtype=np_dtype)

    mean_net = MeanNetwork(norms, resnet=pars.pretrain.resnet, dtype=torch_dtype)
    x_ref = snapshot.x[snapshot.geom_mask]
    y_ref = snapshot.y[snapshot.geom_mask]
    model = JointModel(
        mean_net,
        make_sparse_vgp(x_ref, y_ref, norms, pars, 'eta', torch_dtype),
        make_sparse_vgp(x_ref, y_ref, norms, pars, 'lambda', torch_dtype),
        dtype=torch_dtype,
    ).to(device)

    ckpt_file = checkpoint_path(pars.train.checkdir, pars.train.checkname_new)
    print(f'loaded checkpoint: {ckpt_file}')
    state = torch_load_checkpoint(ckpt_file, map_location=device)
    _predict.require_joint_checkpoint_architecture(state, ckpt_file)
    model_state = {
        k: v for k, v in state['model'].items() if not k.startswith('mean_net_ref.')
    }
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    unexpected = [k for k in unexpected if not k.startswith('mean_net_ref.')]
    if missing or unexpected:
        raise RuntimeError(
            f'Checkpoint load mismatch for {ckpt_file}: missing={missing} unexpected={unexpected}')

    mean_net = model.mean_net
    vgp_eta = model.vgp_eta
    vgp_lambda = model.vgp_lambda
    preds = _predict._batched_mean_predictions(
        mean_net, arrays, pars.predict.batch_size, pars.prior.thickness_min, device, torch_dtype)

    n_samp = int(num_samples if num_samples is not None else pars.predict.num_samples)
    X = np.column_stack([arrays['x'].squeeze(), arrays['y'].squeeze()])
    Xn = normalize_tensor(
        torch.as_tensor(X, dtype=torch_dtype, device=device),
        mean_net.iW_coord, mean_net.b_coord)

    eta_init = float(getattr(pars.prior, 'eta_init', 12.0))
    eta_log_center = math.log(max(eta_init, 1.0e-12))
    eta_log_shift = float(model.eta_log_shift.detach().cpu().item())
    eta_log_min = math.log(float(pars.prior.eta_min))
    eta_log_max = math.log(float(pars.prior.eta_max))
    lambda_init = float(getattr(pars.prior, 'lambda_init', 0.5))
    lambda_logit_center = math.log(lambda_init / max(1.0 - lambda_init, 1.0e-12))
    lambda_logit_shift = float(model.lambda_logit_shift.detach().cpu().item())

    with torch.no_grad():
        theta_samples = vgp_eta.sample(n_samp, Xn).cpu().numpy()
        lambda_logit_samples = vgp_lambda.sample(n_samp, Xn).cpu().numpy()
        theta_mean, theta_var, _, _, _ = vgp_eta.posterior_stats(Xn)
        theta_mean_np = theta_mean.cpu().numpy()
        theta_std = torch.sqrt(theta_var).cpu().numpy()

    eta_samples = np.exp(np.clip(
        eta_log_center + eta_log_shift + theta_samples, eta_log_min, eta_log_max))
    eta_mean = np.exp(np.clip(
        eta_log_center + eta_log_shift + theta_mean_np, eta_log_min, eta_log_max))
    # MAP ≈ posterior mean of θ for this Gaussian variational family.
    eta_map = eta_mean
    lambda_samples = 1.0 / (1.0 + np.exp(
        -(lambda_logit_center + lambda_logit_shift + lambda_logit_samples)))

    out_path = Path(pars.predict.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, 'w') as h5:
        h5.create_dataset('eta_samples', data=eta_samples)
        h5.create_dataset('eta_mean', data=eta_mean)
        h5.create_dataset('eta_map', data=eta_map)
        h5.create_dataset('eta_latent_std', data=theta_std)
        h5.create_dataset('lambda_samples', data=lambda_samples)
        for key, val in preds.items():
            h5.create_dataset(key, data=val)
        h5.attrs['architecture'] = VI_ONLY_ARCHITECTURE
        h5.attrs['kernel_type'] = vgp_eta.kernel_type
        h5.attrs['anisotropic'] = bool(vgp_eta.anisotropic)
        h5.attrs['num_inducing'] = int(vgp_eta.inducing_index_points.shape[0])
        h5.attrs['training_mode'] = 'vi_only_frozen_pinn'
    print(f'wrote {out_path}')
    return 0


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) not in (1, 2):
        print(usage)
        sys.exit(1)
    num_samples = int(args[1]) if len(args) == 2 else None
    raise SystemExit(main(ParameterClass(args[0]), num_samples=num_samples) or 0)
