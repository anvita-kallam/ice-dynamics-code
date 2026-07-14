#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
import math
import sys

import h5py
import numpy as np
import torch

from models_torch import JointModel, MeanNetwork, SparseVariationalGP, normalize_tensor
from utilities_torch import (
    ParameterClass,
    checkpoint_path,
    flatten_snapshot,
    load_snapshot,
    make_normalizers,
    resolve_np_dtype,
    resolve_torch_dtype,
    restore_full_grid,
    fill_missing_nearest_2d,
    torch_load_checkpoint,
)
import matplotlib.pyplot as plt

usage = """
Usage: predict_torch.py run_torch.cfg [num_samples]
"""

JOINT_ARCHITECTURE = 'coordinate_only_joint_model_predict_s_h_eta_lambda_v1'
MEAN_NET_OUTPUTS = ('u', 'v', 's', 'h')


def normalized_length_scale(length_m, norms):
    dx = float(norms['x'].denom)
    dy = float(norms['y'].denom)
    domain = math.sqrt(dx * dy)
    return 2.0 * float(length_m) / domain


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def require_joint_checkpoint_architecture(checkpoint, checkpoint_file):
    arch = checkpoint.get('architecture') if isinstance(checkpoint, dict) else None
    outputs = tuple(checkpoint.get('mean_net_outputs', ())) if isinstance(checkpoint, dict) else ()
    if arch != JOINT_ARCHITECTURE or outputs != MEAN_NET_OUTPUTS:
        raise RuntimeError(
            f'Checkpoint {checkpoint_file} was saved with architecture={arch!r}, '
            f'mean_net_outputs={outputs!r}; expected architecture={JOINT_ARCHITECTURE!r}, '
            f'mean_net_outputs={MEAN_NET_OUTPUTS!r}. Re-run prediction using a joint checkpoint '
            'trained with the s+h MeanNetwork.'
        )


def _batched_mean_predictions(mean_net, arrays, batch_size, thickness_min, device, torch_dtype):
    n_total = arrays['x'].shape[0]
    outs = {'u': [], 'v': [], 's': [], 'h': [], 'b': []}

    mean_net.eval()
    with torch.no_grad():
        for start in range(0, n_total, batch_size):
            stop = min(start + batch_size, n_total)
            sl = slice(start, stop)
            up, vp, sp, hp = mean_net(
                torch.as_tensor(arrays['x'][sl], dtype=torch_dtype, device=device),
                torch.as_tensor(arrays['y'][sl], dtype=torch_dtype, device=device),
                # torch.as_tensor(arrays['u_in'][sl], dtype=torch_dtype, device=device),
                # torch.as_tensor(arrays['v_in'][sl], dtype=torch_dtype, device=device),
                # torch.as_tensor(arrays['s_in'][sl], dtype=torch_dtype, device=device),
                # torch.as_tensor(arrays['h_in'][sl], dtype=torch_dtype, device=device),
                # torch.as_tensor(arrays['b_in'][sl], dtype=torch_dtype, device=device),
                # torch.as_tensor(arrays['uv_mask'][sl], dtype=torch_dtype, device=device),
                inverse_norm=True
            )
            # hp = torch.clamp(hp, min=thickness_min)
            bp = sp - hp
            outs['u'].append(up.cpu().numpy().squeeze())
            outs['v'].append(vp.cpu().numpy().squeeze())
            outs['s'].append(sp.cpu().numpy().squeeze())
            outs['h'].append(hp.cpu().numpy().squeeze())
            outs['b'].append(bp.cpu().numpy().squeeze())
    for key, values in outs.items():
        print(key)
    return {key: np.concatenate(values, axis=0) for key, values in outs.items()}

def pixelwise_corr_fast(eta_samples, lambda_samples, log_eta=True, eps=1e-12):
    eta = np.asarray(eta_samples, dtype=float)
    lam = np.asarray(lambda_samples, dtype=float)

    if log_eta:
        eta = np.log10(np.clip(eta, eps, None))

    # mask invalid values
    valid = np.isfinite(eta) & np.isfinite(lam)

    # replace invalid with nan so nanmean/nanstd work
    eta = np.where(valid, eta, np.nan)
    lam = np.where(valid, lam, np.nan)

    eta_mean = np.mean(eta, axis=0)
    lam_mean = np.mean(lam, axis=0)

    eta_anom = eta - eta_mean
    lam_anom = lam - lam_mean

    cov = np.mean(eta_anom * lam_anom, axis=0)
    eta_std = np.std(eta, axis=0)
    lam_std = np.std(lam, axis=0)

    corr = cov / (eta_std * lam_std + eps)

    # mark low-sample pixels invalid
    count = np.sum(valid, axis=0)
    corr[count < 3] = np.nan

    return corr

def main(pars, num_samples=None):
    torch_dtype = resolve_torch_dtype(pars.runtime.dtype)
    np_dtype = resolve_np_dtype(pars.runtime.dtype)
    device = torch.device('cuda' if pars.torch.device != 'cpu' and torch.cuda.is_available() else 'cpu')
    #---- load data -----#
    snapshot = load_snapshot(pars.data.h5file, pars)
    norms = make_normalizers(snapshot)
    arrays = flatten_snapshot(snapshot, norms, pars.prior.thickness_min, np_dtype=np_dtype)
    #---- set network -----#
    mean_net = MeanNetwork(norms, resnet=pars.pretrain.resnet, dtype=torch_dtype)
    x_ref = snapshot.x[snapshot.geom_mask]
    y_ref = snapshot.y[snapshot.geom_mask]
    eta_length_scale = normalized_length_scale(pars.prior.l_scale_eta, norms)
    lambda_length_scale = normalized_length_scale(pars.prior.l_scale_lambda, norms)

    inducing_placement = getattr(pars.prior, 'inducing_placement', 'ice_fps')
    vgp_eta = SparseVariationalGP(
        x_ref, y_ref,
        pars.prior.num_inducing_x, pars.prior.num_inducing_y, norms,
        trainable_obs_variance=pars.likelihood.trainable_obs_variance,
        amplitude_init=pars.prior.std_eta,
        length_scale_init=eta_length_scale,
        dtype=torch_dtype,
        inducing_placement=inducing_placement)
    vgp_lambda = SparseVariationalGP(
        x_ref, y_ref,
        pars.prior.num_inducing_x, pars.prior.num_inducing_y, norms,
        trainable_obs_variance=pars.likelihood.trainable_obs_variance,
        amplitude_init=pars.prior.std_lambda,
        length_scale_init=lambda_length_scale,
        dtype=torch_dtype,
        inducing_placement=inducing_placement)
    #---- load checkpoints -----#
    # The no-reference model has two trainable scalar offsets in JointModel,
    # so load the full JointModel state rather than only the three submodules.
    model = JointModel(mean_net, vgp_eta, vgp_lambda, dtype=torch_dtype).to(device)
    ckpt_file = checkpoint_path(pars.train.checkdir, pars.train.checkname_new)
    print(f'loaded checkpoint: {ckpt_file}')
    state = torch_load_checkpoint(ckpt_file, map_location=device)
    require_joint_checkpoint_architecture(state, ckpt_file)
    model.load_state_dict(state['model'], strict=True)
    mean_net = model.mean_net
    vgp_eta = model.vgp_eta
    vgp_lambda = model.vgp_lambda
    #---- predict NN-----#
    preds = _batched_mean_predictions(mean_net, arrays, pars.predict.batch_size, pars.prior.thickness_min, device, torch_dtype)  
    # H = arrays['H_obs'].squeeze()       # with unit
    tdx = arrays['tdx_obs'].squeeze()   # with unit
    # tdy = arrays['tdy_obs'].squeeze()   # with unit
    #---- compute ref viscosity -----#
    u_hat, v_hat, s_hat, h_hat, b_hat = preds['u'], preds['v'], preds['s'], preds['h'], preds['b']
    H = np.maximum(h_hat, float(getattr(pars.prior, 'thickness_min', 0.0)))
    idx = snapshot.geom_mask
    x_coords = np.asarray(snapshot.x[0])
    y_coords = np.asarray(snapshot.y[:,0])
    s_grid = restore_full_grid(s_hat, snapshot)  # 2D, NaN outside geom_mask

    # Fill only for numerical gradient calculation.
    # Do not use this filled grid as the final saved prediction.
    s_grid_filled = fill_missing_nearest_2d(s_grid, snapshot.geom_mask)
    s_y, s_x = np.gradient(s_grid_filled, y_coords, x_coords, edge_order=1)
    tdx = -917.0 * 9.80665 * H * s_x[idx]
    tdy = -917.0 * 9.80665 * H * s_y[idx]

    speed_eps = pars.prior.speed_epsilon 
    speed_hat = np.sqrt(u_hat** 2 + v_hat** 2 + speed_eps ** 2)
    # optional: set u<10 to 10 to avoid blowing numerator
    speed_hat[(speed_hat > 0) & (speed_hat < 5)] = 5
    speed_hat[(speed_hat > -5) & (speed_hat < 0)] = -5   
    tau_mag = np.sqrt(tdx ** 2 + tdy ** 2 + 1.0e-12)
    tau_eff = np.sqrt(tau_mag**2 + pars.prior.tau_floor**2)
    # tau_eff is retained only for the optional lambda_diag diagnostic.

    #---- predict VI-----#
    x = torch.as_tensor(arrays['x'], dtype=torch_dtype, device=device)
    y = torch.as_tensor(arrays['y'], dtype=torch_dtype, device=device)
    X = torch.cat([x, y], dim=1)
    Xn = normalize_tensor(X, mean_net.iW_coord, mean_net.b_coord)

    if num_samples is None:
        num_samples = pars.predict.num_samples

    with torch.no_grad():
        theta_eta = vgp_eta.sample(num_samples, Xn).cpu().numpy()
        theta_lambda = vgp_lambda.sample(num_samples, Xn).cpu().numpy()
        theta_eta_loc = vgp_eta.mean(Xn).cpu().numpy()
        theta_eta_sd = vgp_eta.stddev(Xn).cpu().numpy()
        theta_lambda_loc = vgp_lambda.mean(Xn).cpu().numpy()
        theta_lambda_sd = vgp_lambda.stddev(Xn).cpu().numpy()
    # No spatial reference viscosity and no speed-based reference sliding
    # fraction are used.  The VI fields below learn absolute log-viscosity
    # and sliding logit, with only broad physical bounds and GP smoothness.
    speed_eps = pars.prior.speed_epsilon
    speed = np.sqrt(u_hat**2 + v_hat**2 + speed_eps**2 )
    tau_mag = np.sqrt(tdx**2 + tdy**2 + 1.0e-12)
    tau0 = pars.prior.tau_floor
    tau_eff = np.sqrt(tau_mag**2 + tau0**2)
    eta_ref = 0.5 * H * tau_eff / speed
    # eta_ref = eta_ref.clamp(min=pars.prior.eta_min, max=pars.prior.eta_max)

    # with initial spatial reference  
    eta_init = float(getattr(pars.prior, 'eta_init', np.sqrt(pars.prior.eta_min * pars.prior.eta_max)))
    eta_init = min(max(eta_init, float(pars.prior.eta_min)), float(pars.prior.eta_max))
    eta_log_center = np.log(eta_init)
    eta_log_shift = float(model.eta_log_shift.detach().cpu().item())
    eta_log_min = np.log(float(pars.prior.eta_min))
    eta_log_max = np.log(float(pars.prior.eta_max))

    lambda_init = float(getattr(pars.prior, 'lambda_init', 0.5))
    lambda_init = min(max(lambda_init, 1.0e-6), 1.0 - 1.0e-6)
    lambda_logit_center = np.log(lambda_init / (1.0 - lambda_init))
    lambda_logit_shift = float(model.lambda_logit_shift.detach().cpu().item())

    lambda_loc = sigmoid_np(lambda_logit_center + lambda_logit_shift + theta_lambda_loc)
    lambda_diag = 1 - (u_hat*tdx + v_hat*tdy)/speed_hat/tau_eff
    print(f'theta_lambda range: [{theta_lambda.min()}, {theta_lambda.max()}]')

    #---- compute absolute viscosity and sliding fraction -----#
    eta_log_samples = eta_log_center + eta_log_shift + theta_eta
    eta_log_samples = np.clip(eta_log_samples, eta_log_min, eta_log_max)
    eta_samples = np.exp(eta_log_samples)
    # eta_samples = eta_ref * np.exp(theta_eta)
            
    lambda_logit_samples = lambda_logit_center + lambda_logit_shift + theta_lambda
    lambda_samples = sigmoid_np(lambda_logit_samples)
    
    eta_mean = np.mean(eta_samples, axis=0)
    eta_std = np.std(eta_samples, axis=0)
    eta_log_mean = np.mean(eta_log_samples, axis=0)
    theta_eta_mean = np.mean(theta_eta, axis=0)
    lambda_mean = np.mean(lambda_samples, axis=0)
    lambda_std = np.std(lambda_samples, axis=0)
    corr_map = pixelwise_corr_fast(eta_samples, lambda_samples, log_eta=True)
    print(f'lambda_mean range: [{lambda_mean.min()}, {lambda_mean.max()}]')
    
    with h5py.File(pars.predict.output_file, 'w') as fid:
        fid['x'] = snapshot.x
        fid['y'] = snapshot.y
        fid['geom_mask'] = snapshot.geom_mask.astype(np.uint8)
        fid['uv_mask'] = snapshot.uv_mask.astype(np.uint8)
        fid['u_hat'] = restore_full_grid(preds['u'], snapshot)
        fid['v_hat'] = restore_full_grid(preds['v'], snapshot)
        fid['s_hat'] = restore_full_grid(preds['s'], snapshot)
        fid['h_hat'] = restore_full_grid(preds['h'], snapshot)
        fid['b_hat'] = restore_full_grid(preds['b'], snapshot)
        fid.attrs['eta_init'] = eta_init
        fid.attrs['eta_log_shift'] = eta_log_shift
        fid.attrs['lambda_init'] = lambda_init
        fid.attrs['lambda_logit_shift'] = lambda_logit_shift
        fid['eta_mean'] = restore_full_grid(eta_mean, snapshot)
        fid['eta_std'] = restore_full_grid(eta_std, snapshot)
        fid['eta_log_mean'] = restore_full_grid(eta_log_mean, snapshot)
        fid['theta_eta_mean'] = restore_full_grid(theta_eta_mean, snapshot)
        fid['theta_eta_loc'] = restore_full_grid(theta_eta_loc, snapshot)
        fid['theta_eta_std'] = restore_full_grid(theta_eta_sd, snapshot)
        fid['lambda_mean'] = restore_full_grid(lambda_mean, snapshot)
        fid['lambda_std'] = restore_full_grid(lambda_std, snapshot)
        fid['theta_lambda_mean'] = restore_full_grid(theta_lambda_loc, snapshot)
        fid['theta_lambda_std'] = restore_full_grid(theta_lambda_sd, snapshot)
        fid['lambda_loc'] = restore_full_grid(lambda_loc, snapshot)
        fid['lambda_diag'] = restore_full_grid(lambda_diag, snapshot)
        fid['corr_map'] = restore_full_grid(corr_map, snapshot)
        
        # fid['eta_samples'] = restore_full_grid(eta_samples, snapshot)
        # fid['lambda_samples'] = restore_full_grid(lambda_samples, snapshot)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) not in (1, 2):
        print(usage)
        sys.exit()

    pars = ParameterClass(args[0])
    n_samples = None if len(args) == 1 else int(args[1])
    main(pars, num_samples=n_samples)
