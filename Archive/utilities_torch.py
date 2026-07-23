#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
import ast
import configparser
import json
import os
import math
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from scipy.ndimage import distance_transform_edt

import matplotlib.pyplot as plt


def _maybe_save_debug_image(filename, array):
    """Optional snapshot debug PNGs (set DEBUG_SNAPSHOT_PLOTS=1). Off by default for batch jobs."""
    if os.environ.get('DEBUG_SNAPSHOT_PLOTS', '').lower() not in ('1', 'true', 'yes'):
        return
    Path('logs_debugs').mkdir(parents=True, exist_ok=True)
    fig = plt.figure()
    plt.imshow(array)
    plt.colorbar()
    plt.savefig(filename)
    plt.close(fig)


def atomic_torch_save(obj, filename):
    """Write checkpoints atomically (safe for Slurm preemption mid-save)."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_name(filename.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, filename)


class SlurmPreemptMonitor:
    """Track SIGUSR1 from Slurm (--signal=B:USR1@300) for graceful checkpoint+requeue."""

    def __init__(self):
        self.requested = False

    def install(self):
        import signal

        def _handle(signum, frame):
            del signum, frame
            self.requested = True
            print('[slurm] SIGUSR1 received; will checkpoint at next safe point.', flush=True)

        signal.signal(signal.SIGUSR1, _handle)

    def triggered(self):
        return self.requested


def apply_slurm_job_restore_flags(pars):
    """After Slurm requeue, resume from the latest checkpoint in cfg paths."""
    if os.environ.get('SLURM_RESTART_COUNT', '0') == '0':
        return
    if hasattr(pars, 'pretrain'):
        pars.pretrain.restore = True
    if hasattr(pars, 'train'):
        pars.train.restore = True
        pars.train.restore_optimizer = True


def exit_for_slurm_requeue():
    """DSI policy: exit 99 requests automatic job requeue after preemption."""
    raise SystemExit(99)

@dataclass
class Snapshot:
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    s: np.ndarray
    h: np.ndarray
    b: np.ndarray
    u_err: np.ndarray
    v_err: np.ndarray
    s_err: np.ndarray
    h_err: np.ndarray
    b_err: np.ndarray
    geom_mask: np.ndarray
    uv_mask: np.ndarray
    u_fill: np.ndarray
    v_fill: np.ndarray
    shape: tuple
    viscosity: np.ndarray | None = None


default_cfg = """
[runtime]
dtype = 'float64'

[data]
h5file = 'data_snapshot.h5'
x_key = 'x'
y_key = 'y'
u_key = 'u'
v_key = 'v'
s_key = 's'
h_key = 'h'
b_key = 'b'
u_err_key = None
v_err_key = None
s_err_key = None
b_err_key = None
mask_key = None
default_u_err = 25.0
default_v_err = 25.0
default_s_err = 5.0
default_h_err = None
default_b_err = 5.0
# CHANGED: seconds per year — icepack uses the same convention (constants.year).
s2y = 3600 * 24 * 365.25

[prior]
l_scale_eta = 10.0e3
std_eta = 2.0
# CHANGED: effective viscosity prior in icepack units (MPa·yr), matching spin-up NPZ viscosity.
eta_init = 1.0
l_scale_lambda = 10.0e3
std_lambda = 1.0
lambda_init = 0.5
kl = 0.05
num_inducing_x = 28
num_inducing_y = 28
# 'ice_fps': farthest-point sample on ice mask; 'bbox_grid': uniform bbox mesh.
inducing_placement = 'ice_fps'
eta_min = 1.0
eta_max = 1.0e6
thickness_min = 1.0
# CHANGED: speed regularization in m/yr (icepack velocity units).
speed_epsilon = 1.0
tau_floor = 1.0
# CHANGED: icepack SSA rheology/sliding parameters (see icepack/constants.py).
year = 3600*24*365.25
glen_exponent = 3.0
weertman_exponent = 3.0
strain_rate_min = 1.0e-5
# CHANGED: default cold-ice fluidity A and basal friction C (overridden from cfg_json when present).
fluidity_A = 3.985e-13 * 3600*24*365.25 * 1.0e18
friction_C = 1.0
# CHANGED: diagnostic SSA solve does not enforce continuity (icepack IceStream diagnostic).
ssa_enforce_continuity = False
# CHANGED: if False, membrane uses μ_Glen(A, ε); if True (default), μ = inferred η.
ssa_use_inferred_eta = True

[likelihood]
rx_std = 5.0
ry_std = 5.0
rH_std = 5.0
trainable_obs_variance = False

[train]
lr = 0.0002
mean_net_lr = None
vgp_eta_lr = None
vgp_lambda_lr = None
# Freeze mean_net for this many epochs counted from start_epoch of THIS run
# (resume-safe). If None, uses freeze_mean_net_fraction * n_epochs.
freeze_mean_net_fraction = 0.4
freeze_mean_net_epochs = None
freeze_from_run_start = False
restore = False
n_epochs = 1000
batch_size = 2048
physics_batch_size = 512
# Explicit ELBO term weights (data / physics / KL / pretrained-state anchor).
data_scale = 1.0
phys_scale = 2.0
state_reg_scale = 1.0
# Soft log-η prior toward log(eta_init); blocks physics-driven η→0 collapse.
# Keep mild (0.1–0.3) so the prior anchors the mean without flattening spatial η.
eta_prior_scale = 0.2
eta_prior_std = 1.0
# Log a warning when unfrozen mean_net grads dominate vgp_eta by more than this factor.
grad_eta_warn_ratio = 100.0
# Split optimizers: independent PINN vs VGP Adam (optional L-BFGS after unfreeze).
mean_net_optimizer = 'adam'
mean_net_optimizer_after_unfreeze = 'adam'
vgp_optimizer = 'adam'
vgp_steps_per_mean_step = 1
mean_net_grad_clip = None
vgp_grad_clip = None
lbfgs_max_iter = 20
lbfgs_history_size = 10
meannet_checkdir = 'checkpoints/torch_pretrain'
checkdir = 'checkpoints/torch_joint'
logfile = 'log_train_torch'
optimizer = 'adam'
# Joint LR schedule: 'none' | 'cosine' | 'plateau'
lr_scheduler = 'cosine'
lr_scheduler_factor = 0.1
lr_scheduler_patience = 50
quadrature_size = 2
max_steps_per_epoch = None
test_every = 1
eta_eval_every = 10
grad_clip = None
restore_optimizer = False
meannet_checkname = 'model_best'
checkname_old = 'model'
checkname_new = 'model'
require_pretrain_checkpoint = True
verify_pretrain_load = True
verify_pretrain_loss_rtol = 5.0e-2
verify_pretrain_loss_atol = 1.0e-8
verify_joint_restore_load = True

[pretrain]
lr = 0.0002
restore = False
restore_from = 'last'
restore_checkname = None
restore_file = None
n_epochs = 1000
batch_size = 2048
checkdir = 'checkpoints/torch_pretrain'
logfile = 'log_pretrain_torch'
resnet = True
optimizer = 'adam'
max_steps_per_epoch = None
test_every = 1
checkname = 'model'
grad_clip = None

[predict]
batch_size = 8192
num_samples = 200
output_file = 'sia_posterior_samples_torch.h5'

[torch]
backend = 'gloo'
device = 'auto'
num_workers = 0
pin_memory = False
train_drop_last = True
threads = None
master_port = None
"""


class GenericClass:
    pass


def _safe_eval_numeric_expr(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _safe_eval_numeric_expr(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left = _safe_eval_numeric_expr(node.left)
        right = _safe_eval_numeric_expr(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return left ** right
    raise ValueError(f'Unsupported config expression: {ast.dump(node)}')


def parse_config_value(value):
    value = value.strip()
    if value == '':
        raise ValueError('Empty config value is not supported')
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        expr = ast.parse(value, mode='eval')
        return _safe_eval_numeric_expr(expr.body)


class ParameterClass:
    def __init__(self, cfgfile=None):
        cfg = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
        # Preserve key case so friction_C / fluidity_A / rH_std match cfg spelling.
        cfg.optionxform = str
        cfg.read_string(default_cfg)
        sections = ('runtime', 'data', 'prior', 'likelihood', 'train', 'pretrain', 'predict', 'torch')
        for section in sections:
            sub_pars = GenericClass()
            for key, value in cfg[section].items():
                setattr(sub_pars, key, parse_config_value(value))
            _sync_case_aliases(sub_pars)
            setattr(self, section, sub_pars)

        if cfgfile is not None:
            cfg = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
            cfg.optionxform = str
            cfg.read(cfgfile)
            for section in sections:
                sub_pars = getattr(self, section)
                if not cfg.has_section(section):
                    continue
                for key, value in cfg[section].items():
                    setattr(sub_pars, key, parse_config_value(value))
                _sync_case_aliases(sub_pars)
                setattr(self, section, sub_pars)


def _sync_case_aliases(sub_pars):
    """Expose both cfg spelling and lowercase aliases (legacy getattr sites)."""
    for key, value in list(vars(sub_pars).items()):
        lower = key.lower()
        if lower != key and not hasattr(sub_pars, lower):
            setattr(sub_pars, lower, value)


class Normalizer:
    """
    Same affine normalizer semantics as the TensorFlow path.
    """
    def __init__(self, xmin, xmax, pos=False, log=False):
        self.xmin = xmin
        self.xmax = xmax
        raw_denom = xmax - xmin
        self.pos = pos
        self.log = log
        self.log_eps = 0.05

        if (not np.isfinite(raw_denom)) or abs(raw_denom) < 1.0e-12:
            # Constant field: avoid division by zero.
            self.denom = 1.0
            self.xmax = xmin + 1.0
            self.is_constant = True
        else:
            self.denom = raw_denom
            self.is_constant = False

      
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if self.pos:
            return (x - self.xmin) / self.denom
        if self.log:
            xn = (x - self.xmin + self.log_eps) / self.denom
            return np.log(xn)
        return 2.0 * (x - self.xmin) / self.denom - 1.0

    def forward_scale(self, scale):
        if self.pos:
            return scale / self.denom
        if self.log:
            raise NotImplementedError
        return 2.0 * scale / self.denom

    def inverse(self, xn):
        if self.pos:
            return self.denom * xn + self.xmin
        if self.log:
            return self.denom * np.exp(xn) + self.xmin - self.log_eps
        return 0.5 * self.denom * (xn + 1.0) + self.xmin


class IndexedDictDataset(Dataset):
    """Index view over one shared flattened tensor dictionary.

    This avoids materializing separate obs/phys train/test copies of every
    flattened field. DistributedSampler can still shard the index view.
    """

    def __init__(self, arrays, indices, torch_dtype):
        self.keys = sorted(arrays.keys())
        self.arrays = {}
        for key in self.keys:
            tensor = torch.from_numpy(arrays[key])
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=torch_dtype)
            self.arrays[key] = tensor
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.n_samples = int(self.indices.numel())

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return {key: value[real_idx] for key, value in self.arrays.items()}


def split_indices(n_data, train_fraction=0.9, shuffle=True, seed=None):
    return train_test_indices(n_data, train_fraction=train_fraction, shuffle=shuffle, seed=seed)


def resolve_np_dtype(dtype_name):
    return np.float32 if str(dtype_name).lower() == 'float32' else np.float64


def resolve_torch_dtype(dtype_name):
    return torch.float32 if str(dtype_name).lower() == 'float32' else torch.float64


def compute_bounds(x, n_sigma=1.0, method='normal'):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError('Cannot compute bounds on an empty array')
    if method == 'normal':
        mean = np.mean(x)
        std = np.std(x)
        if std == 0.0:
            std = 1.0
        return [mean - n_sigma * std, mean + n_sigma * std]
    if method == 'minmax':
        return [np.nanmin(x), np.nanmax(x)]
    raise ValueError('Unsupported bounds determination method')


def _nearest_fill_1d(values, valid_mask):
    out = np.asarray(values, dtype=np.float64).copy()
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size == 0:
        return np.zeros_like(out)
    if valid_idx.size == out.size:
        return out

    sample_idx = np.arange(out.size)
    nearest = valid_idx[np.argmin(np.abs(sample_idx[:, None] - valid_idx[None, :]), axis=1)]
    out[~valid_mask] = out[nearest[~valid_mask]]
    return out


def fill_missing_nearest_2d(field, valid_mask):
    filled = np.asarray(field, dtype=np.float64).copy()
    mask = np.asarray(valid_mask, dtype=bool)
    filled[~mask] = np.nan

    for row in range(filled.shape[0]):
        if mask[row].any():
            filled[row] = _nearest_fill_1d(filled[row], mask[row])

    current_mask = np.isfinite(filled)
    for col in range(filled.shape[1]):
        if current_mask[:, col].any():
            filled[:, col] = _nearest_fill_1d(filled[:, col], current_mask[:, col])

    if np.isnan(filled).any():
        valid_coords = np.argwhere(np.isfinite(filled))
        if valid_coords.size == 0:
            raise ValueError('Cannot fill geometry field with no valid pixels.')
        missing_coords = np.argwhere(~np.isfinite(filled))
        for row, col in missing_coords:
            dist_sq = (valid_coords[:, 0] - row) ** 2 + (valid_coords[:, 1] - col) ** 2
            nearest = valid_coords[np.argmin(dist_sq)]
            filled[row, col] = filled[nearest[0], nearest[1]]

    return filled

def masked_gradient(field_filled, valid_mask, y_coords, x_coords):
    """
    Compute gradient with one-sided differences at NaN boundaries.
    - Interior valid pixels: central difference (accurate, unchanged)
    - Valid pixels adjacent to NaN: one-sided difference (valid side only)
    - Originally invalid pixels: restored to NaN (count preserved)
    """
    ny, nx = field_filled.shape

    grad_y = np.gradient(field_filled, y_coords, axis=0, edge_order=1)
    grad_x = np.gradient(field_filled, x_coords, axis=1, edge_order=1)

    # ── X-direction ──────────────────────────────────────────────────────────
    left_ok  = np.zeros_like(valid_mask)
    right_ok = np.zeros_like(valid_mask)
    left_ok[:,  1:]  = valid_mask[:, :-1]   # left_ok[i,j]  = valid_mask[i, j-1]
    right_ok[:, :-1] = valid_mask[:, 1:]    # right_ok[i,j] = valid_mask[i, j+1]

    fwd_x = valid_mask & ~left_ok  & right_ok   # left invalid → forward diff
    bwd_x = valid_mask & ~right_ok & left_ok    # right invalid → backward diff

    dx_fwd          = np.empty(nx)
    dx_fwd[:-1]     = np.diff(x_coords)
    dx_fwd[-1]      = dx_fwd[-2]

    dx_bwd          = np.empty(nx)
    dx_bwd[1:]      = np.diff(x_coords)
    dx_bwd[0]       = dx_bwd[1]

    f_right          = np.empty_like(field_filled)
    f_right[:, :-1]  = field_filled[:, 1:]
    f_right[:, -1]   = field_filled[:, -1]

    f_left           = np.empty_like(field_filled)
    f_left[:, 1:]    = field_filled[:, :-1]
    f_left[:, 0]     = field_filled[:, 0]

    if fwd_x.any():
        grad_x[fwd_x] = (f_right[fwd_x] - field_filled[fwd_x]) / dx_fwd[np.where(fwd_x)[1]]
    if bwd_x.any():
        grad_x[bwd_x] = (field_filled[bwd_x] - f_left[bwd_x])  / dx_bwd[np.where(bwd_x)[1]]

    # ── Y-direction ──────────────────────────────────────────────────────────
    top_ok           = np.zeros_like(valid_mask)
    bot_ok           = np.zeros_like(valid_mask)
    top_ok[1:,  :]   = valid_mask[:-1, :]   # top_ok[i,j] = valid_mask[i-1, j]
    bot_ok[:-1, :]   = valid_mask[1:,  :]   # bot_ok[i,j] = valid_mask[i+1, j]

    fwd_y = valid_mask & ~top_ok & bot_ok
    bwd_y = valid_mask & ~bot_ok & top_ok

    dy_fwd          = np.empty(ny)
    dy_fwd[:-1]     = np.diff(y_coords)
    dy_fwd[-1]      = dy_fwd[-2]

    dy_bwd          = np.empty(ny)
    dy_bwd[1:]      = np.diff(y_coords)
    dy_bwd[0]       = dy_bwd[1]

    f_below          = np.empty_like(field_filled)
    f_below[:-1, :]  = field_filled[1:, :]
    f_below[-1,  :]  = field_filled[-1, :]

    f_above          = np.empty_like(field_filled)
    f_above[1:,  :]  = field_filled[:-1, :]
    f_above[0,   :]  = field_filled[0,  :]

    if fwd_y.any():
        grad_y[fwd_y] = (f_below[fwd_y] - field_filled[fwd_y]) / dy_fwd[np.where(fwd_y)[0]]
    if bwd_y.any():
        grad_y[bwd_y] = (field_filled[bwd_y] - f_above[bwd_y]) / dy_bwd[np.where(bwd_y)[0]]

    # ── Restore original NaN mask (valid count unchanged) ────────────────────
    grad_y[~valid_mask] = np.nan
    grad_x[~valid_mask] = np.nan

    return grad_y, grad_x
    
def compute_observed_geometry_fields(snapshot, thickness_min, np_dtype=np.float64):
    x_coords = np.asarray(snapshot.x[0], dtype=np.float64)
    y_coords = np.asarray(snapshot.y[:, 0], dtype=np.float64)

    geom_mask = np.isfinite(snapshot.s) & np.isfinite(snapshot.b)
    s_filled = fill_missing_nearest_2d(snapshot.s, snapshot.geom_mask)
    b_filled = fill_missing_nearest_2d(snapshot.b, snapshot.geom_mask)
    H = np.maximum(s_filled - b_filled, thickness_min)
    s_y, s_x = masked_gradient(s_filled, geom_mask, y_coords, x_coords)
    H_y, H_x = masked_gradient(H,        geom_mask, y_coords, x_coords)    

    rho_ice = 917.0
    g = 9.80665
    tdx = -rho_ice * g * H * s_x
    tdy = -rho_ice * g * H * s_y

    A = (H ** 2) * tdx / 3.0
    B = (H ** 2) * tdy / 3.0
    A_filled = fill_missing_nearest_2d(A, geom_mask)
    B_filled = fill_missing_nearest_2d(B, geom_mask)
    _, A_x = masked_gradient(A_filled, geom_mask, y_coords, x_coords)
    B_y, _ =masked_gradient(B_filled, geom_mask, y_coords, x_coords)

    _maybe_save_debug_image('logs_debugs/debug_tdx.png', tdx)
    _maybe_save_debug_image('logs_debugs/debug_tdy.png', tdy)
    _maybe_save_debug_image('logs_debugs/debug_A.png', A)
    _maybe_save_debug_image('logs_debugs/debug_B.png', B)
    _maybe_save_debug_image('logs_debugs/debug_A_x.png', A_x)
    _maybe_save_debug_image('logs_debugs/debug_B_y.png', B_y)
    
    names_to_fields = {
        'H_obs': H,
        's_x_obs': s_x,
        's_y_obs': s_y,
        'H_x_obs': H_x,
        'H_y_obs': H_y,
        'tdx_obs': tdx,
        'tdy_obs': tdy,
        'A_obs': A,
        'B_obs': B,
        'A_x_obs': A_x,
        'B_y_obs': B_y,
    }
    return {
        name: np.asarray(field, dtype=np_dtype)
        for name, field in names_to_fields.items()
    }


def apply_spinup_cfg_to_pars(pars, cfg):
    """CHANGED: copy spin-up icepack parameters into the VI prior section."""
    if cfg is None:
        return
    if 'C' in cfg:
        pars.prior.friction_C = float(cfg['C'])
    if 'A' in cfg:
        pars.prior.fluidity_A = float(cfg['A'])


def load_snapshot(h5file, pars):
    data = np.load(h5file, allow_pickle=True)
    print(f'reading data file: {h5file}')
    # CHANGED: keep velocity in m/yr to match icepack spin-up NPZ and SSA residuals.
    u = data['ux']
    v = data['uy']
    if 'cfg_json' in data:
        spinup_cfg = json.loads(str(data['cfg_json']))
        apply_spinup_cfg_to_pars(pars, spinup_cfg)
        print(
            'loaded spin-up cfg_json: '
            f"C={getattr(pars.prior, 'friction_C', None)}, "
            f"A={getattr(pars.prior, 'fluidity_A', None)}"
        )
    h = data['h'] if 'h' in data else data['thickness'] if 'thickness' in data else data['s'] - data['bed']
    b = data['bed']
    s = data['s'] if 's' in data else data['surface'] if 'surface' in data else b + h
    u_err = data['ux_err'] if 'ux_err' in data else np.full_like(u, pars.data.default_u_err)
    v_err = data['uy_err'] if 'uy_err' in data else np.full_like(v, pars.data.default_v_err)
    b_err = data['bed_err'] if 'bed_err' in data else np.full_like(b, pars.data.default_b_err)
    s_err = data['surf_err'] if 'surf_err' in data else np.full_like(s, pars.data.default_s_err)
    if 'h_err' in data:
        h_err = data['h_err']
    else:
        # If no direct thickness error is supplied, use independent-error
        # propagation for H = s - b.
        h_err = np.sqrt(s_err ** 2 + b_err ** 2)
    xmin, xmax, ymin, ymax = data['xmin'], data['xmax'], data['ymin'], data['ymax']
    nx, ny = u.shape[1], v.shape[0]
    
    crop_box =  [xmin, xmax, ymin, ymax] #full: [xmin, xmax, ymin, ymax] sub: [-1.7e6, -1.48e6, -2.6e5, -0.7e5], tiny: [-1.69e6, -1.51e6, -1.31e5, 0.5e5]
    # data shape: full: (2331, 3805). sub:[950, 1099]. tiny: (904, 899)
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymax, ymin, ny)
    ix = (x >= crop_box[0]) & (x <= crop_box[1])
    iy = (y >= crop_box[2]) & (y <= crop_box[3])
    X, Y = np.meshgrid(x, y)
    X, Y = X[iy][:, ix], Y[iy][:, ix]
    
    H, W = int(np.sum(iy)), int(np.sum(ix))
    u, v, s, h, b = u[iy][:, ix], v[iy][:, ix], s[iy][:, ix], h[iy][:, ix], b[iy][:, ix]
    u_err, v_err = u_err[iy][:, ix], v_err[iy][:, ix]
    s_err, h_err, b_err = s_err[iy][:, ix], h_err[iy][:, ix], b_err[iy][:, ix]
    viscosity = None
    if 'viscosity' in data:
        viscosity = np.asarray(data['viscosity'], dtype=float)[iy][:, ix]

    geom_mask = np.isfinite(s) & np.isfinite(h) & np.isfinite(b)
    uv_mask = geom_mask & np.isfinite(u) & np.isfinite(v)

    u_fill = np.where(np.isfinite(u), u, 0.0)
    v_fill = np.where(np.isfinite(v), v, 0.0)

    s = np.where(geom_mask, s, np.nan)
    h = np.where(geom_mask, h, np.nan)
    b = np.where(geom_mask, b, np.nan)

    _maybe_save_debug_image('logs_debugs/debug_u.png', u)
    _maybe_save_debug_image('logs_debugs/debug_v.png', v)
    _maybe_save_debug_image('logs_debugs/debug_u_err.png', u_err)
    _maybe_save_debug_image('logs_debugs/debug_v_err.png', v_err)
    _maybe_save_debug_image('logs_debugs/debug_bed.png', b)
    _maybe_save_debug_image('logs_debugs/debug_bed_err.png', b_err)
    _maybe_save_debug_image('logs_debugs/debug_s.png', s)
    _maybe_save_debug_image('logs_debugs/debug_h.png', h)

    return Snapshot(
        x=X,
        y=Y,
        u=u,
        v=v,
        s=s,
        h=h,
        b=b,
        u_err=np.where(np.isfinite(u_err), u_err, pars.data.default_u_err),
        v_err=np.where(np.isfinite(v_err), v_err, pars.data.default_v_err),
        s_err=np.where(np.isfinite(s_err), s_err, pars.data.default_s_err),
        h_err=np.where(
            np.isfinite(h_err),
            h_err,
            math.sqrt(pars.data.default_s_err ** 2 + pars.data.default_b_err ** 2)
            if pars.data.default_h_err is None else pars.data.default_h_err),
        b_err=np.where(np.isfinite(b_err), b_err, pars.data.default_b_err),
        geom_mask=geom_mask,
        uv_mask=uv_mask,
        u_fill=u_fill,
        v_fill=v_fill,
        shape=(H, W),
        viscosity=viscosity,
    )


def make_normalizers(snapshot):
    geom_mask = snapshot.geom_mask
    uv_mask = snapshot.uv_mask
    return {
        'x': Normalizer(*compute_bounds(snapshot.x[geom_mask], method='minmax')),
        'y': Normalizer(*compute_bounds(snapshot.y[geom_mask], method='minmax')),
        'u': Normalizer(*compute_bounds(snapshot.u[uv_mask], method='minmax')),
        'v': Normalizer(*compute_bounds(snapshot.v[uv_mask], method='minmax')),
        's': Normalizer(*compute_bounds(snapshot.s[geom_mask], method='minmax')),
        'h': Normalizer(*compute_bounds(snapshot.h[geom_mask], method='minmax')),
        'b': Normalizer(*compute_bounds(snapshot.b[geom_mask], method='minmax')),
    }


def flatten_snapshot(snapshot, norms, thickness_min, np_dtype=np.float64):
    idx = snapshot.geom_mask
    geom_fields = compute_observed_geometry_fields(snapshot, thickness_min=thickness_min, np_dtype=np_dtype)
    x = snapshot.x[idx].astype(np_dtype).reshape(-1, 1)
    y = snapshot.y[idx].astype(np_dtype).reshape(-1, 1)
    u_in = snapshot.u_fill[idx].astype(np_dtype).reshape(-1, 1)
    v_in = snapshot.v_fill[idx].astype(np_dtype).reshape(-1, 1)
    s_in = snapshot.s[idx].astype(np_dtype).reshape(-1, 1)
    h_in = snapshot.h[idx].astype(np_dtype).reshape(-1, 1)
    b_in = snapshot.b[idx].astype(np_dtype).reshape(-1, 1)
    uv_mask = snapshot.uv_mask[idx].astype(np_dtype).reshape(-1, 1)
    geom_mask = np.ones_like(uv_mask, dtype=np_dtype)

    u = np.where(snapshot.uv_mask[idx], snapshot.u[idx], 0.0).astype(np_dtype).reshape(-1, 1)
    v = np.where(snapshot.uv_mask[idx], snapshot.v[idx], 0.0).astype(np_dtype).reshape(-1, 1)
    s = snapshot.s[idx].astype(np_dtype).reshape(-1, 1)
    h = snapshot.h[idx].astype(np_dtype).reshape(-1, 1)
    b = snapshot.b[idx].astype(np_dtype).reshape(-1, 1)
    u_err = snapshot.u_err[idx].astype(np_dtype).reshape(-1, 1)
    v_err = snapshot.v_err[idx].astype(np_dtype).reshape(-1, 1)
    s_err = snapshot.s_err[idx].astype(np_dtype).reshape(-1, 1)
    h_err = snapshot.h_err[idx].astype(np_dtype).reshape(-1, 1)
    b_err = snapshot.b_err[idx].astype(np_dtype).reshape(-1, 1)

    u_in_norm = norms['u'](u_in)
    v_in_norm = norms['v'](v_in)
    u_in_norm[uv_mask == 0.0] = 0.0
    v_in_norm[uv_mask == 0.0] = 0.0

    arrays = {
        'x': x,
        'y': y,
        'u_in': u_in_norm,
        'v_in': v_in_norm,
        's_in': norms['s'](s_in),
        'h_in': norms['h'](h_in),
        'b_in': norms['b'](b_in),
        'uv_mask': uv_mask,
        'geom_mask': geom_mask,
        'u': norms['u'](u),
        'v': norms['v'](v),
        's': norms['s'](s),
        'h': norms['h'](h),
        'b': norms['b'](b),
        'u_err': norms['u'].forward_scale(u_err),
        'v_err': norms['v'].forward_scale(v_err),
        's_err': norms['s'].forward_scale(s_err),
        'h_err': norms['h'].forward_scale(h_err),
        'b_err': norms['b'].forward_scale(b_err),
        'flat_index': np.flatnonzero(snapshot.geom_mask).astype(np.int64).reshape(-1, 1)
    }

    for key, field in geom_fields.items():
        arrays[key] = field[idx].reshape(-1, 1)

    for key, value in arrays.items():
        if key == 'flat_index':
            continue
        arrays[key] = value.astype(np_dtype, copy=False)

    return arrays


def train_test_indices(n_data, train_fraction=0.9, shuffle=True, seed=None):
    rng = np.random.RandomState(seed=seed)
    n_train = int(np.floor(train_fraction * n_data))
    if shuffle:
        indices = rng.permutation(n_data)
    else:
        indices = np.arange(n_data, dtype=int)
    return indices[:n_train], indices[n_train:]


def split_arrays(arrays, train_fraction=0.9, shuffle=True, seed=None):
    first_key = next(iter(arrays))
    train_idx, test_idx = train_test_indices(
        arrays[first_key].shape[0], train_fraction=train_fraction, shuffle=shuffle, seed=seed
    )
    train_arrays = {key: value[train_idx] for key, value in arrays.items()}
    test_arrays = {key: value[test_idx] for key, value in arrays.items()}
    return train_arrays, test_arrays


class DictDataset(Dataset):

    def __init__(self, arrays, torch_dtype):
        self.keys = sorted(arrays.keys())
        self.arrays = {}
        for key in self.keys:
            tensor = torch.from_numpy(arrays[key])
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=torch_dtype)
            self.arrays[key] = tensor
        self.n_samples = self.arrays[self.keys[0]].shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.arrays.items()}


def create_loader(dataset, batch_size, shuffle, drop_last, pars, distributed=False):
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=pars.torch.num_workers,
        pin_memory=pars.torch.pin_memory
    )


def make_pretrain_datasets(pars, torch_dtype):
    np_dtype = resolve_np_dtype(pars.runtime.dtype)
    snapshot = load_snapshot(pars.data.h5file, pars)
    norms = make_normalizers(snapshot)
    arrays = flatten_snapshot(snapshot, norms, pars.prior.thickness_min, np_dtype=np_dtype)

    # train_arrays, test_arrays = split_arrays(arrays, train_fraction=0.9, shuffle=True, seed=77)
    # return DictDataset(train_arrays, torch_dtype), DictDataset(test_arrays, torch_dtype), norms, snapshot
    n_data = arrays[next(iter(arrays))].shape[0]
    train_idx, test_idx = split_indices(n_data, train_fraction=0.9, shuffle=True, seed=77)
    return (
        IndexedDictDataset(arrays, train_idx, torch_dtype),
        IndexedDictDataset(arrays, test_idx, torch_dtype),
        norms,
        snapshot,)

def make_joint_datasets(pars, torch_dtype):
    np_dtype = resolve_np_dtype(pars.runtime.dtype)
    snapshot = load_snapshot(pars.data.h5file, pars)
    norms = make_normalizers(snapshot)
    arrays = flatten_snapshot(snapshot, norms, pars.prior.thickness_min, np_dtype=np_dtype)

    # obs_train, obs_test = split_arrays(arrays, train_fraction=0.9, shuffle=True, seed=77)
    # phys_train, phys_test = split_arrays(arrays, train_fraction=0.9, shuffle=True, seed=60)
    n_data = arrays[next(iter(arrays))].shape[0]
    obs_train_idx, obs_test_idx = split_indices(n_data, train_fraction=0.9, shuffle=True, seed=77)
    phys_train_idx, phys_test_idx = split_indices(n_data, train_fraction=0.9, shuffle=True, seed=60)

    return (
        # DictDataset(obs_train, torch_dtype),
        # DictDataset(obs_test, torch_dtype),
        # DictDataset(phys_train, torch_dtype),
        # DictDataset(phys_test, torch_dtype),
        IndexedDictDataset(arrays, obs_train_idx, torch_dtype),
        IndexedDictDataset(arrays, obs_test_idx, torch_dtype),
        IndexedDictDataset(arrays, phys_train_idx, torch_dtype),
        IndexedDictDataset(arrays, phys_test_idx, torch_dtype),
        norms,
        snapshot,)


def move_batch_to_device(batch, device, torch_dtype):
    out = {}
    for key, value in batch.items():
        value = value.to(device=device, non_blocking=True)
        if value.is_floating_point():
            value = value.to(dtype=torch_dtype)
        out[key] = value
    return out


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def maybe_set_torch_threads(pars):
    if pars.torch.threads is not None:
        torch.set_num_threads(pars.torch.threads)


def torch_load_checkpoint(filename, map_location=None):
    try:
        return torch.load(filename, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(filename, map_location=map_location)


def checkpoint_path(checkdir, checkname):
    return str(Path(checkdir) / f'{checkname}.pt')


def restore_full_grid(flat_values, snapshot, fill_value=np.nan):
    flat_values = np.asarray(flat_values)
    if flat_values.ndim == 1:
        flat_values = flat_values[None, :]

    full_shape = (flat_values.shape[0],) + snapshot.shape
    full = np.full(full_shape, fill_value, dtype=flat_values.dtype)
    full[:, snapshot.geom_mask] = flat_values

    if full.shape[0] == 1:
        return full[0]
    return full


def slurm_world_info():
    if 'SLURM_NTASKS' not in os.environ:
        return None
    return {
        'world_size': int(os.environ['SLURM_NTASKS']),
        'rank': int(os.environ.get('SLURM_PROCID', '0')),
        'local_rank': int(os.environ.get('SLURM_LOCALID', '0')),
    }


def _first_slurm_hostname():
    nodelist = os.environ.get('SLURM_NODELIST')
    if not nodelist:
        return os.environ.get('SLURMD_NODENAME')
    try:
        output = subprocess.check_output(
            ['scontrol', 'show', 'hostnames', nodelist],
            text=True
        )
        hostnames = [line.strip() for line in output.splitlines() if line.strip()]
        if hostnames:
            return hostnames[0]
    except Exception:
        pass
    return os.environ.get('SLURMD_NODENAME')


def configure_slurm_ddp_env(pars):
    info = slurm_world_info()
    if info is None:
        return None

    os.environ.setdefault('WORLD_SIZE', str(info['world_size']))
    os.environ.setdefault('RANK', str(info['rank']))
    os.environ.setdefault('LOCAL_RANK', str(info['local_rank']))

    master_addr = os.environ.get('MASTER_ADDR')
    if not master_addr:
        master_addr = _first_slurm_hostname()
        if master_addr:
            os.environ['MASTER_ADDR'] = master_addr

    master_port = os.environ.get('MASTER_PORT')
    if not master_port:
        if pars.torch.master_port is not None:
            master_port = str(pars.torch.master_port)
        else:
            job_id = int(os.environ.get('SLURM_JOB_ID', '0'))
            master_port = str(15000 + (job_id % 20000))
        os.environ['MASTER_PORT'] = master_port

    return info
