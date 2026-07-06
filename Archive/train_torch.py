#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import math
from pathlib import Path
import logging
import os
import sys
import pprint

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models_torch import JointModel, MeanNetwork, SparseVariationalGP, create_optimizer
from utilities_torch import (
    configure_slurm_ddp_env,
    ParameterClass,
    checkpoint_path,
    create_loader,
    make_joint_datasets,
    maybe_set_torch_threads,
    move_batch_to_device,
    resolve_torch_dtype,
    torch_load_checkpoint,
)

usage = """
Usage:
  python train_torch.py run_torch.cfg
  torchrun --standalone --nproc_per_node=4 train_torch.py run_torch.cfg
  srun python train_torch.py run_torch.cfg
"""

MEAN_NET_ARCHITECTURE = 'coordinate_only_mean_net_predict_s_h_v1'
JOINT_ARCHITECTURE = 'coordinate_only_joint_model_predict_s_h_eta_lambda_v1'
MEAN_NET_OUTPUTS = ('u', 'v', 's', 'h')

def to_dict(obj):
    if hasattr(obj, "__dict__"):
        return {k: to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    else:
        return obj

def resolve_physics_approximation(pars):
    approx = getattr(pars.train, 'physics_approximation', getattr(pars.train, 'physics', 'SIA'))
    approx = str(approx).strip().upper()
    if approx not in ('SIA', 'SSA'):
        raise ValueError(
            f"Unsupported physics approximation {approx!r}. Use either 'SIA' or 'SSA'."
        )
    pars.train.physics_approximation = approx
    return approx
    
def init_distributed(pars):
    configure_slurm_ddp_env(pars)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend=pars.torch.backend, init_method='env://')
        rank = int(os.environ.get('RANK', str(dist.get_rank())))
        local_rank = int(os.environ.get('LOCAL_RANK', str(rank)))
    else:
        rank = 0
        local_rank = 0
    return distributed, rank, local_rank, world_size


def choose_device(pars, local_rank):
    if pars.torch.device == 'cpu':
        return torch.device('cpu')
    if pars.torch.device == 'cuda':
        torch.cuda.set_device(local_rank)
        return torch.device('cuda', local_rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device('cuda', local_rank)
    return torch.device('cpu')


def setup_logging(logfile, rank):
    if rank == 0:
        logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO)


def reduce_mean(value, world_size):
    if world_size == 1:
        return value
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / world_size

def joint_loss(model, batch_obs, batch_phys, grid, weights, pars, torch_dtype, world_size, return_debug=False):
    del world_size
    return model(batch_obs, batch_phys, grid, weights, pars, torch_dtype, return_debug=return_debug)


def module_grad_norm(module):
    grad_sq_sum = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        grad_sq_sum += float(torch.sum(grad * grad).item())
    return grad_sq_sum ** 0.5


def init_debug_running():
    names = ('eta_log', 'theta_eta', 'eta', 'lambda_logit', 'lambda', 'rux', 'rvy', 'rh')
    return {
        'grad_norm_sum': {'mean_net': 0.0, 'vgp_eta': 0.0, 'vgp_lambda': 0.0},
        'field_minmax': {name: {'min': float('inf'), 'max': float('-inf')} for name in names},
        'count': 0,
    }

def update_debug_running(running_debug, step_debug, grad_norms):
    running_debug['count'] += 1
    for name, value in grad_norms.items():
        running_debug['grad_norm_sum'][name] += float(value)
    for field_name, field_stats in step_debug.items():
        running_field = running_debug['field_minmax'][field_name]
        running_field['min'] = min(running_field['min'], float(field_stats['min']))
        running_field['max'] = max(running_field['max'], float(field_stats['max']))


def format_debug_log(epoch, running_debug):
    count = max(running_debug['count'], 1)
    grad_part = ' '.join(
        f'{name}={running_debug["grad_norm_sum"][name] / count:.6e}'
        for name in ('mean_net', 'vgp_eta', 'vgp_lambda'))
    field_parts = []
    for field_name in ('eta_log', 'theta_eta', 'eta', 'lambda_logit', 'lambda', 'rux', 'rvy', 'rh'):
        stats = running_debug['field_minmax'][field_name]
        field_parts.append(f'{field_name}=[{stats["min"]:.6e},{stats["max"]:.6e}]')
    return f'debug {epoch} grad_norms {grad_part} ' + ' '.join(field_parts)


def evaluate(model, obs_loader, phys_loader, device, torch_dtype, grid, weights, pars, world_size):
    model.eval()
    obs_iter = iter(obs_loader)
    phys_iter = iter(phys_loader)
    n_eval = min(len(obs_loader), len(phys_loader))
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    for _ in range(n_eval):
        batch_obs = move_batch_to_device(next(obs_iter), device, torch_dtype)
        batch_phys = move_batch_to_device(next(phys_iter), device, torch_dtype)
        losses = joint_loss(model, batch_obs, batch_phys, grid, weights, pars, torch_dtype, world_size)
        totals[0] += losses[0].detach().to(torch.float64)
        totals[1] += losses[1].detach().to(torch.float64)
        totals[2] += losses[2].detach().to(torch.float64)
        totals[3] += 1.0
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = max(totals[3].item(), 1.0)
    return (totals[:3] / count).tolist()


def next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def resolve_joint_group_lrs(pars):
    base_lr = pars.train.lr
    return {
        'mean_net': base_lr if pars.train.mean_net_lr is None else pars.train.mean_net_lr,
        'vgp_eta': base_lr if pars.train.vgp_eta_lr is None else pars.train.vgp_eta_lr,
        'vgp_lambda': base_lr if pars.train.vgp_lambda_lr is None else pars.train.vgp_lambda_lr,
    }


def resolve_mean_net_freeze_epochs(pars):
    if pars.train.freeze_mean_net_epochs is not None:
        freeze_epochs = int(pars.train.freeze_mean_net_epochs)
    else:
        freeze_epochs = int(math.ceil(pars.train.n_epochs * pars.train.freeze_mean_net_fraction))
    return min(max(freeze_epochs, 0), int(pars.train.n_epochs))


def normalized_length_scale(length_m, norms):
    # SparseVariationalGP kernels use normalized x,y coordinates, while the
    # config values are in meters. Convert meters to the coordinate scale used
    # by normalize_tensor/norms.
    dx = float(norms['x'].denom)
    dy = float(norms['y'].denom)
    domain = math.sqrt(dx * dy)
    return 2.0 * float(length_m) / domain


def set_module_requires_grad(module, requires_grad):
    for param in module.parameters():
        param.requires_grad_(requires_grad)


def first_state_layer(mean_net):
    try:
        return mean_net.state_dense.layers[0]
    except (AttributeError, IndexError) as exc:
        raise RuntimeError(
            'MeanNetwork must expose state_dense.layers[0] for architecture checks.'
        ) from exc


def assert_coordinate_only_mean_net(mean_net):
    first_layer = first_state_layer(mean_net)
    in_features = getattr(first_layer, 'in_features', None)
    if in_features != 2:
        raise RuntimeError(
            f'MeanNetwork is not coordinate-only: first state layer has '
            f'in_features={in_features}, expected 2 for inputs (x, y). '
            'Do not start joint training from this model.'
        )
    for removed_name in ('W_in', 'iW_in', 'b_in'):
        if hasattr(mean_net, removed_name):
            raise RuntimeError(
                f'MeanNetwork still has old observation-input buffer {removed_name!r}; '
                'use the coordinate-only models_torch.py.'
            )


def checkpoint_model_state(checkpoint, checkpoint_file):
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise KeyError(
        f'Checkpoint {checkpoint_file} does not contain a model state dict under key "model".'
    )


def require_checkpoint_architecture(checkpoint, checkpoint_file, expected_architecture, label):
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'{label} checkpoint {checkpoint_file} has no metadata. The current MeanNetwork '
            'expects outputs (u, v, s, h), with bed computed as b=s-h; do not load '
            'an old state-dict-only checkpoint whose fourth output may be bed.'
        )
    arch = checkpoint.get('architecture')
    outputs = tuple(checkpoint.get('mean_net_outputs', ()))
    if arch != expected_architecture or outputs != MEAN_NET_OUTPUTS:
        raise RuntimeError(
            f'{label} checkpoint {checkpoint_file} was saved with architecture={arch!r}, '
            f'mean_net_outputs={outputs!r}; expected architecture={expected_architecture!r}, '
            f'mean_net_outputs={MEAN_NET_OUTPUTS!r}. Redo pretraining/joint training with '
            'the s+h MeanNetwork before using this checkpoint.'
        )


def strict_load_state_dict(module, state_dict, checkpoint_file, component_name):
    """Strict checkpoint loading with a useful architecture-mismatch error.

    This intentionally does not fall back to compatible-only loading. A 7-input
    observation-conditioned MeanNetwork checkpoint must fail here; otherwise the
    joint run can silently start from a corrupted/non-equivalent pretrain state.
    """
    current_state = module.state_dict()
    missing = [name for name in current_state if name not in state_dict]
    unexpected = [name for name in state_dict if name not in current_state]
    shape_mismatch = []
    for name in sorted(set(current_state).intersection(state_dict)):
        if tuple(current_state[name].shape) != tuple(state_dict[name].shape):
            shape_mismatch.append((name, tuple(state_dict[name].shape), tuple(current_state[name].shape)))

    if missing or unexpected or shape_mismatch:
        detail_lines = []
        if shape_mismatch:
            detail_lines.append('shape mismatches:')
            for name, ckpt_shape, model_shape in shape_mismatch[:20]:
                detail_lines.append(f'  {name}: checkpoint {ckpt_shape} vs current {model_shape}')
            if len(shape_mismatch) > 20:
                detail_lines.append(f'  ... {len(shape_mismatch) - 20} more shape mismatches')
        if missing:
            shown = ', '.join(missing[:20])
            detail_lines.append(f'missing keys in checkpoint ({len(missing)}): {shown}')
        if unexpected:
            shown = ', '.join(unexpected[:20])
            detail_lines.append(f'unexpected keys in checkpoint ({len(unexpected)}): {shown}')

        hint = ''
        for first_weight_key in ('state_dense.layers.0.weight', 'mean_net.state_dense.layers.0.weight'):
            first_weight = state_dict.get(first_weight_key)
            if first_weight is not None and len(first_weight.shape) == 2 and first_weight.shape[1] != 2:
                hint = (
                    '\nThis looks like an old observation-conditioned MeanNetwork checkpoint '
                    f'with {first_weight.shape[1]} input features. The coordinate-only model '
                    'requires exactly 2 input features: normalized x and y. '
                    'Redo pretraining with the modified pretrain_solution_torch.py.'
                )
                break
        raise RuntimeError(
            f'Cannot strictly load {component_name} checkpoint {checkpoint_file}.\n'
            + '\n'.join(detail_lines)
            + hint
        )
    module.load_state_dict(state_dict, strict=True)


def mean_net_observation_loss_components(mean_net, batch):
    up, vp, sp, hp = mean_net(batch['x'], batch['y'], inverse_norm=False)

    u_err = batch['u_err'].clamp_min(1e-8)
    v_err = batch['v_err'].clamp_min(1e-8)
    s_err = batch['s_err'].clamp_min(1e-8)
    h_err = batch['h_err'].clamp_min(1e-8)

    u_num = (batch['uv_mask'] * (up - batch['u']).square() / u_err.square()).sum()
    v_num = (batch['uv_mask'] * (vp - batch['v']).square() / v_err.square()).sum()
    s_num = ((sp - batch['s']).square() / s_err.square()).sum()
    h_num = ((hp - batch['h']).square() / h_err.square()).sum()

    u_den = batch['uv_mask'].sum().clamp_min(1.0)
    v_den = batch['uv_mask'].sum().clamp_min(1.0)
    s_den = batch['geom_mask'].sum().clamp_min(1.0)
    h_den = batch['geom_mask'].sum().clamp_min(1.0)

    total_num = u_num + v_num + s_num + h_num
    total_den = u_den + v_den + s_den + h_den
    loss = total_num / total_den
    stats = torch.stack([
        u_num.detach(), v_num.detach(), s_num.detach(), h_num.detach(),
        u_den.detach(), v_den.detach(), s_den.detach(), h_den.detach(),
    ])
    return loss, stats


def evaluate_mean_net_observation(mean_net, loader, device, torch_dtype, world_size):
    was_training = mean_net.training
    mean_net.eval()
    totals = torch.zeros(10, dtype=torch.float64, device=device)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device, torch_dtype)
            loss, parts = mean_net_observation_loss_components(mean_net, batch)
            totals[0] += loss.detach().to(torch.float64)
            totals[1] += 1.0
            totals[2:10] += parts.to(torch.float64)
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if was_training:
        mean_net.train()
    count = totals[1].clamp_min(1.0)
    return {
        'loss': (totals[0] / count).item(),
        'u_loss': (totals[2] / totals[6].clamp_min(1.0)).item(),
        'v_loss': (totals[3] / totals[7].clamp_min(1.0)).item(),
        's_loss': (totals[4] / totals[8].clamp_min(1.0)).item(),
        'h_loss': (totals[5] / totals[9].clamp_min(1.0)).item(),
    }


def verify_checkpoint_metric(current_value, recorded_value, checkpoint_file, label, rtol, atol):
    if recorded_value is None:
        return
    try:
        recorded_value = float(recorded_value)
    except (TypeError, ValueError):
        return
    if not (math.isfinite(current_value) and math.isfinite(recorded_value)):
        return
    allowed = max(float(atol), float(rtol) * max(abs(recorded_value), 1.0))
    diff = abs(float(current_value) - recorded_value)
    if diff > allowed:
        raise RuntimeError(
            f'{label} verification failed for {checkpoint_file}: '
            f'evaluated value={current_value:.10g}, checkpoint value={recorded_value:.10g}, '
            f'abs diff={diff:.3g}, allowed={allowed:.3g}. '
            'This usually means the checkpoint was produced with a different data file, '
            'normalization, train/test split, or MeanNetwork architecture.'
        )


def log_mean_net_obs_stats(prefix, stats):
    logging.info(
        '%s loss=%.10f components u=%.6e v=%.6e s=%.6e h=%.6e',
        prefix,
        stats['loss'], stats['u_loss'], stats['v_loss'], stats['s_loss'], stats['h_loss']
    )


def main(pars):
    physics_approximation = resolve_physics_approximation(pars)
    maybe_set_torch_threads(pars)
    torch_dtype = resolve_torch_dtype(pars.runtime.dtype)
    distributed, rank, local_rank, world_size = init_distributed(pars)
    device = choose_device(pars, local_rank)
    setup_logging(pars.train.logfile, rank)
    if rank == 0:
        logging.info(pprint.pformat(to_dict(pars), indent=2))
        logging.info('using %s physics approximation for PINN loss', physics_approximation)
    if rank == 0 and pars.train.physics_batch_size > 512:
        logging.warning(
            'physics_batch_size=%d may be expensive with batchwise GP KL; '
            'the TensorFlow reference uses 512.',
            pars.train.physics_batch_size
        )

    obs_train, obs_test, phys_train, phys_test, norms, snapshot = make_joint_datasets(pars, torch_dtype)
    obs_train_loader = create_loader(obs_train, pars.train.batch_size, True, pars.torch.train_drop_last, pars, distributed)
    phys_train_loader = create_loader(phys_train, pars.train.physics_batch_size, True, pars.torch.train_drop_last, pars, distributed)
    obs_test_loader = create_loader(obs_test, pars.train.batch_size, False, False, pars, distributed)
    phys_test_loader = create_loader(phys_test, pars.train.physics_batch_size, False, False, pars, distributed)

    steps_this_epoch = max(len(obs_train_loader), len(phys_train_loader))
    for loader in (obs_train_loader, phys_train_loader):
        obs_iter = iter(obs_train_loader)
        phys_iter = iter(phys_train_loader)
        for _ in range(steps_this_epoch):
            batch_obs, obs_iter = next_or_restart(obs_iter, obs_train_loader)
            batch_phys, phys_iter = next_or_restart(phys_iter, phys_train_loader)
    # print(f'example bact obs: {batch_obs}')
    # print(f'example bact phys: {batch_phys}')
    mean_net = MeanNetwork(norms, resnet=pars.pretrain.resnet, dtype=torch_dtype)
    x_ref = snapshot.x[snapshot.geom_mask]
    y_ref = snapshot.y[snapshot.geom_mask]
    eta_length_scale = normalized_length_scale(pars.prior.l_scale_eta, norms)
    lambda_length_scale = normalized_length_scale(pars.prior.l_scale_lambda, norms)

    vgp_eta = SparseVariationalGP(
        x_ref, y_ref,
        pars.prior.num_inducing_x, pars.prior.num_inducing_y, norms,
        trainable_obs_variance=pars.likelihood.trainable_obs_variance,
        amplitude_init=pars.prior.std_eta,
        length_scale_init=eta_length_scale,
        dtype=torch_dtype)
    vgp_lambda = SparseVariationalGP(
        x_ref, y_ref,
        pars.prior.num_inducing_x, pars.prior.num_inducing_y, norms,
        trainable_obs_variance=pars.likelihood.trainable_obs_variance,
        amplitude_init=pars.prior.std_lambda,
        length_scale_init=lambda_length_scale,
        dtype=torch_dtype)
    model = JointModel(mean_net, vgp_eta, vgp_lambda, dtype=torch_dtype).to(device)
    assert_coordinate_only_mean_net(model.mean_net)
    if not pars.train.restore:
        model.vgp_eta.initialize_variational_to_prior()
        model.vgp_lambda.initialize_variational_to_prior()
        
    if rank == 0:
        logging.info(
            'MeanNetwork architecture: coordinate-only state_net(x, y) -> u, v, s, H; bed is derived as b=s-H; first layer in_features=%d',
            first_state_layer(model.mean_net).in_features,
        )

    # load pretrain checkpoint and verify
    pretrain_ckpt = checkpoint_path(pars.train.meannet_checkdir, pars.train.meannet_checkname)
    if not pars.train.restore:
        if Path(pretrain_ckpt).exists():
            state = torch_load_checkpoint(pretrain_ckpt, map_location=device)
            require_checkpoint_architecture(state, pretrain_ckpt, MEAN_NET_ARCHITECTURE, 'mean_net pretrain')
            strict_load_state_dict(
                model.mean_net,
                checkpoint_model_state(state, pretrain_ckpt),
                pretrain_ckpt,
                'mean_net pretrain',)
            assert_coordinate_only_mean_net(model.mean_net)
            verify_pretrain = bool(getattr(pars.train, 'verify_pretrain_load', True))
            if verify_pretrain:
                pretrain_verify_stats = evaluate_mean_net_observation(
                    model.mean_net, obs_test_loader, device, torch_dtype, world_size)
                verify_checkpoint_metric(
                    pretrain_verify_stats['loss'],
                    state.get('test_loss') if isinstance(state, dict) else None,
                    pretrain_ckpt,
                    'Pretrain checkpoint test-loss',
                    getattr(pars.train, 'verify_pretrain_loss_rtol', 5.0e-2),
                    getattr(pars.train, 'verify_pretrain_loss_atol', 1.0e-8),)
                if rank == 0:
                    logging.info('strictly loaded mean_net pretrain checkpoint: %s', pretrain_ckpt)
                    if isinstance(state, dict) and state.get('test_loss') is not None:
                        logging.info('checkpoint recorded test_loss=%.10f', float(state['test_loss']))
                    log_mean_net_obs_stats('pretrain-load verification on obs_test', pretrain_verify_stats)
        else:
            message = f'mean_net pretrain checkpoint not found: {pretrain_ckpt}'
            if bool(getattr(pars.train, 'require_pretrain_checkpoint', True)):
                raise FileNotFoundError(message)
            if rank == 0:
                logging.warning('%s; continuing from random MeanNetwork because require_pretrain_checkpoint=False', message)

    if distributed:
        ddp_kwargs = {'device_ids': [local_rank], 'output_device': local_rank} if device.type == 'cuda' else {}
        ddp_kwargs['find_unused_parameters'] = True
        model = DDP(model, **ddp_kwargs)

    raw_model = model.module if isinstance(model, DDP) else model
    joint_lrs = resolve_joint_group_lrs(pars)
    optimizer = create_optimizer(
        pars.train.optimizer,
        [
            {'params': raw_model.mean_net.parameters(), 'lr': joint_lrs['mean_net']},
            {'params': raw_model.vgp_eta.parameters(), 'lr': joint_lrs['vgp_eta']},
            {'params': raw_model.vgp_lambda.parameters(), 'lr': joint_lrs['vgp_lambda']},
            {'params': [raw_model.eta_log_shift], 'lr': getattr(pars.train, 'eta_shift_lr', joint_lrs['vgp_eta'])},
            {'params': [raw_model.lambda_logit_shift], 'lr': getattr(pars.train, 'lambda_shift_lr', joint_lrs['vgp_lambda'])},
        ],
        learning_rate=pars.train.lr)
    ckpt_file_old = checkpoint_path(pars.train.checkdir, pars.train.checkname_old)
    ckpt_file_new = checkpoint_path(pars.train.checkdir, pars.train.checkname_new)
    Path(pars.train.checkdir).mkdir(parents=True, exist_ok=True)
    print(
        f'rank {rank} local_rank {local_rank} world_size {world_size} '
        f'has {len(obs_train)} local obs samples across {len(obs_train_loader)} local batches'
        f'has {len(phys_train)} local phys samples across {len(phys_train_loader)} local batches')
    
    # restore if there is an old train checkpoint and want to continue training
    n_additional_epochs = int(pars.train.n_epochs)
    if n_additional_epochs < 0:
        raise ValueError(f'pretrain.n_epochs must be non-negative, got {pars.pretrain.n_epochs}')
    stop_epoch = n_additional_epochs
    print(pars.train.restore, ckpt_file_old, Path(ckpt_file_old).exists())
    start_epoch = 0
    stop_epoch = start_epoch + n_additional_epochs
    if pars.train.restore:
        if not Path(ckpt_file_old).exists():
            raise FileNotFoundError(f'train.restore=True but checkpoint does not exist: {ckpt_file_old}')
        state = torch_load_checkpoint(ckpt_file_old, map_location=device)
        require_checkpoint_architecture(state, ckpt_file_old, JOINT_ARCHITECTURE, 'joint model')
        target = model.module if isinstance(model, DDP) else model
        strict_load_state_dict(
            target,
            checkpoint_model_state(state, ckpt_file_old),
            ckpt_file_old,
            'joint model',
        )
        assert_coordinate_only_mean_net(target.mean_net)
        if rank == 0:
            logging.info('strictly loaded previous joint checkpoint: %s', ckpt_file_old)
        restore_optimizer = getattr(pars.train, 'restore_optimizer', False)
        if restore_optimizer:
            try:
                optimizer.load_state_dict(state['optimizer'])
                completed_epoch =state.get('epoch')
                completed_epoch = int(completed_epoch)
                start_epoch = completed_epoch + 1
                stop_epoch = start_epoch + n_additional_epochs
                if rank == 0:
                    logging.info('loaded previous optimizer: %s', ckpt_file_old)
            except (KeyError, ValueError, RuntimeError) as exc:
                if rank == 0:
                    logging.warning(
                        'Skipping optimizer state restore due to missing optimizer state or mismatch: %s', exc)
        elif rank == 0:
            logging.info('NOT loaded previous optimizer: %s', ckpt_file_old)
        if bool(getattr(pars.train, 'verify_joint_restore_load', True)):
            joint_verify_stats = evaluate_mean_net_observation(
                target.mean_net, obs_test_loader, device, torch_dtype, world_size
            )
            if rank == 0:
                log_mean_net_obs_stats('joint-restore verification on obs_test', joint_verify_stats)
    
    # Always enforce cfg learning rates after restore
    optimizer.param_groups[0]['lr'] = joint_lrs['mean_net']
    optimizer.param_groups[1]['lr'] = joint_lrs['vgp_eta']
    optimizer.param_groups[2]['lr'] = joint_lrs['vgp_lambda']

    grid_np, weights_np = np.polynomial.hermite.hermgauss(pars.train.quadrature_size)
    grid = torch.tensor(grid_np, device=device, dtype=torch_dtype)
    weights = torch.tensor(weights_np / np.sqrt(np.pi), device=device, dtype=torch_dtype)

    # set up freeze NN
    last_test = [float('nan')] * 3
    steps_this_epoch = max(len(obs_train_loader), len(phys_train_loader))
    if pars.train.max_steps_per_epoch is not None:
        steps_this_epoch = min(steps_this_epoch, pars.train.max_steps_per_epoch)
    freeze_mean_net_epochs = resolve_mean_net_freeze_epochs(pars)
    mean_net_is_frozen = None

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        for loader in (obs_train_loader, phys_train_loader):
            if distributed and isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(epoch)

        freeze_mean_net = epoch < freeze_mean_net_epochs
        if freeze_mean_net != mean_net_is_frozen:
            set_module_requires_grad(raw_model.mean_net, not freeze_mean_net)
            mean_net_is_frozen = freeze_mean_net
            if rank == 0:
                state_text = 'frozen' if freeze_mean_net else 'unfrozen'
                logging.info(
                    'epoch %d mean_net %s (freeze epochs: %d, lr_mean=%.6e, lr_eta=%.6e, lr_lambda=%.6e)',
                    epoch,
                    state_text,
                    freeze_mean_net_epochs,
                    joint_lrs['mean_net'],
                    joint_lrs['vgp_eta'],
                    joint_lrs['vgp_lambda'],)

        obs_iter = iter(obs_train_loader)
        phys_iter = iter(phys_train_loader)
        running = torch.zeros(4, dtype=torch.float64, device=device)
        running_debug = init_debug_running()
        for _ in range(steps_this_epoch):
            batch_obs, obs_iter = next_or_restart(obs_iter, obs_train_loader)
            batch_phys, phys_iter = next_or_restart(phys_iter, phys_train_loader)
            batch_obs = move_batch_to_device(batch_obs, device, torch_dtype)
            batch_phys = move_batch_to_device(batch_phys, device, torch_dtype)

            optimizer.zero_grad(set_to_none=True)
            losses = joint_loss(
                model, batch_obs, batch_phys, grid, weights, pars, torch_dtype, world_size, return_debug=True)
            total_loss = losses[0] + losses[1] + losses[2]
            total_loss.backward()
            # log norm FIRST
            grad_norms = {
                'mean_net':    module_grad_norm(raw_model.mean_net),
                'vgp_eta':     module_grad_norm(raw_model.vgp_eta),
                'vgp_lambda':  module_grad_norm(raw_model.vgp_lambda),
            }
            # then clip each module to 5, not to 5 globally (double checked is correct)
            if pars.train.grad_clip is not None:
                for module in (raw_model.mean_net, raw_model.vgp_eta, raw_model.vgp_lambda):
                    params = [p for p in module.parameters() if p.requires_grad and p.grad is not None]
                    if params:
                        torch.nn.utils.clip_grad_norm_(params, pars.train.grad_clip)
                shift_params = [
                    p for p in (raw_model.eta_log_shift, raw_model.lambda_logit_shift)
                    if p.requires_grad and p.grad is not None
                ]
                if shift_params:
                    torch.nn.utils.clip_grad_norm_(shift_params, pars.train.grad_clip)
            update_debug_running(running_debug, losses[3], grad_norms)
            optimizer.step()
            # print(f'losses.shape: {len(losses)}, running shape: {running.shape}, losses[0]: {losses[0]}, losses[2]: {losses[2]}')
            running[0] += losses[0].detach().to(torch.float64)
            running[1] += losses[1].detach().to(torch.float64)
            running[2] += losses[2].detach().to(torch.float64)
            running[3] += 1.0

        train_stats = reduce_mean(running[:3] / torch.clamp(running[3], min=1.0), world_size).tolist()
        if epoch % pars.train.test_every == 0:
            last_test = evaluate(model, obs_test_loader, phys_test_loader, device, torch_dtype, grid, weights, pars, world_size)

        if rank == 0:
            train_total = train_stats[0] + train_stats[1] + train_stats[2]
            test_total = last_test[0] + last_test[1] + last_test[2]
            logging.info(
                '%d %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f',
                epoch,
                train_stats[0], train_stats[1], train_stats[2], train_total,
                last_test[0], last_test[1], last_test[2], test_total,)
            logging.info('%s', format_debug_log(epoch, running_debug))
            torch.save(
                {
                    'model': (model.module if isinstance(model, DDP) else model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'architecture': JOINT_ARCHITECTURE,
                    'mean_net_outputs': MEAN_NET_OUTPUTS,
                    'physics_approximation': physics_approximation,
                    'coordinate_only': True,
                    'mean_net_first_layer_in_features': first_state_layer((model.module if isinstance(model, DDP) else model).mean_net).in_features,
                    'kl_note': 'Uses batchwise posterior-vs-prior KL inside JointModel.forward.',
                },
                ckpt_file_new
            )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print(usage)
        sys.exit()
    main(ParameterClass(args[0]))
