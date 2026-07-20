#!/usr/bin/env python3
"""Stage 2 (sequential): VI on η with a frozen pretrained PINN — no joint updates.

Optimizes only VGP (+ scalar shifts). The MeanNetwork is permanently frozen.
Improvements for spatial-η recovery (kernels, AdamW/NGD, schedulers, early
stopping, diagnostics) live here and do not change train_torch.py joint runs.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models_torch import (
    JointModel,
    MeanNetwork,
    VariationalNaturalGradient,
    create_optimizer,
    make_sparse_vgp,
    normalize_tensor,
)
from training_metrics import (
    VI_ONLY_FIELDS,
    EpochMetricsWriter,
    maybe_plot_training,
    resolve_metrics_csv,
    resolve_plot_dir,
    resolve_plot_every,
    summarize_debug_running,
)
from train_torch import (
    MEAN_NET_ARCHITECTURE,
    MEAN_NET_OUTPUTS,
    assert_coordinate_only_mean_net,
    checkpoint_model_state,
    choose_device,
    clip_module_grads,
    clip_param_grads,
    collect_grad_norms,
    evaluate,
    evaluate_mean_net_observation,
    first_state_layer,
    format_debug_log,
    init_debug_running,
    init_distributed,
    joint_loss,
    log_mean_net_obs_stats,
    next_or_restart,
    record_clip_norm,
    reduce_mean,
    require_checkpoint_architecture,
    resolve_grad_clips,
    resolve_physics_approximation,
    set_module_requires_grad,
    set_vgp_trainable,
    setup_logging,
    strict_load_state_dict,
    to_dict,
    update_debug_running,
    verify_checkpoint_metric,
)
from utilities_torch import (
    ParameterClass,
    SlurmPreemptMonitor,
    apply_slurm_job_restore_flags,
    atomic_torch_save,
    checkpoint_path,
    create_loader,
    exit_for_slurm_requeue,
    make_joint_datasets,
    maybe_set_torch_threads,
    move_batch_to_device,
    resolve_torch_dtype,
    torch_load_checkpoint,
)

usage = """
Usage:
  python train_vi_only_torch.py run_torch_vi_only.cfg
  torchrun --standalone --nproc_per_node=1 train_vi_only_torch.py run_torch_vi_only.cfg
"""

VI_ONLY_ARCHITECTURE = 'coordinate_only_vi_only_frozen_pinn_v1'
TRAINING_STAGE = 'vi_only'


def resolve_vgp_lrs(pars):
    base = pars.train.lr
    return {
        'vgp_eta': base if getattr(pars.train, 'vgp_eta_lr', None) is None else pars.train.vgp_eta_lr,
        'vgp_lambda': (
            base if getattr(pars.train, 'vgp_lambda_lr', None) is None else pars.train.vgp_lambda_lr),
    }


def _vgp_hyper_params(vgp):
    params = [vgp.raw_amplitude]
    if vgp.anisotropic:
        if vgp.raw_length_scale_x.requires_grad:
            params.append(vgp.raw_length_scale_x)
        if vgp.raw_length_scale_y.requires_grad:
            params.append(vgp.raw_length_scale_y)
    elif vgp.raw_length_scale.requires_grad:
        params.append(vgp.raw_length_scale)
    if vgp.raw_noise_variance.requires_grad:
        params.append(vgp.raw_noise_variance)
    return params


def _vgp_variational_params(vgp, include_loc=True):
    params = [vgp.raw_variational_inducing_scale]
    if include_loc:
        params.append(vgp.variational_inducing_loc)
    return params


def build_vi_optimizers(raw_model, pars, vgp_lrs):
    """Adam / AdamW / AdamW+NGD over VGP params only (PINN excluded)."""
    opt_name = str(getattr(pars.train, 'vgp_optimizer', 'adamw') or 'adamw').strip().lower()
    use_ngd = opt_name in ('ngd', 'adamw_ngd', 'adam_ngd')
    base_opt = 'adamw' if 'adamw' in opt_name or opt_name == 'ngd' else 'adam'
    if opt_name in ('adam', 'adamw'):
        base_opt = opt_name
        use_ngd = False

    eta_lr = vgp_lrs['vgp_eta']
    lam_lr = vgp_lrs['vgp_lambda']
    ngd_lr = float(getattr(pars.train, 'vgp_ngd_lr', eta_lr) or eta_lr)

    if use_ngd:
        hyper_groups = [
            {'params': _vgp_hyper_params(raw_model.vgp_eta), 'lr': eta_lr},
            {'params': _vgp_hyper_params(raw_model.vgp_lambda), 'lr': lam_lr},
            {
                'params': _vgp_variational_params(raw_model.vgp_eta, include_loc=False)
                + _vgp_variational_params(raw_model.vgp_lambda, include_loc=False),
                'lr': eta_lr,
            },
            {
                'params': [raw_model.eta_log_shift],
                'lr': float(getattr(pars.train, 'eta_shift_lr', eta_lr)),
            },
            {
                'params': [raw_model.lambda_logit_shift],
                'lr': float(getattr(pars.train, 'lambda_shift_lr', lam_lr)),
            },
        ]
        hyper_groups = [g for g in hyper_groups if g['params']]
        optimizer = create_optimizer(base_opt, hyper_groups, learning_rate=eta_lr)
        ngd = VariationalNaturalGradient(
            [raw_model.vgp_eta, raw_model.vgp_lambda], learning_rate=ngd_lr)
        return optimizer, ngd, f'{base_opt}+ngd'

    groups = [
        {'params': list(raw_model.vgp_eta.parameters()), 'lr': eta_lr},
        {'params': list(raw_model.vgp_lambda.parameters()), 'lr': lam_lr},
        {
            'params': [raw_model.eta_log_shift],
            'lr': float(getattr(pars.train, 'eta_shift_lr', eta_lr)),
        },
        {
            'params': [raw_model.lambda_logit_shift],
            'lr': float(getattr(pars.train, 'lambda_shift_lr', lam_lr)),
        },
    ]
    optimizer = create_optimizer(base_opt, groups, learning_rate=eta_lr)
    return optimizer, None, base_opt


def build_vgp_scheduler(optimizer, pars, n_epochs_this_run):
    kind = str(getattr(pars.train, 'lr_scheduler', 'none') or 'none').strip().lower()
    if kind in ('', 'none', 'null', 'false'):
        return None, kind
    if kind == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(n_epochs_this_run), 1),
            eta_min=float(getattr(pars.train, 'lr_scheduler_eta_min', 0.0) or 0.0),
        ), kind
    if kind == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(getattr(pars.train, 'lr_scheduler_factor', 0.1) or 0.1),
            patience=int(getattr(pars.train, 'lr_scheduler_patience', 50) or 50),
        ), kind
    if kind in ('exp', 'exponential'):
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(getattr(pars.train, 'lr_scheduler_gamma', 0.995) or 0.995),
        ), kind
    raise ValueError(f"Unsupported train.lr_scheduler={kind!r}")


def pack_vi_only_checkpoint(raw_model, optimizer_vgp, epoch, physics_approximation, extra=None):
    payload = {
        'model': raw_model.state_dict(),
        'optimizer_vgp': optimizer_vgp.state_dict(),
        'optimizer': optimizer_vgp.state_dict(),
        'epoch': epoch,
        'architecture': VI_ONLY_ARCHITECTURE,
        'mean_net_outputs': MEAN_NET_OUTPUTS,
        'physics_approximation': physics_approximation,
        'coordinate_only': True,
        'mean_net_frozen': True,
        'mean_net_first_layer_in_features': first_state_layer(raw_model.mean_net).in_features,
        'optimizer_layout': 'vi_only_vgp_v1',
        'training_mode': 'vi_only_frozen_pinn',
        'kernel_type': raw_model.vgp_eta.kernel_type,
        'anisotropic': bool(raw_model.vgp_eta.anisotropic),
        'num_inducing': int(raw_model.vgp_eta.inducing_index_points.shape[0]),
    }
    if extra:
        payload.update(extra)
    return payload


def evaluate_eta_posterior(model, snapshot, device, torch_dtype, pars, max_points=8192):
    """η posterior mean / variance vs spin-up viscosity (+ spatial correlation)."""
    if snapshot.viscosity is None:
        return None
    raw = model.module if isinstance(model, DDP) else model
    geom = snapshot.geom_mask & np.isfinite(snapshot.viscosity) & (snapshot.viscosity > 0)
    ys, xs = np.where(geom)
    if ys.size == 0:
        return None
    if ys.size > max_points:
        rng = np.random.default_rng(0)
        pick = rng.choice(ys.size, size=max_points, replace=False)
        ys, xs = ys[pick], xs[pick]
    x = torch.as_tensor(snapshot.x[ys, xs], dtype=torch_dtype, device=device).reshape(-1, 1)
    y = torch.as_tensor(snapshot.y[ys, xs], dtype=torch_dtype, device=device).reshape(-1, 1)
    Xn = normalize_tensor(torch.cat([x, y], dim=1), raw.mean_net.iW_coord, raw.mean_net.b_coord)
    with torch.no_grad():
        theta, var_theta, _, _, _ = raw.vgp_eta.posterior_stats(Xn)
        theta = theta.detach().cpu().numpy().reshape(-1)
        std_theta = torch.sqrt(var_theta).detach().cpu().numpy().reshape(-1)
    eta_init = float(getattr(pars.prior, 'eta_init', 12.0))
    eta_log_center = math.log(max(eta_init, 1.0e-12))
    eta_log_shift = float(raw.eta_log_shift.detach().cpu().item())
    eta_log_min = math.log(float(pars.prior.eta_min))
    eta_log_max = math.log(float(pars.prior.eta_max))
    eta_log = np.clip(eta_log_center + eta_log_shift + theta, eta_log_min, eta_log_max)
    eta_pred = np.exp(eta_log)
    # delta-method approx for log10(η) std from latent θ std
    log10_std = (std_theta / math.log(10.0))
    eta_ref = snapshot.viscosity[ys, xs].astype(float)
    log_err = np.log10(eta_pred) - np.log10(eta_ref)
    abs_err = np.abs(log_err)
    # Rough calibration: fraction of truth within ±1 / ±2 predictive σ
    within_1 = float(np.mean(abs_err <= log10_std))
    within_2 = float(np.mean(abs_err <= 2.0 * log10_std))
    return {
        'log10_eta_rmse': float(np.sqrt(np.mean(log_err ** 2))),
        'log10_eta_bias': float(np.mean(log_err)),
        'log10_eta_r': float(np.corrcoef(np.log10(eta_pred), np.log10(eta_ref))[0, 1]),
        'rel_eta_rmse': float(np.sqrt(np.mean(((eta_pred - eta_ref) / eta_ref) ** 2))),
        'eta_pred_mean': float(np.mean(eta_pred)),
        'eta_ref_mean': float(np.mean(eta_ref)),
        'eta_post_var_mean': float(np.mean(std_theta ** 2)),
        'eta_post_std_mean': float(np.mean(std_theta)),
        'eta_post_std_p90': float(np.percentile(std_theta, 90)),
        'calibration_within_1sigma': within_1,
        'calibration_within_2sigma': within_2,
        'n_points': int(eta_pred.size),
    }


class EarlyStopper:
    def __init__(self, metric, patience, mode='max', min_delta=0.0):
        self.metric = metric
        self.patience = int(patience)
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best = None
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, value):
        if value is None or not math.isfinite(float(value)):
            return False
        value = float(value)
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False
        improved = (
            value > self.best + self.min_delta if self.mode == 'max'
            else value < self.best - self.min_delta)
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
            return True
        return False


def main(pars):
    apply_slurm_job_restore_flags(pars)
    preempt = SlurmPreemptMonitor()
    preempt.install()
    physics_approximation = resolve_physics_approximation(pars)
    maybe_set_torch_threads(pars)
    torch_dtype = resolve_torch_dtype(pars.runtime.dtype)
    distributed, rank, local_rank, world_size = init_distributed(pars)
    device = choose_device(pars, local_rank)
    setup_logging(pars.train.logfile, rank, append=bool(getattr(pars.train, 'restore', False)))
    if rank == 0:
        logging.info('VI-only training (frozen PINN):\n%s', __import__('pprint').pformat(to_dict(pars), indent=2))
        logging.info('physics=%s', physics_approximation)

    obs_train, obs_test, phys_train, phys_test, norms, snapshot = make_joint_datasets(pars, torch_dtype)
    obs_train_loader = create_loader(
        obs_train, pars.train.batch_size, True, pars.torch.train_drop_last, pars, distributed)
    phys_train_loader = create_loader(
        phys_train, pars.train.physics_batch_size, True, pars.torch.train_drop_last, pars, distributed)
    obs_test_loader = create_loader(obs_test, pars.train.batch_size, False, False, pars, distributed)
    phys_test_loader = create_loader(phys_test, pars.train.physics_batch_size, False, False, pars, distributed)

    mean_net = MeanNetwork(norms, resnet=pars.pretrain.resnet, dtype=torch_dtype)
    x_ref = snapshot.x[snapshot.geom_mask]
    y_ref = snapshot.y[snapshot.geom_mask]
    vgp_eta = make_sparse_vgp(x_ref, y_ref, norms, pars, 'eta', torch_dtype)
    vgp_lambda = make_sparse_vgp(x_ref, y_ref, norms, pars, 'lambda', torch_dtype)
    model = JointModel(mean_net, vgp_eta, vgp_lambda, dtype=torch_dtype).to(device)
    assert_coordinate_only_mean_net(model.mean_net)

    if not pars.train.restore:
        model.vgp_eta.initialize_variational_to_prior()
        model.vgp_lambda.initialize_variational_to_prior()

    pretrain_ckpt = checkpoint_path(pars.train.meannet_checkdir, pars.train.meannet_checkname)
    if not pars.train.restore:
        if not Path(pretrain_ckpt).exists():
            message = f'mean_net pretrain checkpoint not found: {pretrain_ckpt}'
            if bool(getattr(pars.train, 'require_pretrain_checkpoint', True)):
                raise FileNotFoundError(message)
            if rank == 0:
                logging.warning('%s; continuing without pretrain load', message)
        else:
            state = torch_load_checkpoint(pretrain_ckpt, map_location=device)
            require_checkpoint_architecture(state, pretrain_ckpt, MEAN_NET_ARCHITECTURE, 'mean_net pretrain')
            strict_load_state_dict(
                model.mean_net, checkpoint_model_state(state, pretrain_ckpt), pretrain_ckpt, 'mean_net pretrain')
            assert_coordinate_only_mean_net(model.mean_net)
            if bool(getattr(pars.train, 'verify_pretrain_load', True)):
                pretrain_verify_stats = evaluate_mean_net_observation(
                    model.mean_net, obs_test_loader, device, torch_dtype, world_size)
                verify_checkpoint_metric(
                    pretrain_verify_stats['loss'],
                    state.get('test_loss') if isinstance(state, dict) else None,
                    pretrain_ckpt, 'Pretrain checkpoint test-loss',
                    getattr(pars.train, 'verify_pretrain_loss_rtol', 5.0e-2),
                    getattr(pars.train, 'verify_pretrain_loss_atol', 1.0e-8))
                if rank == 0:
                    logging.info('loaded frozen mean_net from pretrain: %s', pretrain_ckpt)
                    log_mean_net_obs_stats('pretrain-load verification on obs_test', pretrain_verify_stats)

    # Permanently freeze PINN before wrapping DDP.
    set_module_requires_grad(model.mean_net, False)
    model.mean_net.eval()
    set_vgp_trainable(model, True)

    if distributed:
        ddp_kwargs = {'device_ids': [local_rank], 'output_device': local_rank} if device.type == 'cuda' else {}
        ddp_kwargs['find_unused_parameters'] = True
        model = DDP(model, **ddp_kwargs)

    raw_model = model.module if isinstance(model, DDP) else model
    set_module_requires_grad(raw_model.mean_net, False)
    raw_model.mean_net.eval()

    vgp_lrs = resolve_vgp_lrs(pars)
    optimizer_vgp, ngd_opt, opt_kind = build_vi_optimizers(raw_model, pars, vgp_lrs)
    _, vgp_grad_clip = resolve_grad_clips(pars)

    ckpt_file_old = checkpoint_path(pars.train.checkdir, pars.train.checkname_old)
    ckpt_file_new = checkpoint_path(pars.train.checkdir, pars.train.checkname_new)
    Path(pars.train.checkdir).mkdir(parents=True, exist_ok=True)

    n_additional_epochs = int(pars.train.n_epochs)
    start_epoch = 0
    stop_epoch = start_epoch + n_additional_epochs
    if pars.train.restore:
        if not Path(ckpt_file_old).exists():
            raise FileNotFoundError(f'train.restore=True but checkpoint missing: {ckpt_file_old}')
        state = torch_load_checkpoint(ckpt_file_old, map_location=device)
        require_checkpoint_architecture(state, ckpt_file_old, VI_ONLY_ARCHITECTURE, 'vi-only model')
        target = model.module if isinstance(model, DDP) else model
        joint_state = {
            k: v for k, v in checkpoint_model_state(state, ckpt_file_old).items()
            if not k.startswith('mean_net_ref.')
        }
        strict_load_state_dict(target, joint_state, ckpt_file_old, 'vi-only model')
        set_module_requires_grad(raw_model.mean_net, False)
        raw_model.mean_net.eval()
        if getattr(pars.train, 'restore_optimizer', False) and 'optimizer_vgp' in state:
            try:
                optimizer_vgp.load_state_dict(state['optimizer_vgp'])
            except (ValueError, RuntimeError) as exc:
                if rank == 0:
                    logging.warning('Skipping optimizer_vgp restore: %s', exc)
        completed = int(state.get('epoch', -1))
        start_epoch = completed + 1
        stop_epoch = start_epoch + n_additional_epochs
        if rank == 0:
            logging.info('restored vi-only checkpoint %s (resume epoch %d)', ckpt_file_old, start_epoch)

    grid_np, weights_np = np.polynomial.hermite.hermgauss(pars.train.quadrature_size)
    grid = torch.tensor(grid_np, device=device, dtype=torch_dtype)
    weights = torch.tensor(weights_np / np.sqrt(np.pi), device=device, dtype=torch_dtype)

    scheduler, scheduler_kind = build_vgp_scheduler(optimizer_vgp, pars, n_additional_epochs)
    last_test = [float('nan')] * 6
    last_eta_metrics = None
    steps_this_epoch = max(len(obs_train_loader), len(phys_train_loader))
    if getattr(pars.train, 'max_steps_per_epoch', None) is not None:
        steps_this_epoch = min(steps_this_epoch, pars.train.max_steps_per_epoch)

    early_metric = str(getattr(pars.train, 'early_stop_metric', '') or '').strip().lower()
    early_patience = int(getattr(pars.train, 'early_stop_patience', 0) or 0)
    early_stopper = None
    if early_metric and early_patience > 0:
        mode = 'min' if early_metric in ('rmse', 'log10_eta_rmse', 'elbo', 'test_total', 'val_elbo') else 'max'
        early_stopper = EarlyStopper(early_metric, early_patience, mode=mode)
        if rank == 0:
            logging.info('early stopping on %s (patience=%d, mode=%s)', early_metric, early_patience, mode)

    if rank == 0:
        kd = raw_model.vgp_eta.kernel_diagnostics()
        logging.info(
            'vi-only: data=%.4g phys=%.4g state_reg=%.4g eta_prior=%.4g (std=%.4g) | '
            'kl_eta=%.4g | optimizer=%s scheduler=%s | '
            'kernel=%s anisotropic=%s inducing=%d ls=%.4g amp=%.4g | '
            'vgp_eta_lr=%.3e',
            float(getattr(pars.train, 'data_scale', 1.0) or 1.0),
            float(getattr(pars.train, 'phys_scale', 1.0) or 1.0),
            float(getattr(pars.train, 'state_reg_scale', 0.0) or 0.0),
            float(getattr(pars.train, 'eta_prior_scale', 0.0) or 0.0),
            float(getattr(pars.train, 'eta_prior_std', 1.0) or 1.0),
            float(pars.prior.kl_eta),
            opt_kind,
            scheduler_kind,
            kd['kernel_type'],
            bool(kd['anisotropic']),
            int(kd['num_inducing']),
            kd['length_scale'],
            kd['amplitude'],
            vgp_lrs['vgp_eta'],
        )

    metrics_csv = resolve_metrics_csv(pars.train, pars.train.logfile, TRAINING_STAGE)
    plot_dir = resolve_plot_dir(pars.train, pars.train.logfile, TRAINING_STAGE)
    plot_every = resolve_plot_every(pars.train)
    metrics_writer = None
    if rank == 0:
        metrics_writer = EpochMetricsWriter(
            metrics_csv, VI_ONLY_FIELDS, append=bool(getattr(pars.train, 'restore', False)))

    prev_lrs = [g['lr'] for g in optimizer_vgp.param_groups]

    for epoch in range(start_epoch, stop_epoch):
        # Keep PINN in eval + frozen; only VGP in train mode for dropout-free BN consistency.
        model.train()
        raw_model.mean_net.eval()
        set_module_requires_grad(raw_model.mean_net, False)

        for loader in (obs_train_loader, phys_train_loader):
            if distributed and isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(epoch)

        obs_iter = iter(obs_train_loader)
        phys_iter = iter(phys_train_loader)
        running = torch.zeros(5, dtype=torch.float64, device=device)
        running_debug = init_debug_running()
        for _ in range(steps_this_epoch):
            batch_obs, obs_iter = next_or_restart(obs_iter, obs_train_loader)
            batch_phys, phys_iter = next_or_restart(phys_iter, phys_train_loader)
            batch_obs = move_batch_to_device(batch_obs, device, torch_dtype)
            batch_phys = move_batch_to_device(batch_phys, device, torch_dtype)

            optimizer_vgp.zero_grad(set_to_none=True)
            if ngd_opt is not None:
                ngd_opt.zero_grad()

            # PINN weights have requires_grad=False; forward still differentiates
            # w.r.t. coordinates for PDE residuals and w.r.t. η (VGP).
            losses = joint_loss(
                model, batch_obs, batch_phys, grid, weights, pars, torch_dtype, world_size,
                return_debug=True)
            total_loss = losses[0] + losses[1] + losses[2] + losses[3]
            total_loss.backward()

            # Drop any accidental mean_net grads (should already be None).
            for p in raw_model.mean_net.parameters():
                if p.grad is not None:
                    p.grad = None

            grad_norms = collect_grad_norms(raw_model)
            clipped = clip_module_grads(raw_model.vgp_eta, vgp_grad_clip)
            clip_module_grads(raw_model.vgp_lambda, vgp_grad_clip)
            clip_param_grads(
                [raw_model.eta_log_shift, raw_model.lambda_logit_shift], vgp_grad_clip)
            record_clip_norm(running_debug, 'vgp', clipped)
            update_debug_running(running_debug, losses[4], grad_norms)

            if ngd_opt is not None:
                ngd_opt.step()
            optimizer_vgp.step()
            running_debug['vgp_updates'] += 1
            running[0] += losses[0].detach().to(torch.float64)
            running[1] += losses[1].detach().to(torch.float64)
            running[2] += losses[2].detach().to(torch.float64)
            running[3] += losses[3].detach().to(torch.float64)
            running[4] += 1.0

            if preempt.triggered() and rank == 0:
                atomic_torch_save(
                    pack_vi_only_checkpoint(
                        raw_model, optimizer_vgp, epoch, physics_approximation,
                        extra={'preempt_checkpoint': True}),
                    ckpt_file_old,
                )
                print('[slurm] Preemption checkpoint saved. Exiting for requeue.', flush=True)
                exit_for_slurm_requeue()

        train_stats = reduce_mean(running[:4] / torch.clamp(running[4], min=1.0), world_size).tolist()
        if epoch % pars.train.test_every == 0:
            last_test = evaluate(
                model, obs_test_loader, phys_test_loader, device, torch_dtype,
                grid, weights, pars, world_size)

        eta_eval_every = int(getattr(pars.train, 'eta_eval_every', 0) or 0)
        if eta_eval_every > 0 and epoch % eta_eval_every == 0 and rank == 0:
            last_eta_metrics = evaluate_eta_posterior(model, snapshot, device, torch_dtype, pars)

        train_total = sum(train_stats[:4])
        test_total = sum(last_test[:4])
        if scheduler is not None:
            if scheduler_kind == 'plateau':
                # Prefer validation ELBO (test_total) when available.
                plateau_val = test_total if math.isfinite(test_total) else train_total
                scheduler.step(plateau_val)
            else:
                scheduler.step()

        if rank == 0:
            cur_lrs = [g['lr'] for g in optimizer_vgp.param_groups]
            for i, (old, new) in enumerate(zip(prev_lrs, cur_lrs)):
                if abs(old - new) > 1.0e-18 * max(1.0, abs(old)):
                    logging.info('lr_change epoch=%d group=%d %.6e -> %.6e', epoch, i, old, new)
            prev_lrs = list(cur_lrs)

            logging.info(
                '%d %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f',
                epoch, *train_stats[:4], train_total, *last_test[:4], test_total)
            logging.info('%s', format_debug_log(epoch, running_debug))
            kd = raw_model.vgp_eta.kernel_diagnostics()
            logging.info(
                'kernel epoch=%d type=%s amp=%.6e ls=%.6e ls_x=%.6e ls_y=%.6e inducing=%d',
                epoch, kd['kernel_type'], kd['amplitude'], kd['length_scale'],
                kd['length_scale_x'], kd['length_scale_y'], int(kd['num_inducing']))
            lr_eta = optimizer_vgp.param_groups[0]['lr']
            logging.info(
                'lrs epoch=%d vgp=%.6e (%s) updates_vgp=%d mean_net=frozen',
                epoch, lr_eta, opt_kind, int(running_debug.get('vgp_updates', 0)))
            if last_eta_metrics is not None:
                logging.info(
                    'eta_vs_ref epoch=%d log10_rmse=%.6f log10_bias=%.6f log10_r=%.6f '
                    'rel_rmse=%.6f eta_pred_mean=%.6g post_std_mean=%.6g '
                    'cal_1s=%.3f cal_2s=%.3f n=%d',
                    epoch,
                    last_eta_metrics['log10_eta_rmse'],
                    last_eta_metrics['log10_eta_bias'],
                    last_eta_metrics['log10_eta_r'],
                    last_eta_metrics['rel_eta_rmse'],
                    last_eta_metrics['eta_pred_mean'],
                    last_eta_metrics['eta_post_std_mean'],
                    last_eta_metrics['calibration_within_1sigma'],
                    last_eta_metrics['calibration_within_2sigma'],
                    last_eta_metrics['n_points'],
                )

            debug_summary = summarize_debug_running(running_debug)
            if metrics_writer is not None:
                row = {
                    'epoch': epoch,
                    'train_data': train_stats[0],
                    'train_phys': train_stats[1],
                    'train_kl': train_stats[2],
                    'train_state_reg': train_stats[3],
                    'train_total': train_total,
                    'train_pde': debug_summary.get('train_pde', float('nan')),
                    'train_bc': debug_summary.get('train_bc', float('nan')),
                    'test_data': last_test[0],
                    'test_phys': last_test[1],
                    'test_kl': last_test[2],
                    'test_state_reg': last_test[3],
                    'test_pde': last_test[4],
                    'test_bc': last_test[5],
                    'test_total': test_total,
                    'lr_vgp_eta': lr_eta,
                    'vgp_updates': int(running_debug.get('vgp_updates', 0)),
                    'mean_net_frozen': 1,
                    'kernel_amplitude': kd['amplitude'],
                    'kernel_length_scale': kd['length_scale'],
                    'kernel_length_scale_x': kd['length_scale_x'],
                    'kernel_length_scale_y': kd['length_scale_y'],
                    'num_inducing': kd['num_inducing'],
                    **{k: v for k, v in debug_summary.items() if k not in ('train_pde', 'train_bc')},
                }
                if last_eta_metrics is not None:
                    row.update(last_eta_metrics)
                metrics_writer.write_row(row)

            payload = pack_vi_only_checkpoint(raw_model, optimizer_vgp, epoch, physics_approximation)
            atomic_torch_save(payload, ckpt_file_new)
            atomic_torch_save(payload, ckpt_file_old)
            if metrics_writer is not None:
                maybe_plot_training(
                    TRAINING_STAGE, metrics_csv, plot_dir, epoch, plot_every, pars.train.logfile)

            if early_stopper is not None:
                if early_metric in ('log10_eta_r', 'r', 'corr'):
                    score = None if last_eta_metrics is None else last_eta_metrics['log10_eta_r']
                elif early_metric in ('rmse', 'log10_eta_rmse'):
                    score = None if last_eta_metrics is None else last_eta_metrics['log10_eta_rmse']
                elif early_metric in ('elbo', 'val_elbo', 'test_total'):
                    score = test_total
                else:
                    score = None if last_eta_metrics is None else last_eta_metrics.get(early_metric)
                if early_stopper.step(score):
                    logging.info(
                        'early stopping at epoch %d (metric=%s best=%.6g)',
                        epoch, early_metric, early_stopper.best)
                    break

    if rank == 0 and metrics_writer is not None:
        maybe_plot_training(
            TRAINING_STAGE, metrics_csv, plot_dir, max(stop_epoch - 1, 0),
            max(plot_every, 1), pars.train.logfile)
        metrics_writer.close()

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    argv = sys.argv[1:]
    if len(argv) != 1:
        print(usage)
        sys.exit(1)
    main(ParameterClass(argv[0]))
