#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from pathlib import Path
import logging
import math
import os
import random
import sys
import pprint

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from models_torch import MeanNetwork, create_optimizer
from utilities_torch import (
    configure_slurm_ddp_env,
    ParameterClass,
    checkpoint_path,
    create_loader,
    make_pretrain_datasets,
    maybe_set_torch_threads,
    move_batch_to_device,
    resolve_torch_dtype,
    torch_load_checkpoint,
)

usage = """
Usage:
  python pretrain_solution_torch.py run_torch.cfg
  torchrun --standalone --nproc_per_node=4 pretrain_solution_torch.py run_torch.cfg
  srun python pretrain_solution_torch.py run_torch.cfg
"""

MEAN_NET_ARCHITECTURE = 'coordinate_only_mean_net_predict_s_h_v1'
MEAN_NET_OUTPUTS = ('u', 'v', 's', 'h')

def to_dict(obj):
    if hasattr(obj, "__dict__"):
        return {k: to_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    else:
        return obj

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


def setup_logging(logfile, rank, append=False):
    if rank == 0:
        filemode = 'a' if append else 'w'
        logging.basicConfig(filename=logfile, filemode=filemode, level=logging.INFO)


def reduce_mean(value, world_size):
    if world_size == 1:
        return value
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value / world_size


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
            'Regenerate the pretrain checkpoint with the coordinate-only code.'
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


def require_mean_net_architecture(checkpoint, checkpoint_file):
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'Checkpoint {checkpoint_file} has no metadata. The current MeanNetwork expects '
            'outputs (u, v, s, h), with bed computed as b=s-h; do not restore an old '
            'state-dict-only checkpoint whose fourth output may be bed.'
        )
    arch = checkpoint.get('architecture')
    outputs = tuple(checkpoint.get('mean_net_outputs', ()))
    if arch != MEAN_NET_ARCHITECTURE or outputs != MEAN_NET_OUTPUTS:
        raise RuntimeError(
            f'Checkpoint {checkpoint_file} was saved with architecture={arch!r}, '
            f'mean_net_outputs={outputs!r}; expected architecture={MEAN_NET_ARCHITECTURE!r}, '
            f'mean_net_outputs={MEAN_NET_OUTPUTS!r}. Redo pretraining with the s+h '
            'MeanNetwork before using this checkpoint.'
        )


def strict_load_state_dict(module, state_dict, checkpoint_file, component_name):
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
        if missing:
            detail_lines.append('missing keys in checkpoint: ' + ', '.join(missing[:20]))
        if unexpected:
            detail_lines.append('unexpected keys in checkpoint: ' + ', '.join(unexpected[:20]))

        hint = ''
        first_weight = state_dict.get('state_dense.layers.0.weight')
        if first_weight is not None and len(first_weight.shape) == 2 and first_weight.shape[1] != 2:
            hint = (
                '\nThis looks like an old observation-conditioned MeanNetwork checkpoint '
                f'with {first_weight.shape[1]} input features. The coordinate-only model '
                'requires exactly 2 input features: normalized x and y.'
            )
        raise RuntimeError(
            f'Cannot strictly load {component_name} checkpoint {checkpoint_file}.\n'
            + '\n'.join(detail_lines)
            + hint
        )
    module.load_state_dict(state_dict, strict=True)


def verify_checkpoint_loss(current_loss, recorded_loss, checkpoint_file, label, rtol=5.0e-2, atol=1.0e-8):
    if recorded_loss is None:
        return
    try:
        recorded_loss = float(recorded_loss)
    except (TypeError, ValueError):
        return
    if not (math.isfinite(current_loss) and math.isfinite(recorded_loss)):
        return
    allowed = max(float(atol), float(rtol) * max(abs(recorded_loss), 1.0))
    diff = abs(float(current_loss) - recorded_loss)
    if diff > allowed:
        raise RuntimeError(
            f'{label} verification failed for {checkpoint_file}: '
            f'evaluated test loss={current_loss:.10g}, checkpoint test_loss={recorded_loss:.10g}, '
            f'abs diff={diff:.3g}, allowed={allowed:.3g}. '
            'This usually means the checkpoint, data file, normalization, or architecture does not match.'
        )


def pretrain_loss_components(model, batch):
    # Exact coordinate-only pretraining objective:
    #   state_net(x, y) -> u, v, s, H
    # Observed u/v/s/H are targets in the loss only; they are not inputs.
    # Bed is derived from physical predictions as b = s - H.
    up, vp, sp, hp = model(batch['x'], batch['y'], inverse_norm=False)
    # hp.clamp_min(80.0) # hard clamp h min to 80 m
    
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


def evaluate(model, loader, device, torch_dtype, world_size):
    model.eval()
    total = torch.zeros(2, device=device, dtype=torch.float64)
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device, torch_dtype)
            loss, _ = pretrain_loss_components(model, batch)
            total[0] += loss.to(torch.float64)
            total[1] += 1.0
    if world_size > 1:
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if total[1].item() == 0:
        return float('nan')
    return (total[0] / total[1]).item()


def checkpoint_rng_state():
    """Capture RNG states needed to continue deterministically from an epoch checkpoint."""
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    else:
        state['cuda'] = None
    return state


def restore_rng_state(checkpoint, device, rank):
    """Restore RNG state from a checkpoint. Returns True if all available states were restored."""
    if not isinstance(checkpoint, dict) or checkpoint.get('rng_state') is None:
        if rank == 0:
            logging.warning('checkpoint has no rng_state; model/optimizer restore is valid, but exact stochastic replay is not guaranteed')
        return False

    rng_state = checkpoint['rng_state']
    random.setstate(rng_state['python'])
    np.random.set_state(rng_state['numpy'])
    torch.set_rng_state(rng_state['torch'].detach().cpu())

    cuda_states = rng_state.get('cuda')
    if device.type == 'cuda' and cuda_states is not None:
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in cuda_states])
    return True


def atomic_torch_save(obj, filename):
    """Write checkpoints atomically so a preemption during save does not corrupt the previous checkpoint."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_name(filename.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, filename)


def resolve_pretrain_restore_file(pars, ckpt_file, last_file, best_file):
    """Resolve the checkpoint used for pretraining restart."""
    restore_file = getattr(pars.pretrain, 'restore_file', None)
    if restore_file is not None:
        return str(restore_file)

    restore_checkname = getattr(pars.pretrain, 'restore_checkname', None)
    if restore_checkname is not None:
        return checkpoint_path(pars.pretrain.checkdir, restore_checkname)

    restore_from = str(getattr(pars.pretrain, 'restore_from', 'last')).strip().lower()
    if restore_from == 'last':
        return last_file
    if restore_from == 'best':
        return best_file
    if restore_from in ('checkname', 'base', 'model'):
        return ckpt_file
    raise ValueError(
        f"Unsupported pretrain.restore_from={restore_from!r}. "
        "Use 'last', 'best', or 'checkname', or set pretrain.restore_file/pretrain.restore_checkname."
    )


def load_scheduler_state_or_reconstruct(scheduler, checkpoint, checkpoint_file, completed_epoch, rank):
    """Restore scheduler state, with a deterministic fallback for old pretrain checkpoints."""
    if isinstance(checkpoint, dict) and checkpoint.get('scheduler') is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])
        return True

    # Older checkpoints saved optimizer/model/epoch but not scheduler state.  They were saved
    # immediately before scheduler.step() at the end of the epoch, so advance the newly-created
    # scheduler by one end-of-epoch step from the saved optimizer LR.
    if completed_epoch is not None:
        scheduler.last_epoch = int(completed_epoch)
        scheduler._step_count = max(int(completed_epoch) + 1, 1)
        scheduler._last_lr = [group['lr'] for group in scheduler.optimizer.param_groups]
        scheduler.step()
        if rank == 0:
            logging.warning(
                'checkpoint %s has no scheduler state; reconstructed MultiStepLR state from completed epoch %d',
                checkpoint_file,
                int(completed_epoch),
            )
        return False

    raise KeyError(f'Checkpoint {checkpoint_file} has no scheduler state and no epoch for reconstruction.')


def read_checkpoint_float(checkpoint, key, default=float('nan')):
    if not isinstance(checkpoint, dict) or checkpoint.get(key) is None:
        return default
    try:
        return float(checkpoint[key])
    except (TypeError, ValueError):
        return default


def main(pars):
    maybe_set_torch_threads(pars)
    torch_dtype = resolve_torch_dtype(pars.runtime.dtype)
    distributed, rank, local_rank, world_size = init_distributed(pars)
    device = choose_device(pars, local_rank)
    setup_logging(pars.pretrain.logfile, rank, append=bool(getattr(pars.pretrain, 'restore', False)))

    if rank == 0:
        logging.info(pprint.pformat(to_dict(pars), indent=2))
        
    train_dataset, test_dataset, norms, _ = make_pretrain_datasets(pars, torch_dtype)
    train_loader = create_loader(
        train_dataset,
        batch_size=pars.pretrain.batch_size,
        shuffle=True,
        drop_last=pars.torch.train_drop_last,
        pars=pars,
        distributed=distributed)
    test_loader = create_loader(
        test_dataset,
        batch_size=pars.pretrain.batch_size,
        shuffle=False,
        drop_last=False,
        pars=pars,
        distributed=distributed)

    model = MeanNetwork(norms, resnet=pars.pretrain.resnet, dtype=torch_dtype).to(device)
    assert_coordinate_only_mean_net(model)
    if rank == 0:
        logging.info(
            'MeanNetwork architecture: coordinate-only state_net(x, y) -> u, v, s, H; bed is derived as b=s-H; first layer in_features=%d',
            first_state_layer(model).in_features,
        )
    if distributed:
        ddp_kwargs = {'device_ids': [local_rank], 'output_device': local_rank} if device.type == 'cuda' else {}
        model = DDP(model, **ddp_kwargs)

    # optimizer and LR schedule
    optimizer = create_optimizer(pars.pretrain.optimizer, model.parameters(), learning_rate=pars.pretrain.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,milestones=[10000, 20000, 40000, 80000, 160000, 320000], gamma=0.5)

    ckpt_file = checkpoint_path(pars.pretrain.checkdir, pars.pretrain.checkname)
    best_file = checkpoint_path(pars.pretrain.checkdir, pars.pretrain.checkname + '_best')
    last_file = checkpoint_path(pars.pretrain.checkdir, pars.pretrain.checkname + '_last')
    Path(pars.pretrain.checkdir).mkdir(parents=True, exist_ok=True)
    print(f'rank {rank} local_rank {local_rank} world_size {world_size} '
          f'has {len(train_dataset)} local samples across {len(train_loader)} local batches' )

    last_test = float('nan')
    best_test = float('inf')
    start_epoch = 0
    n_additional_epochs = int(pars.pretrain.n_epochs)
    if n_additional_epochs < 0:
        raise ValueError(f'pretrain.n_epochs must be non-negative, got {pars.pretrain.n_epochs}')
    stop_epoch = n_additional_epochs
    restored_state = None
    restore_file = None

    # Restore only checkpoints that exactly match the coordinate-only architecture.
    # On restart, pretrain.n_epochs means "additional epochs to run after the restored checkpoint".
    if pars.pretrain.restore:
        restore_file = resolve_pretrain_restore_file(pars, ckpt_file, last_file, best_file)
        if not Path(restore_file).exists():
            raise FileNotFoundError(f'pretrain.restore=True but checkpoint does not exist: {restore_file}')
        restored_state = torch_load_checkpoint(restore_file, map_location=device)
        require_mean_net_architecture(restored_state, restore_file)
        target = model.module if isinstance(model, DDP) else model
        strict_load_state_dict(target, checkpoint_model_state(restored_state, restore_file), restore_file, 'pretrain MeanNetwork')
        if not isinstance(restored_state, dict) or restored_state.get('optimizer') is None:
            raise KeyError(f'Checkpoint {restore_file} has no optimizer state; exact pretrain restart requires it.')
        optimizer.load_state_dict(restored_state['optimizer'])

        completed_epoch = restored_state.get('epoch') if isinstance(restored_state, dict) else None
        if completed_epoch is None:
            raise KeyError(f'Checkpoint {restore_file} has no epoch; cannot determine the restart epoch.')
        completed_epoch = int(completed_epoch)
        start_epoch = completed_epoch + 1
        stop_epoch = start_epoch + n_additional_epochs
        last_test = read_checkpoint_float(restored_state, 'test_loss', default=float('nan'))
        best_test = read_checkpoint_float(restored_state, 'best_test_loss', default=float('inf'))
        if not math.isfinite(best_test) and Path(best_file).exists():
            try:
                best_state = torch_load_checkpoint(best_file, map_location='cpu')
                best_test = read_checkpoint_float(best_state, 'test_loss', default=float('inf'))
            except Exception as exc:
                if rank == 0:
                    logging.warning('Could not read previous best checkpoint %s: %s', best_file, exc)
        if not math.isfinite(best_test):
            best_test = last_test if math.isfinite(last_test) else float('inf')

        load_scheduler_state_or_reconstruct(scheduler, restored_state, restore_file, completed_epoch, rank)

        restored_test = evaluate(model, test_loader, device, torch_dtype, world_size)
        verify_checkpoint_loss(
            restored_test,
            restored_state.get('test_loss') if isinstance(restored_state, dict) else None,
            restore_file,
            'Pretrain restore',
        )
        # Restore RNG after verification as well, so DataLoader worker seeding or any
        # future stochastic validation code cannot perturb the next training epoch.
        rng_restored = restore_rng_state(restored_state, device, rank)
        if rank == 0:
            logging.info(
                'strictly restored pretrain checkpoint %s at completed epoch %d; next epoch=%d; '
                'running %d additional epochs through epoch %d; verified test loss %.10f%s; rng_restored=%s',
                restore_file,
                completed_epoch,
                start_epoch,
                n_additional_epochs,
                stop_epoch - 1,
                restored_test,
                '' if not isinstance(restored_state, dict) or restored_state.get('test_loss') is None else f' (checkpoint test_loss={float(restored_state["test_loss"]):.10f})',
                rng_restored,
            )

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        if distributed and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        # log different loss components
        running = torch.zeros(10, device=device, dtype=torch.float64)
        for step, batch in enumerate(train_loader):
            if pars.pretrain.max_steps_per_epoch is not None and step >= pars.pretrain.max_steps_per_epoch:
                break
            batch = move_batch_to_device(batch, device, torch_dtype)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = pretrain_loss_components(model, batch)
            loss.backward()
            grad_norm = float('nan')
            if pars.pretrain.grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), pars.pretrain.grad_clip)
            optimizer.step()
            # log different loss components
            running[0] += loss.detach().to(torch.float64)
            running[1] += 1.0
            running[2:10] += parts.to(torch.float64)

        # log different loss components
        train_loss = (running[0] / running[1].clamp_min(1.0)).item()
        u_loss = (running[2] / running[6].clamp_min(1.0)).item()
        v_loss = (running[3] / running[7].clamp_min(1.0)).item()
        s_loss = (running[4] / running[8].clamp_min(1.0)).item()
        h_loss = (running[5] / running[9].clamp_min(1.0)).item()
        evaluated = False
        if epoch % pars.pretrain.test_every == 0:
            last_test = evaluate(model, test_loader, device, torch_dtype, world_size)
            evaluated = True

        # End-of-epoch scheduler state is part of the restart state.  Step it before saving
        # so model_last.pt resumes at the next epoch exactly as an uninterrupted run would.
        scheduler.step()
        if evaluated and last_test < best_test:
            best_test = last_test

        raw_model = model.module if isinstance(model, DDP) else model
        state = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'rng_state': checkpoint_rng_state(),
            'epoch': epoch,
            'next_epoch': epoch + 1,
            'train_loss': train_loss,
            'test_loss': last_test,
            'best_test_loss': best_test,
            'pretrain_n_epochs_this_run': n_additional_epochs,
            'run_start_epoch': start_epoch,
            'run_stop_epoch_exclusive': stop_epoch,
            'architecture': MEAN_NET_ARCHITECTURE,
            'mean_net_outputs': MEAN_NET_OUTPUTS,
            'coordinate_only': True,
            'mean_net_first_layer_in_features': first_state_layer(raw_model).in_features,
        }
        if rank == 0:
            logging.info(
                '%d %.10f %.10f %.6e %.6e %.6e %.6e %.2f',
                epoch, train_loss, last_test, u_loss, v_loss, s_loss, h_loss, grad_norm)
            atomic_torch_save(state, last_file)
            if evaluated and last_test <= best_test:
                atomic_torch_save(state, best_file)

        
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        print(usage)
        sys.exit()
    main(ParameterClass(args[0]))
