#!/usr/bin/env python3
"""Epoch metrics CSV I/O, log parsing, and training diagnostic plots."""

from __future__ import annotations

import csv
import configparser
import logging
import os
import re
from pathlib import Path

import matplotlib

if os.environ.get('MPLBACKEND') is None:
    try:
        get_ipython  # noqa: B018
    except NameError:
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PRETRAIN_FIELDS = (
    'epoch',
    'train_loss',
    'test_loss',
    'u_loss',
    'v_loss',
    's_loss',
    'h_loss',
    'grad_norm',
)

JOINT_FIELDS = (
    'epoch',
    'train_data',
    'train_phys',
    'train_kl',
    'train_state_reg',
    'train_total',
    'train_pde',
    'train_bc',
    'train_eta_prior',
    'train_kl_only',
    'test_data',
    'test_phys',
    'test_kl',
    'test_state_reg',
    'test_total',
    'test_pde',
    'test_bc',
    'grad_mean_net',
    'grad_vgp_eta',
    'grad_vgp_lambda',
    'grad_ratio',
    'vgp_updates',
    'mean_updates',
    'lr_mean_net',
    'lr_vgp_eta',
    'lr_vgp_lambda',
    'mean_net_opt_kind',
    'log10_eta_rmse',
    'log10_eta_bias',
    'log10_eta_r',
    'rel_eta_rmse',
    'eta_pred_mean',
    'eta_ref_mean',
    'mean_net_frozen',
    'eta_min',
    'eta_max',
    'lambda_min',
    'lambda_max',
    'rux_min',
    'rux_max',
    'rvy_min',
    'rvy_max',
    'rh_min',
    'rh_max',
)

DEBUG_FIELD_RE = re.compile(r'(\w+)=\[([\d.eE+-]+),([\d.eE+-]+)\]')

_LOG_FLOAT = r'(?:[\d.eE+-]+|nan|inf|-inf)'

PRETRAIN_LOG_RE = re.compile(
    rf'^INFO:root:(\d+)\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s*$',
    re.IGNORECASE,
)
JOINT_LOG_RE = re.compile(
    rf'^INFO:root:(\d+)\s+'
    rf'({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+'
    rf'({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s*$',
    re.IGNORECASE,
)
DEBUG_GRAD_RE = re.compile(
    rf'debug\s+(\d+)\s+grad_norms\s+'
    rf'mean_net=({_LOG_FLOAT})\s+vgp_eta=({_LOG_FLOAT})\s+vgp_lambda=({_LOG_FLOAT})'
    rf'(?:\s+eta_log_shift={_LOG_FLOAT})?'
    rf'(?:\s+mean_over_vgp={_LOG_FLOAT})?(?:\s+vgp_over_mean={_LOG_FLOAT})?',
    re.IGNORECASE,
)
DEBUG_FIELD_RE = re.compile(
    rf'(\w+)=\[({_LOG_FLOAT}),({_LOG_FLOAT})\]',
    re.IGNORECASE,
)
ETA_VS_REF_RE = re.compile(
    rf'eta_vs_ref\s+epoch=(\d+)\s+'
    rf'log10_rmse=({_LOG_FLOAT})\s+log10_bias=({_LOG_FLOAT})\s+log10_r=({_LOG_FLOAT})\s+'
    rf'rel_rmse=({_LOG_FLOAT})\s+eta_pred_mean=({_LOG_FLOAT})\s+eta_ref_mean=({_LOG_FLOAT})',
    re.IGNORECASE,
)
MEAN_NET_FREEZE_RE = re.compile(
    r'epoch\s+(\d+)\s+mean_net\s+(frozen|unfrozen)',
    re.IGNORECASE,
)


class CfgPlotSection:
    """Minimal cfg view for plotting (no PyTorch dependency)."""

    __slots__ = ('logfile', 'metrics_csv', 'plot_dir')

    def __init__(self, logfile: str, metrics_csv: str | None = None, plot_dir: str | None = None):
        self.logfile = logfile
        self.metrics_csv = metrics_csv
        self.plot_dir = plot_dir


def strip_cfg_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def read_plot_paths_from_cfg(cfgfile: str | Path) -> dict[str, CfgPlotSection]:
    """Read log/metrics/plot paths from run_torch.cfg without importing torch."""
    cfg = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    cfg.read(cfgfile)

    def section_paths(section: str, default_logfile: str) -> CfgPlotSection:
        if not cfg.has_section(section):
            return CfgPlotSection(default_logfile)

        def option(key: str) -> str | None:
            if not cfg.has_option(section, key):
                return None
            return strip_cfg_value(cfg.get(section, key))

        logfile = option('logfile') or default_logfile
        return CfgPlotSection(logfile, option('metrics_csv'), option('plot_dir'))

    return {
        'pretrain': section_paths('pretrain', 'logs/log_pretrain_torch'),
        'joint': section_paths('train', 'logs/log_train_torch'),
    }


def metrics_csv_path(logfile: str | Path, stage: str) -> Path:
    log_path = Path(logfile)
    stem = log_path.name or 'training'
    return log_path.parent / f'metrics_{stage}_{stem}.csv'


def default_plot_dir(logfile: str | Path, stage: str) -> Path:
    log_path = Path(logfile)
    stem = log_path.name or 'training'
    return log_path.parent / 'figures' / f'{stage}_{stem}'


def resolve_plot_every(pars_section, default: int = 25) -> int:
    value = getattr(pars_section, 'plot_every', default)
    if value is None:
        return default
    return max(int(value), 0)


def resolve_metrics_csv(pars_section, logfile: str, stage: str) -> Path:
    custom = getattr(pars_section, 'metrics_csv', None)
    if custom:
        return Path(custom)
    return metrics_csv_path(logfile, stage)


def resolve_plot_dir(pars_section, logfile: str, stage: str) -> Path:
    custom = getattr(pars_section, 'plot_dir', None)
    if custom:
        return Path(custom)
    return default_plot_dir(logfile, stage)


class EpochMetricsWriter:
    def __init__(self, csv_path: str | Path, fieldnames: tuple[str, ...], append: bool = False):
        self.csv_path = Path(csv_path)
        self.fieldnames = fieldnames
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Fresh runs must truncate ('w'); only true resumes append ('a').
        # Previously append=False still opened 'a' and rewrote a second header mid-file.
        file_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        if append and file_exists:
            mode = 'a'
            write_header = False
        else:
            mode = 'w'
            write_header = True
        self._file = self.csv_path.open(mode, newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def write_row(self, row: dict) -> None:
        self._writer.writerow({key: row.get(key, '') for key in self.fieldnames})
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def summarize_debug_running(running_debug: dict) -> dict[str, float]:
    count = max(running_debug['count'], 1)

    def _component_avg(name: str) -> float:
        component_count = running_debug.get('loss_component_count', {}).get(name, 0)
        if component_count <= 0:
            return float('nan')
        return running_debug['loss_component_sum'][name] / component_count

    summary = {
        'grad_mean_net': running_debug['grad_norm_sum']['mean_net'] / count,
        'grad_vgp_eta': running_debug['grad_norm_sum']['vgp_eta'] / count,
        'grad_vgp_lambda': running_debug['grad_norm_sum']['vgp_lambda'] / count,
        'train_pde': _component_avg('momentum_nll'),
        'train_bc': _component_avg('continuity_nll'),
        'train_eta_prior': _component_avg('eta_prior'),
        'train_kl_only': _component_avg('kl_only'),
        'vgp_updates': float(running_debug.get('vgp_updates', 0)),
        'mean_updates': float(running_debug.get('mean_updates', 0)),
    }
    mean_g = summary['grad_mean_net']
    eta_g = summary['grad_vgp_eta']
    summary['grad_ratio'] = (mean_g / eta_g) if eta_g > 0.0 else float('nan')

    for field_name, csv_prefix in (
        ('eta', 'eta'),
        ('lambda', 'lambda'),
        ('rux', 'rux'),
        ('rvy', 'rvy'),
        ('rh', 'rh'),
    ):
        stats = running_debug['field_minmax'][field_name]
        summary[f'{csv_prefix}_min'] = stats['min']
        summary[f'{csv_prefix}_max'] = stats['max']
    return summary


def _is_numeric_epoch(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _rows_to_arrays(rows: list[dict], fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    if not rows:
        return {field: np.array([], dtype=float) for field in fields}
    deduped: dict[int, dict] = {}
    for row in rows:
        # Skip duplicated CSV header rows (epoch == 'epoch') from older writer bugs.
        if not _is_numeric_epoch(row.get('epoch')):
            continue
        epoch = int(float(row['epoch']))
        deduped[epoch] = row
    ordered = [deduped[epoch] for epoch in sorted(deduped)]
    arrays = {}
    for field in fields:
        if field == 'epoch':
            arrays[field] = np.array([int(float(row[field])) for row in ordered], dtype=int)
        else:
            values = []
            for row in ordered:
                raw = row.get(field, '')
                if raw is None or raw == '' or not _is_numeric_epoch(raw):
                    values.append(np.nan)
                else:
                    values.append(float(raw))
            arrays[field] = np.array(values, dtype=float)
    return arrays


def load_metrics_csv(csv_path: str | Path, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    path = Path(csv_path)
    if not path.exists():
        return _rows_to_arrays([], fields)
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    return _rows_to_arrays(rows, fields)


def parse_pretrain_log(logfile: str | Path) -> dict[str, np.ndarray]:
    rows = []
    with Path(logfile).open() as handle:
        for line in handle:
            match = PRETRAIN_LOG_RE.match(line.strip())
            if not match:
                continue
            rows.append(
                {
                    'epoch': match.group(1),
                    'train_loss': match.group(2),
                    'test_loss': match.group(3),
                    'u_loss': match.group(4),
                    'v_loss': match.group(5),
                    's_loss': match.group(6),
                    'h_loss': match.group(7),
                    'grad_norm': match.group(8),
                }
            )
    return _rows_to_arrays(rows, PRETRAIN_FIELDS)


def parse_joint_log(logfile: str | Path) -> dict[str, np.ndarray]:
    epoch_rows: dict[int, dict] = {}
    debug_by_epoch: dict[int, dict] = {}
    with Path(logfile).open() as handle:
        for line in handle:
            stripped = line.strip()
            match = JOINT_LOG_RE.match(stripped)
            if match:
                epoch = int(match.group(1))
                # New format: data phys kl state_reg total | test_data test_phys test_kl test_state_reg test_total
                epoch_rows[epoch] = {
                    'epoch': epoch,
                    'train_data': match.group(2),
                    'train_phys': match.group(3),
                    'train_kl': match.group(4),
                    'train_state_reg': match.group(5),
                    'train_total': match.group(6),
                    'test_data': match.group(7),
                    'test_phys': match.group(8),
                    'test_kl': match.group(9),
                    'test_state_reg': match.group(10),
                    'test_total': match.group(11),
                }
                continue
            # Legacy format without state_reg (9 floats after epoch).
            legacy = re.match(
                rf'^INFO:root:(\d+)\s+'
                rf'({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+'
                rf'({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s+({_LOG_FLOAT})\s*$',
                stripped,
                re.IGNORECASE,
            )
            if legacy:
                epoch = int(legacy.group(1))
                epoch_rows[epoch] = {
                    'epoch': epoch,
                    'train_data': legacy.group(2),
                    'train_phys': legacy.group(3),
                    'train_kl': legacy.group(4),
                    'train_total': legacy.group(5),
                    'test_data': legacy.group(6),
                    'test_phys': legacy.group(7),
                    'test_kl': legacy.group(8),
                    'test_total': legacy.group(9),
                }
                continue
            eta_match = ETA_VS_REF_RE.search(stripped)
            if eta_match:
                epoch = int(eta_match.group(1))
                debug_by_epoch.setdefault(epoch, {})
                debug_by_epoch[epoch].update(
                    {
                        'log10_eta_rmse': eta_match.group(2),
                        'log10_eta_bias': eta_match.group(3),
                        'log10_eta_r': eta_match.group(4),
                        'rel_eta_rmse': eta_match.group(5),
                        'eta_pred_mean': eta_match.group(6),
                        'eta_ref_mean': eta_match.group(7),
                    }
                )
                continue
            freeze_match = MEAN_NET_FREEZE_RE.search(stripped)
            if freeze_match:
                epoch = int(freeze_match.group(1))
                debug_by_epoch.setdefault(epoch, {})
                debug_by_epoch[epoch]['mean_net_frozen'] = (
                    1.0 if freeze_match.group(2).lower() == 'frozen' else 0.0
                )
                continue
            debug_match = DEBUG_GRAD_RE.search(stripped)
            if debug_match:
                epoch = int(debug_match.group(1))
                debug_by_epoch.setdefault(epoch, {})
                debug_by_epoch[epoch].update(
                    {
                        'grad_mean_net': debug_match.group(2),
                        'grad_vgp_eta': debug_match.group(3),
                        'grad_vgp_lambda': debug_match.group(4),
                    }
                )
            for field_match in DEBUG_FIELD_RE.finditer(stripped):
                name = field_match.group(1)
                if name in {'eta', 'lambda', 'rux', 'rvy', 'rh'}:
                    epoch_match = DEBUG_GRAD_RE.search(stripped)
                    if epoch_match is None:
                        continue
                    epoch = int(epoch_match.group(1))
                    debug_by_epoch.setdefault(epoch, {})
                    debug_by_epoch[epoch][f'{name}_min'] = field_match.group(2)
                    debug_by_epoch[epoch][f'{name}_max'] = field_match.group(3)

    # Carry eta_vs_ref metrics onto nearby epoch rows even if logged only every N epochs.
    rows = []
    for epoch in sorted(epoch_rows):
        row = dict(epoch_rows[epoch])
        row.update(debug_by_epoch.get(epoch, {}))
        rows.append(row)
    # Also keep eta-only epochs that may lack a full loss line.
    for epoch, extras in debug_by_epoch.items():
        if epoch in epoch_rows:
            continue
        if any(key.startswith('log10_eta') or key == 'eta_pred_mean' for key in extras):
            row = {'epoch': epoch}
            row.update(extras)
            rows.append(row)
    rows.sort(key=lambda item: int(float(item['epoch'])))
    return _rows_to_arrays(rows, JOINT_FIELDS)


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        mask &= np.isfinite(array)
    return mask


def _save_figure(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _log10_positive(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10(np.where(values > 0, values, np.nan))


def _enrich_joint_metrics(metrics: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Fill PDE/BC columns from legacy physics totals when component logs are absent."""
    metrics = dict(metrics)
    n = metrics['epoch'].size
    if n == 0:
        return metrics
    nan_arr = np.full(n, np.nan, dtype=float)

    if not np.any(np.isfinite(metrics.get('train_pde', nan_arr))):
        metrics['train_pde'] = np.asarray(metrics.get('train_phys', nan_arr), dtype=float)
    if not np.any(np.isfinite(metrics.get('test_pde', nan_arr))):
        metrics['test_pde'] = np.asarray(metrics.get('test_phys', nan_arr), dtype=float)
    if 'train_bc' not in metrics:
        metrics['train_bc'] = nan_arr.copy()
    if 'test_bc' not in metrics:
        metrics['test_bc'] = nan_arr.copy()
    return metrics


def _plot_log_train_val(
    ax,
    epoch: np.ndarray,
    train_values: np.ndarray,
    val_values: np.ndarray,
    label: str,
    color: str,
    *,
    linewidth: float = 1.8,
):
    train_y = _log10_positive(train_values)
    val_y = _log10_positive(val_values)
    train_mask = np.isfinite(train_y)
    val_mask = np.isfinite(val_y)
    if train_mask.any():
        ax.plot(
            epoch[train_mask],
            train_y[train_mask],
            color=color,
            linestyle='-',
            linewidth=linewidth,
            label=f'{label} (train)',
        )
    if val_mask.any():
        ax.plot(
            epoch[val_mask],
            val_y[val_mask],
            color=color,
            linestyle='--',
            linewidth=linewidth,
            alpha=0.9,
            label=f'{label} (val)',
        )


def iter_pretrain_figures(metrics: dict[str, np.ndarray]):
    """Yield recommended PINN pretrain loss curves."""
    epoch = metrics['epoch']
    if epoch.size == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.0]})

    _plot_log_train_val(
        axes[0],
        epoch,
        metrics['train_loss'],
        metrics['test_loss'],
        'total loss',
        'black',
        linewidth=2.4,
    )
    axes[0].set_ylabel(r'$\log_{10}$(loss)')
    axes[0].set_title('PINN pretrain — total & validation (most important)')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    _plot_log_train_val(
        axes[1],
        epoch,
        metrics['train_loss'],
        metrics['test_loss'],
        'data loss',
        'tab:blue',
    )
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel(r'$\log_{10}$(loss)')
    axes[1].set_title('PINN pretrain — observation / data fit (no PDE at this stage)')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    yield 'recommended_losses', fig

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in (
        ('u_loss', 'u'),
        ('v_loss', 'v'),
        ('s_loss', 's'),
        ('h_loss', 'h'),
    ):
        mask = _finite_mask(metrics[key])
        if mask.any():
            ax.plot(epoch[mask], _log10_positive(metrics[key][mask]), label=label)
    ax.set_xlabel('epoch')
    ax.set_ylabel(r'$\log_{10}$(component loss)')
    ax.set_title('PINN pretrain — field components')
    ax.legend()
    ax.grid(True, alpha=0.3)
    yield 'loss_components', fig


def _mark_freeze_transition(ax, metrics: dict[str, np.ndarray]) -> None:
    frozen = metrics.get('mean_net_frozen')
    epoch = metrics.get('epoch')
    if frozen is None or epoch is None or epoch.size == 0:
        return
    mask = np.isfinite(frozen)
    if not mask.any():
        return
    # Vertical line at first unfrozen epoch if present.
    unfrozen = epoch[mask & (frozen < 0.5)]
    if unfrozen.size:
        ax.axvline(float(unfrozen[0]), color='0.4', linestyle=':', linewidth=1.2, label='mean_net unfreeze')


def _plot_signed_train_val(
    ax,
    epoch: np.ndarray,
    train_values: np.ndarray,
    val_values: np.ndarray,
    label: str,
    color: str,
):
    train_mask = np.isfinite(train_values)
    val_mask = np.isfinite(val_values)
    if train_mask.any():
        ax.plot(epoch[train_mask], train_values[train_mask], color=color, linestyle='-', label=f'{label} (train)')
    if val_mask.any():
        ax.plot(
            epoch[val_mask],
            val_values[val_mask],
            color=color,
            linestyle='--',
            alpha=0.9,
            label=f'{label} (val)',
        )


def iter_joint_figures(metrics: dict[str, np.ndarray]):
    """Yield recommended VI + PINN joint loss / η-recovery curves."""
    metrics = _enrich_joint_metrics(metrics)
    epoch = metrics['epoch']
    if epoch.size == 0:
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True, gridspec_kw={'height_ratios': [1.1, 1.0, 1.0]})

    # Total can be negative when physics NLL has a small-σ log(σ) floor — plot signed.
    _plot_signed_train_val(
        axes[0],
        epoch,
        metrics['train_total'],
        metrics['test_total'],
        'total loss (ELBO)',
        'black',
    )
    axes[0].set_ylabel('loss')
    axes[0].set_title('Joint train — total & validation (data + PDE + KL + anchors)')
    _mark_freeze_transition(axes[0], metrics)
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    _plot_log_train_val(axes[1], epoch, metrics['train_data'], metrics['test_data'], 'data loss', 'tab:blue')
    if np.any(np.isfinite(metrics.get('train_kl', np.array([])))):
        _plot_log_train_val(
            axes[1], epoch, metrics['train_kl'], metrics['test_kl'], 'KL (+η prior)', 'tab:purple'
        )
    axes[1].set_ylabel(r'$\log_{10}$(loss)')
    axes[1].set_title('Joint train — data and KL / η-prior')
    _mark_freeze_transition(axes[1], metrics)
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Physics can be negative with tight ssa_*_std; use signed axis.
    _plot_signed_train_val(axes[2], epoch, metrics['train_phys'], metrics['test_phys'], 'physics NLL', 'tab:orange')
    if np.any(np.isfinite(metrics.get('train_state_reg', np.array([])))):
        _plot_signed_train_val(
            axes[2],
            epoch,
            metrics['train_state_reg'],
            metrics['test_state_reg'],
            'state reg',
            'tab:green',
        )
    axes[2].set_xlabel('epoch')
    axes[2].set_ylabel('NLL / reg')
    axes[2].set_title('Joint train — physics NLL and state regularization')
    _mark_freeze_transition(axes[2], metrics)
    axes[2].legend(loc='best', fontsize=8)
    axes[2].grid(True, alpha=0.3)
    yield 'recommended_losses', fig

    # η recovery vs spin-up reference (most important for this tuning campaign).
    has_eta = np.any(np.isfinite(metrics.get('log10_eta_bias', np.array([np.nan]))))
    if has_eta:
        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        bias = metrics['log10_eta_bias']
        rmse = metrics['log10_eta_rmse']
        corr = metrics['log10_eta_r']
        mask = np.isfinite(bias) | np.isfinite(rmse) | np.isfinite(corr)
        axes[0].plot(epoch[np.isfinite(bias)], bias[np.isfinite(bias)], color='tab:red', label=r'$\log_{10}$ bias')
        axes[0].axhline(0.0, color='0.5', linestyle='--', linewidth=1.0)
        axes[0].set_ylabel(r'$\log_{10}\eta$ bias')
        axes[0].set_title(r'η recovery vs spin-up — bias (0 = unbiased mean)')
        _mark_freeze_transition(axes[0], metrics)
        axes[0].legend(loc='best', fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epoch[np.isfinite(rmse)], rmse[np.isfinite(rmse)], color='tab:orange', label=r'$\log_{10}$ RMSE')
        if np.any(np.isfinite(corr)):
            axes[1].plot(epoch[np.isfinite(corr)], corr[np.isfinite(corr)], color='tab:green', label=r'$\log_{10}$ corr $r$')
        axes[1].set_ylabel('score')
        axes[1].set_title(r'η recovery — $\log_{10}$ RMSE and correlation')
        _mark_freeze_transition(axes[1], metrics)
        axes[1].legend(loc='best', fontsize=8)
        axes[1].grid(True, alpha=0.3)

        pred = metrics.get('eta_pred_mean')
        ref = metrics.get('eta_ref_mean')
        if pred is not None and np.any(np.isfinite(pred)):
            axes[2].plot(epoch[np.isfinite(pred)], pred[np.isfinite(pred)], color='tab:blue', label=r'predicted $\langle\eta\rangle$')
        if ref is not None and np.any(np.isfinite(ref)):
            axes[2].plot(epoch[np.isfinite(ref)], ref[np.isfinite(ref)], color='black', linestyle='--', label=r'reference $\langle\eta\rangle$')
        axes[2].set_xlabel('epoch')
        axes[2].set_ylabel(r'$\eta$ (MPa·yr)')
        axes[2].set_title('η recovery — mean predicted vs reference')
        _mark_freeze_transition(axes[2], metrics)
        axes[2].legend(loc='best', fontsize=8)
        axes[2].grid(True, alpha=0.3)
        yield 'eta_vs_ref', fig

    # Gradient norms: confirm η pathway is alive relative to mean_net.
    has_grads = np.any(np.isfinite(metrics.get('grad_vgp_eta', np.array([np.nan]))))
    if has_grads:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for key, label, color in (
            ('grad_mean_net', 'mean_net', 'tab:blue'),
            ('grad_vgp_eta', 'vgp_eta', 'tab:red'),
            ('grad_vgp_lambda', 'vgp_lambda', 'tab:gray'),
        ):
            values = metrics.get(key)
            if values is None:
                continue
            mask = np.isfinite(values) & (values > 0)
            if mask.any():
                ax.semilogy(epoch[mask], values[mask], label=label, color=color)
        ax.set_xlabel('epoch')
        ax.set_ylabel('grad norm')
        ax.set_title('Joint train — module gradient norms')
        _mark_freeze_transition(ax, metrics)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        yield 'grad_norms', fig


def display_pretrain_metrics(metrics: dict[str, np.ndarray]) -> list[plt.Figure]:
    return [fig for _, fig in iter_pretrain_figures(metrics)]


def display_joint_metrics(metrics: dict[str, np.ndarray]) -> list[plt.Figure]:
    return [fig for _, fig in iter_joint_figures(metrics)]


def plot_pretrain_metrics(metrics: dict[str, np.ndarray], plot_dir: str | Path) -> list[Path]:
    plot_dir = Path(plot_dir)
    saved: list[Path] = []
    for name, fig in iter_pretrain_figures(metrics):
        path = plot_dir / f'{name}.png'
        _save_figure(fig, path)
        saved.append(path)
    return saved


def plot_joint_metrics(metrics: dict[str, np.ndarray], plot_dir: str | Path) -> list[Path]:
    plot_dir = Path(plot_dir)
    saved: list[Path] = []
    for name, fig in iter_joint_figures(metrics):
        path = plot_dir / f'{name}.png'
        _save_figure(fig, path)
        saved.append(path)
    return saved


def load_pretrain_metrics(csv_path: str | Path | None, logfile: str | Path | None) -> dict[str, np.ndarray]:
    if csv_path is not None and Path(csv_path).exists():
        return load_metrics_csv(csv_path, PRETRAIN_FIELDS)
    if logfile is not None and Path(logfile).exists():
        return parse_pretrain_log(logfile)
    return _rows_to_arrays([], PRETRAIN_FIELDS)


def load_joint_metrics(csv_path: str | Path | None, logfile: str | Path | None) -> dict[str, np.ndarray]:
    if csv_path is not None and Path(csv_path).exists():
        return load_metrics_csv(csv_path, JOINT_FIELDS)
    if logfile is not None and Path(logfile).exists():
        return parse_joint_log(logfile)
    return _rows_to_arrays([], JOINT_FIELDS)


def maybe_plot_training(
    stage: str,
    metrics_csv: str | Path,
    plot_dir: str | Path,
    epoch: int,
    plot_every: int,
    logfile: str | Path | None = None,
) -> list[Path]:
    if plot_every <= 0:
        return []
    if epoch % plot_every != 0 and epoch != 0:
        return []
    try:
        if stage == 'pretrain':
            metrics = load_pretrain_metrics(metrics_csv, logfile)
            return plot_pretrain_metrics(metrics, plot_dir)
        if stage == 'joint':
            metrics = load_joint_metrics(metrics_csv, logfile)
            return plot_joint_metrics(metrics, plot_dir)
        raise ValueError(f'unknown training stage: {stage}')
    except Exception as exc:
        # Never abort training because plotting/metrics parsing failed.
        logging.warning('training plot skipped at epoch %s (%s): %s', epoch, stage, exc)
        return []
