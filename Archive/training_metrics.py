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
    'lr_mean_net',
    'lr_vgp_eta',
    'lr_vgp_lambda',
    'log10_eta_rmse',
    'log10_eta_bias',
    'log10_eta_r',
    'rel_eta_rmse',
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
    rf'(?:\s+eta_log_shift={_LOG_FLOAT})?(?:\s+vgp_over_mean={_LOG_FLOAT})?',
    re.IGNORECASE,
)
DEBUG_FIELD_RE = re.compile(
    rf'(\w+)=\[({_LOG_FLOAT}),({_LOG_FLOAT})\]',
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
    }
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

    rows = []
    for epoch in sorted(epoch_rows):
        row = dict(epoch_rows[epoch])
        row.update(debug_by_epoch.get(epoch, {}))
        rows.append(row)
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


def iter_joint_figures(metrics: dict[str, np.ndarray]):
    """Yield recommended VI + PINN joint loss curves."""
    metrics = _enrich_joint_metrics(metrics)
    epoch = metrics['epoch']
    if epoch.size == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, gridspec_kw={'height_ratios': [1.2, 1.4]})

    _plot_log_train_val(
        axes[0],
        epoch,
        metrics['train_total'],
        metrics['test_total'],
        'total loss (ELBO)',
        'black',
        linewidth=2.4,
    )
    axes[0].set_ylabel(r'$\log_{10}$(loss)')
    axes[0].set_title('Joint train — total & validation (data + PDE + KL)')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    _plot_log_train_val(axes[1], epoch, metrics['train_data'], metrics['test_data'], 'data loss', 'tab:blue')
    _plot_log_train_val(axes[1], epoch, metrics['train_pde'], metrics['test_pde'], 'PDE residual', 'tab:orange')
    if np.any(np.isfinite(metrics['train_bc'])) or np.any(np.isfinite(metrics['test_bc'])):
        _plot_log_train_val(
            axes[1], epoch, metrics['train_bc'], metrics['test_bc'], 'BC / continuity', 'tab:green'
        )
    else:
        axes[1].text(
            0.02,
            0.04,
            'BC loss not logged (ssa_enforce_continuity=False in this run)',
            transform=axes[1].transAxes,
            fontsize=9,
            color='0.35',
        )
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel(r'$\log_{10}$(loss)')
    axes[1].set_title('Joint train — data, PDE momentum, and boundary / continuity')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    yield 'recommended_losses', fig


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
