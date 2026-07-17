#!/usr/bin/env python3
"""Generate PINN/VI training diagnostic plots from metrics CSV or log files.

Does not require PyTorch — only numpy/matplotlib and the training log/CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

from training_metrics import (
    load_joint_metrics,
    load_pretrain_metrics,
    plot_joint_metrics,
    plot_pretrain_metrics,
    read_plot_paths_from_cfg,
    resolve_metrics_csv,
    resolve_plot_dir,
)

usage = """
Usage:
  python plot_training.py run_torch.cfg [--stage pretrain|joint|all]
  python plot_training.py --metrics-csv path/to/metrics.csv --stage pretrain|joint
  python plot_training.py --logfile path/to/log_train_... --stage joint
"""


def _plot_stage_from_cfg(stage: str, cfg_path: Path) -> list[Path]:
    paths = read_plot_paths_from_cfg(cfg_path)
    section = paths[stage]
    # Build a tiny namespace compatible with resolve_* helpers.
    class _Sec:
        pass

    sec = _Sec()
    sec.logfile = section.logfile
    sec.metrics_csv = section.metrics_csv
    sec.plot_dir = section.plot_dir

    metrics_csv = resolve_metrics_csv(sec, sec.logfile, stage)
    plot_dir = resolve_plot_dir(sec, sec.logfile, stage)
    if stage == 'pretrain':
        metrics = load_pretrain_metrics(
            metrics_csv if Path(metrics_csv).exists() else None,
            sec.logfile if Path(sec.logfile).exists() else None,
        )
        return plot_pretrain_metrics(metrics, plot_dir)
    if stage == 'joint':
        metrics = load_joint_metrics(
            metrics_csv if Path(metrics_csv).exists() else None,
            sec.logfile if Path(sec.logfile).exists() else None,
        )
        return plot_joint_metrics(metrics, plot_dir)
    raise ValueError(stage)


def main(argv: list[str]) -> int:
    stage = 'all'
    metrics_csv = None
    logfile = None
    cfg_path = None

    args = list(argv)
    while args:
        token = args.pop(0)
        if token == '--stage' and args:
            stage = args.pop(0)
        elif token == '--metrics-csv' and args:
            metrics_csv = Path(args.pop(0))
        elif token == '--logfile' and args:
            logfile = Path(args.pop(0))
        elif token in ('-h', '--help'):
            print(usage)
            return 0
        elif cfg_path is None and not token.startswith('-'):
            cfg_path = token
        else:
            print(f'Unrecognized argument: {token}')
            print(usage)
            return 1

    if metrics_csv is not None or logfile is not None:
        if stage not in ('pretrain', 'joint'):
            print('--metrics-csv/--logfile requires --stage pretrain or joint')
            return 1
        if metrics_csv is not None:
            plot_dir = metrics_csv.parent / 'figures' / f'{stage}_{metrics_csv.stem}'
            if stage == 'pretrain':
                metrics = load_pretrain_metrics(metrics_csv, logfile)
                saved = plot_pretrain_metrics(metrics, plot_dir)
            else:
                metrics = load_joint_metrics(metrics_csv, logfile)
                saved = plot_joint_metrics(metrics, plot_dir)
        else:
            plot_dir = logfile.parent / 'figures' / f'{stage}_{logfile.name}'
            if stage == 'pretrain':
                metrics = load_pretrain_metrics(None, logfile)
                saved = plot_pretrain_metrics(metrics, plot_dir)
            else:
                metrics = load_joint_metrics(None, logfile)
                saved = plot_joint_metrics(metrics, plot_dir)
        if not saved:
            print(f'No {stage} metrics parsed from {metrics_csv or logfile}')
            return 1
        print(f'{stage}: wrote {len(saved)} plot(s) to {saved[0].parent}')
        for path in saved:
            print(path)
        return 0

    if cfg_path is None:
        print(usage)
        return 1

    if stage not in ('all', 'pretrain', 'joint'):
        print(f'Unknown stage: {stage}')
        return 1
    stages = ('pretrain', 'joint') if stage == 'all' else (stage,)

    saved_all: list[Path] = []
    for item in stages:
        saved = _plot_stage_from_cfg(item, Path(cfg_path))
        saved_all.extend(saved)
        if not saved:
            print(f'No {item} metrics found for {cfg_path}')
        else:
            print(f'{item}: wrote {len(saved)} plot(s) to {saved[0].parent}')
            for path in saved:
                print(path)
    return 0 if saved_all else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
