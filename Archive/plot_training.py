#!/usr/bin/env python3
"""Generate PINN/VI training diagnostic plots from metrics CSV or log files."""

from __future__ import annotations

import sys
from pathlib import Path

from training_metrics import (
    load_joint_metrics,
    load_pretrain_metrics,
    plot_joint_metrics,
    plot_pretrain_metrics,
    resolve_metrics_csv,
    resolve_plot_dir,
)

usage = """
Usage:
  python plot_training.py run_torch.cfg [--stage pretrain|joint|all]
  python plot_training.py --metrics-csv path/to/metrics.csv --stage pretrain|joint
"""


def _plot_stage(stage: str, pars) -> list[Path]:
    if stage == 'pretrain':
        section = pars.pretrain
        metrics_csv = resolve_metrics_csv(section, section.logfile, 'pretrain')
        plot_dir = resolve_plot_dir(section, section.logfile, 'pretrain')
        metrics = load_pretrain_metrics(metrics_csv, section.logfile)
        return plot_pretrain_metrics(metrics, plot_dir)
    if stage == 'joint':
        section = pars.train
        metrics_csv = resolve_metrics_csv(section, section.logfile, 'joint')
        plot_dir = resolve_plot_dir(section, section.logfile, 'joint')
        metrics = load_joint_metrics(metrics_csv, section.logfile)
        return plot_joint_metrics(metrics, plot_dir)
    raise ValueError(stage)


def main(argv: list[str]) -> int:
    stage = 'all'
    metrics_csv = None
    cfg_path = None

    args = list(argv)
    while args:
        token = args.pop(0)
        if token == '--stage' and args:
            stage = args.pop(0)
        elif token == '--metrics-csv' and args:
            metrics_csv = Path(args.pop(0))
        elif token in ('-h', '--help'):
            print(usage)
            return 0
        elif cfg_path is None:
            cfg_path = token
        else:
            print(f'Unrecognized argument: {token}')
            print(usage)
            return 1

    if metrics_csv is not None:
        if stage not in ('pretrain', 'joint'):
            print('--metrics-csv requires --stage pretrain or joint')
            return 1
        plot_dir = metrics_csv.parent / 'figures' / f'{stage}_{metrics_csv.stem}'
        if stage == 'pretrain':
            saved = plot_pretrain_metrics(load_pretrain_metrics(metrics_csv, None), plot_dir)
        else:
            saved = plot_joint_metrics(load_joint_metrics(metrics_csv, None), plot_dir)
        for path in saved:
            print(path)
        return 0

    if cfg_path is None:
        print(usage)
        return 1

    from utilities_torch import ParameterClass

    pars = ParameterClass(cfg_path)
    if stage not in ('all', 'pretrain', 'joint'):
        print(f'Unknown stage: {stage}')
        return 1
    stages = ('pretrain', 'joint') if stage == 'all' else (stage,)

    saved_all: list[Path] = []
    for item in stages:
        saved = _plot_stage(item, pars)
        saved_all.extend(saved)
        if not saved:
            section = pars.pretrain if item == 'pretrain' else pars.train
            csv_path = resolve_metrics_csv(section, section.logfile, item)
            print(f'No {item} metrics found at {csv_path} or {section.logfile}')
        else:
            print(f'{item}: wrote {len(saved)} plot(s) to {saved[0].parent}')
            for path in saved:
                print(path)
    return 0 if saved_all else 1


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
