#!/usr/bin/env python3
"""Launch a small VI-only hyperparameter sweep from a JSON grid.

Example grid file (Archive/configs/vi_only_sweep_grid.json):
{
  "base_cfg": "run_torch_vi_only.cfg",
  "output_dir": "configs/vi_only_sweeps",
  "slurm_script": "slurm/vi_train_vi_only_more_sliding.sbatch",
  "submit": false,
  "grid": {
    "prior.eta_prior_scale": [0.05, 0.08, 0.15],
    "prior.kl_eta": [0.05, 0.15],
    "train.vgp_eta_lr": [5.0e-4, 1.0e-3],
    "prior.kernel_type": ["rbf", "matern32"],
    "prior.num_inducing_x": [24],
    "prior.num_inducing_y": [24]
  }
}

Writes one cfg per combination and optionally sbatch's them with
  CONFIG=<cfg>  (wrapper must read CONFIG — see generated sbatch note).

Usage:
  python scripts/launch_vi_only_sweep.py configs/vi_only_sweep_grid.json
"""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _format_cfg_value(value):
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if isinstance(value, str):
        return repr(value)
    return value


def _set_cfg_value(text: str, section: str, key: str, value) -> str:
    """Replace or insert key=value inside [section]."""
    value = _format_cfg_value(value)
    lines = text.splitlines(keepends=True)
    out = []
    in_section = False
    replaced = False
    section_header = f'[{section}]'
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            if in_section and not replaced:
                out.append(f'{key} = {value}\n')
                replaced = True
            in_section = stripped == section_header
            out.append(line)
            continue
        if in_section and stripped.startswith(f'{key}') and '=' in stripped:
            # match exact key (avoid eta_prior_scale matching eta_prior_std)
            left = stripped.split('=', 1)[0].strip()
            if left == key:
                out.append(f'{key} = {value}\n')
                replaced = True
                continue
        out.append(line)
    if in_section and not replaced:
        out.append(f'{key} = {value}\n')
    return ''.join(out)


def main(argv):
    if len(argv) != 1:
        print(__doc__)
        return 1
    grid_path = Path(argv[0])
    spec = json.loads(grid_path.read_text())
    archive = Path(__file__).resolve().parents[1]
    base_cfg = archive / spec['base_cfg']
    out_dir = archive / spec.get('output_dir', 'configs/vi_only_sweeps')
    out_dir.mkdir(parents=True, exist_ok=True)
    base_text = base_cfg.read_text()
    grid = spec['grid']
    keys = list(grid.keys())
    values = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]

    written = []
    for combo in itertools.product(*values):
        text = base_text
        tag_parts = []
        for key, val in zip(keys, combo):
            section, name = key.split('.', 1)
            # Map eta_prior_scale from prior.* mistake — accept train. too
            if section == 'prior' and name in (
                'eta_prior_scale', 'eta_prior_std', 'vgp_eta_lr', 'vgp_optimizer',
                'lr_scheduler', 'early_stop_metric', 'early_stop_patience',
            ):
                section = 'train'
            text = _set_cfg_value(text, section, name, val)
            safe = str(val).replace('.', 'p').replace('-', 'm')
            tag_parts.append(f'{name}-{safe}')
        tag = '__'.join(tag_parts)[:120]
        # Isolate checkpoint / log / output per trial
        text = _set_cfg_value(text, 'train', 'checkdir', f'checkpoints/torch_vi_only/sweep_{tag}')
        text = _set_cfg_value(text, 'train', 'logfile', f'logs/log_train_vi_only_sweep_{tag}')
        text = _set_cfg_value(
            text, 'predict', 'output_file',
            f'outputs/vi_only_sweep_{tag}_posterior.h5')
        text = _set_cfg_value(text, 'train', 'restore', 'False')
        cfg_path = out_dir / f'run_{tag}.cfg'
        cfg_path.write_text(text)
        written.append(cfg_path)

    manifest = out_dir / 'manifest.json'
    manifest.write_text(json.dumps([str(p.relative_to(archive)) for p in written], indent=2))
    print(f'Wrote {len(written)} configs under {out_dir}')

    if spec.get('submit'):
        wrapper = archive / 'slurm' / 'vi_train_vi_only_sweep_one.sbatch'
        if not wrapper.exists():
            print(f'Missing {wrapper}; configs written but not submitted')
            return 0
        for cfg in written:
            subprocess.run(
                ['sbatch', str(wrapper), str(cfg.relative_to(archive))],
                cwd=str(archive), check=False)
    else:
        print('submit=false — to launch one trial:')
        print(f'  sbatch slurm/vi_train_vi_only_sweep_one.sbatch {written[0].relative_to(archive)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
