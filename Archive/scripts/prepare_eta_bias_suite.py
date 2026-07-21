#!/usr/bin/env python3
"""Generate and optionally submit the isolated sequential eta-bias suite."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def format_value(value):
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if isinstance(value, str):
        return repr(value)
    return str(value)


def set_cfg_value(text, section, key, value):
    lines = text.splitlines(keepends=True)
    output = []
    in_section = False
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            if in_section and not replaced:
                output.append(f'{key} = {format_value(value)}\n')
                replaced = True
            in_section = stripped == f'[{section}]'
            output.append(line)
            continue
        if in_section and '=' in stripped:
            current_key = stripped.split('=', 1)[0].strip()
            if current_key == key:
                output.append(f'{key} = {format_value(value)}\n')
                replaced = True
                continue
        output.append(line)
    if in_section and not replaced:
        output.append(f'{key} = {format_value(value)}\n')
        replaced = True
    if not replaced:
        raise KeyError(f'Config section [{section}] not found for {key}')
    return ''.join(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'spec', nargs='?', default='configs/vi_only_eta_bias_suite.json')
    parser.add_argument('--submit', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    archive = Path(__file__).resolve().parents[1]
    spec_path = archive / args.spec
    spec = json.loads(spec_path.read_text())
    suite_id = spec['suite_id']
    base_path = archive / spec['base_cfg']
    base_text = base_path.read_text()
    cfg_dir = archive / 'configs' / 'vi_only_eta_bias_suite'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path('outputs/vi_only_eta_bias_suite') / suite_id
    checkpoint_root = Path('checkpoints/torch_vi_only/eta_bias_suite') / suite_id

    records = []
    for experiment in spec['experiments']:
        run_id = experiment['id']
        text = base_text
        common = {
            'train.checkdir': str(checkpoint_root / run_id),
            'train.logfile': f'logs/log_vi_only_eta_bias_{suite_id}_{run_id}',
            'train.evaldir': str(output_root / run_id),
            'train.restore': False,
            'train.restore_optimizer': False,
            'train.checkname_old': 'model',
            'train.checkname_new': 'model',
            'train.checkname_best': 'model_best',
            'train.seed': 20260721,
            'predict.output_file': str(output_root / run_id / 'posterior_samples.h5'),
        }
        all_overrides = {**common, **experiment.get('overrides', {})}
        for dotted_key, value in all_overrides.items():
            section, key = dotted_key.split('.', 1)
            text = set_cfg_value(text, section, key, value)
        header = (
            f'# Eta-bias suite {suite_id}: {run_id}\n'
            f'# {experiment["description"]}\n'
            f'# Generated from {spec["base_cfg"]}; do not edit by hand.\n\n'
        )
        cfg_path = cfg_dir / f'{run_id}.cfg'
        cfg_path.write_text(header + text)
        records.append({
            'id': run_id,
            'description': experiment['description'],
            'config': str(cfg_path.relative_to(archive)),
            'overrides': experiment.get('overrides', {}),
            'checkdir': str(checkpoint_root / run_id),
            'logfile': f'logs/log_vi_only_eta_bias_{suite_id}_{run_id}',
            'evaldir': str(output_root / run_id),
            'summary': str(output_root / run_id / 'posterior_summary.json'),
        })

    manifest = {
        'suite_id': suite_id,
        'base_cfg': spec['base_cfg'],
        'correlation_gate': spec.get('correlation_gate', 0.82),
        'experiments': records,
    }
    manifest_path = cfg_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'Prepared {len(records)} runs: {manifest_path.relative_to(archive)}')

    if not args.submit:
        print(
            f'Submit all runs with: python scripts/prepare_eta_bias_suite.py '
            f'{args.spec} --submit')
        return 0

    job_ids = []
    for record in records:
        result = subprocess.run(
            [
                'sbatch', '--parsable', f'--job-name=eta_{record["id"][:8]}',
                'slurm/vi_only_eta_bias_trial.sbatch',
                record['config'], record['id'],
            ],
            cwd=archive, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(';', 1)[0]
        job_ids.append(job_id)
        print(f'{record["id"]}: {job_id}')

    dependency = 'afterany:' + ':'.join(job_ids)
    result = subprocess.run(
        [
            'sbatch', '--parsable', f'--dependency={dependency}',
            'slurm/vi_only_eta_bias_collect.sbatch',
            str(manifest_path.relative_to(archive)),
        ],
        cwd=archive, check=True, capture_output=True, text=True)
    collector_job = result.stdout.strip().split(';', 1)[0]
    submission = {
        'suite_id': suite_id,
        'jobs': dict(zip((r['id'] for r in records), job_ids)),
        'collector_job': collector_job,
    }
    submission_path = cfg_dir / 'submission.json'
    submission_path.write_text(json.dumps(submission, indent=2))
    print(f'collector: {collector_job}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
