#!/usr/bin/env python3
"""Collect, gate, score, and rank sequential eta-bias suite results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


SCORE_VERSION = 'eta_bias_composite_v1'


def clipped(value):
    return max(0.0, min(1.0, float(value)))


def score_result(metrics):
    corr = float(metrics['log10_eta_r'])
    rmse = float(metrics['log10_eta_rmse'])
    rel_rmse = float(metrics['rel_eta_rmse'])
    bias = abs(float(metrics['log10_eta_bias']))
    mean_ratio = max(float(metrics['eta_mean_ratio']), 1.0e-12)
    cal1 = float(metrics['calibration_within_1sigma'])
    cal2 = float(metrics['calibration_within_2sigma'])
    uncertainty_corr = float(metrics['uncertainty_abs_error_spearman'])
    calibration = clipped(
        1.0 - 0.5 * (
            abs(cal1 - 0.6827) / 0.6827
            + abs(cal2 - 0.9545) / 0.9545))
    components = {
        'correlation': clipped(corr),
        'log10_rmse': math.exp(-rmse / 0.30),
        'relative_rmse': math.exp(-rel_rmse / 0.50),
        'absolute_bias': math.exp(-bias / 0.15),
        'mean_ratio': math.exp(-abs(math.log(mean_ratio)) / 0.35),
        'calibration': calibration,
        'uncertainty_error': clipped((uncertainty_corr + 1.0) / 2.0),
    }
    weights = {
        'correlation': 0.25,
        'log10_rmse': 0.15,
        'relative_rmse': 0.10,
        'absolute_bias': 0.15,
        'mean_ratio': 0.15,
        'calibration': 0.15,
        'uncertainty_error': 0.05,
    }
    score = 100.0 * sum(weights[key] * value for key, value in components.items())
    return score, components, weights


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--require-all', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    archive = Path(__file__).resolve().parents[1]
    manifest_path = archive / args.manifest
    manifest = json.loads(manifest_path.read_text())
    gate = float(manifest.get('correlation_gate', 0.82))
    suite_dir = archive / 'outputs/vi_only_eta_bias_suite' / manifest['suite_id']
    suite_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for experiment in manifest['experiments']:
        summary_path = archive / experiment['summary']
        if not summary_path.exists():
            missing.append({
                'id': experiment['id'],
                'expected_summary': experiment['summary'],
            })
            continue
        metrics = json.loads(summary_path.read_text())
        score, components, weights = score_result(metrics)
        eligible = (
            math.isfinite(float(metrics['log10_eta_r']))
            and float(metrics['log10_eta_r']) >= gate)
        rows.append({
            'id': experiment['id'],
            'description': experiment['description'],
            'config': experiment['config'],
            'eligible': eligible,
            'composite_score': score,
            'log10_eta_r': metrics['log10_eta_r'],
            'log10_eta_rmse': metrics['log10_eta_rmse'],
            'log10_eta_bias': metrics['log10_eta_bias'],
            'rel_eta_rmse': metrics['rel_eta_rmse'],
            'eta_pred_mean': metrics['eta_pred_mean'],
            'eta_ref_mean': metrics['eta_ref_mean'],
            'eta_mean_ratio': metrics['eta_mean_ratio'],
            'calibration_1sigma': metrics['calibration_within_1sigma'],
            'calibration_2sigma': metrics['calibration_within_2sigma'],
            'eta_post_std_mean': metrics['eta_post_std_mean'],
            'eta_post_std_p90': metrics['eta_post_std_p90'],
            'uncertainty_error_spearman': metrics['uncertainty_abs_error_spearman'],
            'state_nrmse': metrics['state_nrmse'],
            'checkpoint_epoch': metrics['checkpoint_epoch'],
            'score_components': components,
        })

    rows.sort(key=lambda row: (
        not row['eligible'],
        -row['composite_score'],
        -float(row['log10_eta_r']),
        float(row['log10_eta_rmse']),
        row['id'],
    ))
    for rank, row in enumerate(rows, start=1):
        row['rank'] = rank

    csv_fields = [
        'rank', 'id', 'eligible', 'composite_score', 'log10_eta_r',
        'log10_eta_rmse', 'log10_eta_bias', 'rel_eta_rmse',
        'eta_pred_mean', 'eta_ref_mean', 'eta_mean_ratio',
        'calibration_1sigma', 'calibration_2sigma',
        'eta_post_std_mean', 'eta_post_std_p90',
        'uncertainty_error_spearman', 'state_nrmse', 'checkpoint_epoch',
        'config', 'description',
    ]
    with (suite_dir / 'ranking.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

    recommendation = next((row for row in rows if row['eligible']), None)
    report = {
        'suite_id': manifest['suite_id'],
        'score_version': SCORE_VERSION,
        'correlation_gate': gate,
        'completed': len(rows),
        'expected': len(manifest['experiments']),
        'missing': missing,
        'weights': weights if rows else {},
        'recommended_default': (
            {
                'id': recommendation['id'],
                'config': recommendation['config'],
                'composite_score': recommendation['composite_score'],
                'reason': (
                    f"Highest composite score among runs satisfying "
                    f"log10_eta_r >= {gate:.3f}."),
            } if recommendation else None),
        'ranking': rows,
    }
    (suite_dir / 'ranking.json').write_text(json.dumps(report, indent=2))
    (suite_dir / 'missing_runs.json').write_text(json.dumps(missing, indent=2))

    lines = [
        f'# Sequential eta-bias suite: {manifest["suite_id"]}',
        '',
        f'Hard correlation gate: `log10_eta_r >= {gate:.3f}`  ',
        f'Completed: {len(rows)}/{len(manifest["experiments"])}',
        '',
        '| Rank | Run | Eligible | Score | r | log RMSE | bias | mean η | cal 1σ | cal 2σ |',
        '|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['rank']} | {row['id']} | {'yes' if row['eligible'] else 'no'} "
            f"| {row['composite_score']:.2f} | {float(row['log10_eta_r']):.3f} "
            f"| {float(row['log10_eta_rmse']):.3f} "
            f"| {float(row['log10_eta_bias']):+.3f} "
            f"| {float(row['eta_pred_mean']):.2f} "
            f"| {float(row['calibration_1sigma']):.3f} "
            f"| {float(row['calibration_2sigma']):.3f} |")
    if recommendation:
        lines.extend([
            '',
            f"Recommended new default: **{recommendation['id']}** "
            f"(`{recommendation['config']}`).",
        ])
    elif rows:
        lines.extend([
            '',
            'No completed run satisfies the spatial-correlation gate; keep the incumbent.',
        ])
    if missing:
        lines.extend(['', f'Missing/failed runs: {", ".join(x["id"] for x in missing)}'])
    (suite_dir / 'comparison.md').write_text('\n'.join(lines) + '\n')

    print('\n'.join(lines))
    if args.require_all and missing:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
