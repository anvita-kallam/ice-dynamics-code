# Sequential VI-only eta-bias suite

This suite searches for better mean-viscosity recovery without sacrificing the
optimized sequential model's full-grid spatial correlation (`log10_eta_r≈0.831`).
The incumbent config and checkpoints are never overwritten.

## Runs

| ID | Controlled change |
|---|---|
| `control_adamw` | Fresh incumbent repeat |
| `weak_prior` | η anchor 0.08→0.03, std 1.5→2.5, KL 0.15→0.10 |
| `raised_prior_center` | η prior center 12→15 MPa·yr |
| `strong_physics` | physics 4→6 and SSA std 0.06→0.05 |
| `gp_capacity` | 32×32, Matérn-5/2, anisotropic length scales |
| `optimizer_ngd` | AdamW + natural-gradient variational-mean updates |
| `optimizer_fast_cosine` | LR 5e-4, shorter cosine schedule |
| `combined_candidate` | Raised center + weak prior/KL + strong physics + 32×32 GP |

The raised-center run is included because the explicit anchor is centered at
`eta_init=12`, while the reference mean is about 14.9. Weakening that anchor
does not necessarily increase η: the SSA residual previously drove η downward.

## Prepare and submit

From `Archive/`:

```bash
python scripts/prepare_eta_bias_suite.py configs/vi_only_eta_bias_suite.json
python scripts/prepare_eta_bias_suite.py configs/vi_only_eta_bias_suite.json --submit
```

Submission starts all eight independent GPU jobs immediately and queues one
CPU collector with an `afterany` dependency. Job IDs are written to:

```text
configs/vi_only_eta_bias_suite/submission.json
```

Each run has isolated config, checkpoint, log, and evaluation directories:

```text
configs/vi_only_eta_bias_suite/<run>.cfg
checkpoints/torch_vi_only/eta_bias_suite/eta_bias_v1/<run>/
logs/log_vi_only_eta_bias_eta_bias_v1_<run>
outputs/vi_only_eta_bias_suite/eta_bias_v1/<run>/
```

## Automatic evaluation and ranking

Every trial evaluates `model_best.pt` on the full geometry mask and writes:

```text
outputs/vi_only_eta_bias_suite/eta_bias_v1/<run>/posterior_summary.json
```

Metrics include η correlation/RMSE/bias/relative RMSE/mean ratio, 1σ and 2σ
coverage, latent uncertainty mean/median/p90, uncertainty-error Spearman
correlation, and state RMSE/normalized RMSE.

The collector writes:

```text
outputs/vi_only_eta_bias_suite/eta_bias_v1/ranking.csv
outputs/vi_only_eta_bias_suite/eta_bias_v1/ranking.json
outputs/vi_only_eta_bias_suite/eta_bias_v1/comparison.md
outputs/vi_only_eta_bias_suite/eta_bias_v1/missing_runs.json
```

Eligibility requires `log10_eta_r >= 0.82`. Ineligible models remain visible
but rank below every eligible model. The composite score rewards correlation,
log and relative RMSE, absolute bias, mean-ratio accuracy, Gaussian calibration,
and uncertainty/error association. `ranking.json` records all components and
weights. The highest-scoring eligible configuration is emitted as the proposed
new default; if no run passes the gate, the incumbent remains the default.

The collector can be rerun safely:

```bash
python scripts/collect_eta_bias_suite.py \
  configs/vi_only_eta_bias_suite/manifest.json
```
