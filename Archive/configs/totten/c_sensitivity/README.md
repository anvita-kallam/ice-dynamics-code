# Totten C-sensitivity investigation

Config-flagged experiments to make no-sliding (`C=100`) vs max-sliding (`C=0.001`)
VI η posteriors more distinguishable **without changing SSA / Icepack equations**.

## New flags (baseline defaults preserve old behavior)

| Flag | Default | Effect |
|------|---------|--------|
| `train.grounded_phys_weight` | `1.0` | Weight SSA physics NLL on τ_c>0 points |
| `train.floating_phys_weight` | `1.0` | Weight on floating points |
| `train.learn_eta_shift` | `True` | Learn global `eta_log_shift` |
| `train.freeze_eta_log_shift` | `False` | If `True` (or `learn_eta_shift=False`), freeze shift at 0 |

## Experiments

See [`EXPERIMENTS.md`](EXPERIMENTS.md) (generated).

| ID | Maps to |
|----|---------|
| `baseline` | Control (1/1 weights, learnable shift) |
| `phys_w2` / `phys_w5` / `phys_w10` | Exp 1 — grounded physics up-weight |
| `freeze_shift` | Exp 3 — no global η shift |
| `gp_short_ls` / `gp_flex` | Exp 4 — shorter / more flexible GP |

Exps **2, 5, 6** are analysis-only (grounded stats, residual maps, dual correlations)
via `scripts/totten_c_sensitivity_analyze.py`.

## Cluster workflow

```bash
# Mac → cluster
rsync -avz Archive/models_torch.py Archive/train_vi_only_torch.py \
  Archive/utilities_torch.py Archive/training_metrics.py \
  Archive/scripts/generate_totten_c_sensitivity_cfgs.py \
  Archive/slurm/vi_train_vi_only_cfg.sbatch \
  Archive/slurm/vi_eval_totten_c_sensitivity.sbatch \
  Archive/slurm/submit_totten_c_sensitivity_suite.sh \
  Archive/configs/totten/ \
  login.ds:~/ice-dynamics/Archive/

rsync -avz scripts/totten_c_sensitivity_*.py \
  login.ds:~/ice-dynamics/scripts/

# Cluster
cd ~/ice-dynamics/Archive
python scripts/generate_totten_c_sensitivity_cfgs.py
bash slurm/submit_totten_c_sensitivity_suite.sh          # all
# or: bash slurm/submit_totten_c_sensitivity_suite.sh phys_w5 freeze_shift gp_flex
```

After jobs finish, pull:

```bash
rsync -avz login.ds:~/ice-dynamics/outputs/figures/vi/totten_c_sensitivity/ \
  outputs/figures/vi/totten_c_sensitivity/
```

Report: `outputs/figures/vi/totten_c_sensitivity/COMPARISON_REPORT.md`
