# Totten η posterior-collapse fix

## What changed

1. **Config knobs** (no SSA / C / g changes):
   - `prior.eta_init` — prior center in MPa·yr
   - `prior.eta_bound_mode` — `log_clamp` (default, legacy) or `softplus_floor`
   - `train.eta_prior_mean` — `'auto'` (= η_init) or a float viscosity

2. **Instrumentation** (`eta_eval_every > 0`): CSV columns for mean/median/min/max/std,
   frac at floor (full + grounded), `eta_log_shift`, KL / soft-prior terms, `grad_vgp_eta`,
   `grad_eta_log_shift`. Plots via `training_metrics` → `eta_collapse.png`.

3. **Sweep configs**: `Archive/configs/totten/eta_collapse_sweep/`
   - η_init ∈ {15, 20, 25, 30} with `log_clamp`
   - η_init ∈ {15, 20} with `softplus_floor`

## Run on DSI

```bash
cd ~/ice-dynamics/Archive
sbatch slurm/vi_totten_eta_collapse_sweep.sbatch
```

Requires existing Totten pretrain: `checkpoints/torch_pretrain/totten/model_best.pt`.

## Compare after runs

```bash
python scripts/compare_totten_eta_collapse_sweep.py
```

Writes `outputs/figures/vi/totten_sliding_comparison/eta_collapse_sweep/COMPARISON.md`.

See also `PRIOR_DIAGNOSIS.md` in that folder.
