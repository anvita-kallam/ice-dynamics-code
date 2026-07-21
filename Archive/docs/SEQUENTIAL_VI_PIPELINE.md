# Sequential vs joint training (more_sliding)

Two independent pipelines share **models**, **data loading**, and **loss** code but use **separate configs, scripts, checkpoints, and logs**.

| | Joint (existing) | Sequential baseline | Sequential optimized |
|---|------------------|---------------------|----------------------|
| Stage 1 PINN | `pretrain_solution_torch.py` + `run_torch.cfg` | `pretrain_solution_torch.py` + `run_torch_sequential.cfg` | same sequential pretrain |
| Stage 2 η | `train_torch.py` + `run_torch.cfg` | `train_vi_only_torch.py` + `run_torch_vi_only.cfg` | `run_torch_vi_only_optimized.cfg` |
| Pretrain ckpt | `checkpoints/torch_pretrain/more_sliding/` | `checkpoints/torch_pretrain/more_sliding_sequential/` | same |
| Train ckpt | `checkpoints/torch_joint/more_sliding/` | `checkpoints/torch_vi_only/more_sliding/` (+ archive `more_sliding_baseline_r082/`) | `checkpoints/torch_vi_only/more_sliding_optimized/` (`model_best.pt`) |
| Log | `logs/log_train_torch_more_sliding` | `logs/log_train_vi_only_more_sliding` | `logs/log_train_vi_only_more_sliding_optimized` |
| Slurm | `slurm/vi_train_more_sliding.sbatch` | `slurm/vi_train_vi_only_more_sliding.sbatch` (refuses overwrite) | `slurm/vi_train_vi_only_optimized_more_sliding.sbatch` |
| Predict | `predict_torch.py` | `predict_vi_only_torch.py` | same + optimized cfg |
| Posterior HDF5 | `outputs/more_sliding_posterior_samples_torch.h5` | `outputs/more_sliding_vi_only_posterior_samples_torch.h5` | `outputs/more_sliding_vi_only_optimized_posterior_samples_torch.h5` |

### Baseline sequential result (job 1141139, preserved)

- Early stop at epoch 100 on `log10_eta_r` (best ≈ **0.826**, final ≈ **0.812**)
- `log10_bias` ≈ −0.15, `eta_pred_mean` ≈ 8.8 vs `eta_ref_mean` ≈ 15.0
- Stronger spatial recovery than joint (`log10_r` ≈ 0.74) → η identifiable with frozen PINN
- Keep `checkpoints/torch_vi_only/more_sliding/`; sbatch auto-archives to `more_sliding_baseline_r082/` before any forced overwrite

## VI-only spatial-η tooling

Config knobs (all in `run_torch_vi_only.cfg`, no code edits):

| Knob | Section | Role |
|------|---------|------|
| `kernel_type` | prior | `rbf` / `matern12` / `matern32` / `matern52` |
| `anisotropic` | prior | separate `l_scale_eta` / `l_scale_eta_y` |
| `learnable_length_scale` | prior | freeze or train length scales |
| `num_inducing_x/y` + `inducing_placement` | prior | spatial capacity (`ice_fps`) |
| `kl_eta` | prior | inducing KL weight |
| `eta_prior_scale` / `eta_prior_std` | train | soft mean prior |
| `vgp_eta_lr` | train | VGP learning rate |
| `vgp_optimizer` | train | `adam` / `adamw` / `adamw_ngd` |
| `vgp_ngd_lr` | train | natural-grad step on variational mean |
| `lr_scheduler` | train | `cosine` / `plateau` / `exponential` / `none` |
| `early_stop_metric` + `patience` | train | `log10_eta_r` / `log10_eta_rmse` / `test_total` |

**Sweep** (writes isolated cfgs + optional sbatch):

```bash
python scripts/launch_vi_only_sweep.py configs/vi_only_sweep_grid.json
sbatch slurm/vi_train_vi_only_sweep_one.sbatch configs/vi_only_sweeps/run_....cfg
```

**Posterior quality** (maps, calibration, uncertainty vs error):

```bash
python evaluate_vi_only_posterior.py run_torch_vi_only.cfg --tag more_sliding
```

## Cluster: sequential pipeline

```bash
cd ~/ice-dynamics/Archive
# Stage 1 (done for more_sliding): sequential pretrain
# sbatch slurm/vi_pretrain_sequential_more_sliding.sbatch

# Optimized Stage 2 (preferred; isolated from baseline)
sbatch slurm/vi_train_vi_only_optimized_more_sliding.sbatch

# Explicitly archive the completed baseline if not already
# cp -a checkpoints/torch_vi_only/more_sliding \
#      checkpoints/torch_vi_only/more_sliding_baseline_r082
# cp -a logs/log_train_vi_only_more_sliding \
#      logs/log_train_vi_only_more_sliding_baseline_r082
```

## Frozen PINN assumptions

1. Loads `meannet_checkdir/model_best.pt` (sequential pretrain) when `restore=False`.
2. `requires_grad=False` + `eval()` for the entire Stage 2; no PINN optimizer.
3. Physics still differentiates through PINN outputs w.r.t. **coordinates** and through **η** (VGP); PINN weights get no gradients.
4. Checkpoints tagged `architecture=coordinate_only_vi_only_frozen_pinn_v1`.
5. GP extensions (`kernel_type`, anisotropic, …) use **defaults that match the old RBF isotropic GP**, so joint `train_torch.py` is unchanged unless those kwargs are passed (they are not).

## Interpreting the experiment

Baseline sequential VI-only already recovered **strong spatial η** (`log10_r≈0.82`) with a frozen PINN, while joint training sat near `≈0.74`. That points to **joint-optimization interference**, not pure η non-identifiability under SSA + observations. Further VI-only tuning (optimized cfg) tests whether mean bias and correlation can improve without unfreezing the PINN.
