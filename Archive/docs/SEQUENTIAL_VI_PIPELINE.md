# Sequential vs joint training (more_sliding)

Two independent pipelines share **models**, **data loading**, and **loss** code but use **separate configs, scripts, checkpoints, and logs**.

| | Joint (existing) | Sequential (new) |
|---|------------------|------------------|
| Stage 1 PINN | `pretrain_solution_torch.py` + `run_torch.cfg` | `pretrain_solution_torch.py` + `run_torch_sequential.cfg` |
| Stage 2 η | `train_torch.py` + `run_torch.cfg` (alternating) | `train_vi_only_torch.py` + `run_torch_vi_only.cfg` (VGP only) |
| Pretrain ckpt | `checkpoints/torch_pretrain/more_sliding/` | `checkpoints/torch_pretrain/more_sliding_sequential/` |
| Train ckpt | `checkpoints/torch_joint/more_sliding/` | `checkpoints/torch_vi_only/more_sliding/` |
| Log | `logs/log_train_torch_more_sliding` | `logs/log_train_vi_only_more_sliding` |
| Slurm | `slurm/vi_train_more_sliding.sbatch` | `slurm/vi_train_vi_only_more_sliding.sbatch` |
| Predict | `predict_torch.py` | `predict_vi_only_torch.py` |
| Eval | validation script | `evaluate_vi_only_posterior.py` |
| Posterior HDF5 | `outputs/more_sliding_posterior_samples_torch.h5` | `outputs/more_sliding_vi_only_posterior_samples_torch.h5` |

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
sbatch slurm/vi_pretrain_sequential_more_sliding.sbatch
sbatch --dependency=afterok:<PRE_JOBID> slurm/vi_train_vi_only_more_sliding.sbatch
# joint can run in parallel on another GPU
sbatch slurm/vi_train_more_sliding.sbatch
```

## Frozen PINN assumptions

1. Loads `meannet_checkdir/model_best.pt` (sequential pretrain) when `restore=False`.
2. `requires_grad=False` + `eval()` for the entire Stage 2; no PINN optimizer.
3. Physics still differentiates through PINN outputs w.r.t. **coordinates** and through **η** (VGP); PINN weights get no gradients.
4. Checkpoints tagged `architecture=coordinate_only_vi_only_frozen_pinn_v1`.
5. GP extensions (`kernel_type`, anisotropic, …) use **defaults that match the old RBF isotropic GP**, so joint `train_torch.py` is unchanged unless those kwargs are passed (they are not).

## Interpreting the experiment

If extensive VI-only sweeps still yield `log10_eta_r ≈ 0` with a strong PINN fit, the limitation is likely **identifiability of η under SSA + observations**, not joint-optimization interference.
