# DSI cluster — VI training (more_sliding)

Slurm batch scripts for the Archive VI+PINN pipeline on the UChicago DSI cluster. They follow the [DSI preemption checkpoint contract](https://dsi.uchicago.edu): `--signal=B:USR1@300`, `--requeue`, and exit code `99` after saving a checkpoint.

## Prerequisites

1. Clone/sync this repo on the cluster (e.g. under `/project/<account>/ice-dynamics`).
2. Create a PyTorch conda env (`TORCH_ENV`, default name `pytorch`) with `torch`, `numpy`, `scipy`, `h5py`, `matplotlib`.
3. Ensure production spin-up NPZ exists at the path in `run_torch.cfg` (or update the path for cluster layout).
4. From `Archive/`: `mkdir -p logs/slurm`

## Quick submit

```bash
cd /path/to/ice-dynamics/Archive

# Stage 1 — PINN pretrain (12h, preemptable general QoS)
sbatch slurm/vi_pretrain_more_sliding.sbatch
# Optional: sbatch --qos=<your-qos> slurm/vi_pretrain_more_sliding.sbatch

# Stage 2 — VI joint train (after pretrain)
sbatch slurm/vi_train_more_sliding.sbatch

# Or chain both:
PRE=$(sbatch --parsable slurm/vi_pretrain_more_sliding.sbatch)
sbatch --dependency=afterok:${PRE} slurm/vi_train_more_sliding.sbatch

# Stage 3 — posterior samples
TRAIN=$(sbatch --parsable slurm/vi_train_more_sliding.sbatch)
sbatch --dependency=afterok:${TRAIN} slurm/vi_predict_more_sliding.sbatch
```

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `TORCH_ENV` | `pytorch` | Conda env name or prefix with PyTorch |
| `ICE_DYNAMICS_ROOT` | auto-detected | Repo root containing `Archive/` |

## Preemption behavior

- `scripts/dsi_preempt_wrapper.sh` runs training under `setsid` and forwards `SIGUSR1` to the process group.
- `pretrain_solution_torch.py` and `train_torch.py` save an atomic checkpoint and exit `99` on `SIGUSR1`.
- After requeue, `SLURM_RESTART_COUNT > 0` sets `pretrain.restore=True` / `train.restore=True` automatically.

Test preemption during the day:

```bash
scancel --signal=USR1 --batch <jobid>
```

Confirm a checkpoint appears under `checkpoints/` and the job exits with code 99.

## QoS tiers

| Goal | Command |
|------|---------|
| Long overnight training (cheap, preemptable) | `sbatch slurm/vi_train_more_sliding.sbatch` (default `general`) |
| Guaranteed short run (2h max, 4× fairshare) | add `#SBATCH --qos=protected` and `#SBATCH --time=02:00:00` |

## Useful commands

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,QOS,State,Elapsed,ExitCode
scancel <jobid>
```
