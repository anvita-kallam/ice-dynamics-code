#!/bin/bash
#SBATCH --job-name=vi_holdout_vel
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Held-out velocity validation on MISMIP A=20 more_sliding.
# Reuses checkpoints/torch_pretrain/more_sliding_sequential/model_best.pt
# (does not retrain the PINN).

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs checkpoints outputs

PRETRAIN_CKPT="checkpoints/torch_pretrain/more_sliding_sequential/model_best.pt"
if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
    echo "Missing sequential pretrain checkpoint: ${PRETRAIN_CKPT}" >&2
    exit 1
fi

CFG="configs/heldout_velocity_validation/run_torch_vi_only_a20_holdout.cfg"
OUT_DIR="${ICE_DYNAMICS_ROOT}/outputs/heldout_velocity_validation"

if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
    # shellcheck source=/dev/null
    source /opt/conda/etc/profile.d/conda.sh
    conda activate "${TORCH_ENV}"
fi

# 1) Build holdout dataset (idempotent; never overwrites the original A=20 NPZ).
python "${ICE_DYNAMICS_ROOT}/scripts/prepare_velocity_holdout_dataset.py" \
    --source "${ICE_DYNAMICS_ROOT}/outputs/spinup/production/more_sliding/SteadyState_more_sliding_10500yr_ramp4000_1refine_grid.npz" \
    --output-dir "${OUT_DIR}/dataset" \
    --holdout-fraction 0.2 \
    --seed 42 \
    --tag f0.2_s42

# 2) VI-only Stage 2 with frozen PINN (preempt-safe).
bash scripts/dsi_preempt_wrapper.sh \
    train_vi_only_torch.py "${CFG}"

# 3) Train vs held-out velocity evaluation + figures.
python "${ICE_DYNAMICS_ROOT}/scripts/evaluate_heldout_velocity_validation.py" \
    --cfg "${ARCHIVE_DIR}/${CFG}" \
    --output-dir "${OUT_DIR}" \
    --checkpoint best
