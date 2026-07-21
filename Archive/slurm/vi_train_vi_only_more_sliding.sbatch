#!/bin/bash
#SBATCH --job-name=vi_only_ms
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Sequential experiment — Stage 2: VI only (frozen PINN from sequential pretrain).
# Config: run_torch_vi_only.cfg
# Checkpoints: checkpoints/torch_vi_only/more_sliding/
#
# Chain after pretrain:
#   PRE=$(sbatch --parsable slurm/vi_pretrain_sequential_more_sliding.sbatch)
#   sbatch --dependency=afterok:${PRE} slurm/vi_train_vi_only_more_sliding.sbatch
#
# Independent of joint training — safe to run in parallel on another GPU.

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
    echo "Run slurm/vi_pretrain_sequential_more_sliding.sbatch first." >&2
    exit 1
fi

# Protect the completed baseline run (log10_r≈0.82) from accidental wipe.
BASELINE_CKPT="checkpoints/torch_vi_only/more_sliding/model.pt"
BASELINE_ARCHIVE="checkpoints/torch_vi_only/more_sliding_baseline_r082"
if [[ -f "${BASELINE_CKPT}" && "${FORCE_OVERWRITE:-0}" != "1" ]]; then
    if [[ ! -d "${BASELINE_ARCHIVE}" ]]; then
        echo "Archiving first sequential run -> ${BASELINE_ARCHIVE}" >&2
        cp -a "checkpoints/torch_vi_only/more_sliding" "${BASELINE_ARCHIVE}"
        if [[ -f logs/log_train_vi_only_more_sliding ]]; then
            cp -a logs/log_train_vi_only_more_sliding \
                logs/log_train_vi_only_more_sliding_baseline_r082
        fi
    fi
    echo "Baseline VI-only checkpoint exists: ${BASELINE_CKPT}" >&2
    echo "Archive: ${BASELINE_ARCHIVE}" >&2
    echo "Refusing to overwrite. Use run_torch_vi_only_optimized.cfg for new runs," >&2
    echo "or set FORCE_OVERWRITE=1 to retrain into more_sliding/." >&2
    exit 1
fi

bash scripts/dsi_preempt_wrapper.sh train_vi_only_torch.py run_torch_vi_only.cfg
