#!/bin/bash
#SBATCH --job-name=vi_train_ms
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# Stage 2 — VI joint training (requires Stage 1 checkpoint model_best).
# Wall time: 12h max on DSI general (24h is rejected). Preemption / wall-clock
# end → USR1 → checkpoint + exit 99 → Slurm requeues; restore is automatic.
# Chain multiple 12h allocations until n_epochs=1000 finishes (~24h total).
#
# Submit after pretrain completes:
#   cd /path/to/ice-dynamics/Archive
#   sbatch slurm/vi_train_more_sliding.sbatch
#
# Or chain with dependency:
#   PRE=$(sbatch --parsable slurm/vi_pretrain_more_sliding.sbatch)
#   sbatch --dependency=afterok:${PRE} slurm/vi_train_more_sliding.sbatch

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
export ARCHIVE_DIR
export ICE_DYNAMICS_ROOT="${ICE_DYNAMICS_ROOT:-$(cd "${ARCHIVE_DIR}/.." && pwd)}"
export TORCH_ENV="${TORCH_ENV:-pytorch}"

cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs checkpoints

if [[ ! -f checkpoints/torch_pretrain/more_sliding/model_best.pt ]]; then
    echo "Missing pretrain checkpoint: checkpoints/torch_pretrain/more_sliding/model_best.pt" >&2
    echo "Run slurm/vi_pretrain_more_sliding.sbatch first." >&2
    exit 1
fi

bash scripts/dsi_preempt_wrapper.sh train_torch.py run_torch.cfg
