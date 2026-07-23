#!/bin/bash
# Submit Totten sequential pipeline on DSI Slurm:
#   1) shared PINN pretrain
#   2) VI no_sliding  (C=100)   — after pretrain
#   3) VI max_sliding (C=0.001) — after pretrain (parallel with no_sliding)
#
# Usage (from Archive/ on the cluster):
#   bash slurm/submit_totten_pipeline.sh
#   bash slurm/submit_totten_pipeline.sh --qos=protected
#
# Extra sbatch flags are forwarded to every job (e.g. --qos=...).

set -euo pipefail

ARCHIVE_DIR="${ARCHIVE_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${ARCHIVE_DIR}"
mkdir -p logs/slurm logs checkpoints outputs

EXTRA_SBATCH_ARGS=("$@")

echo "Submitting Totten pretrain..."
PRE=$(sbatch --parsable "${EXTRA_SBATCH_ARGS[@]}" slurm/vi_pretrain_totten.sbatch)
echo "  pretrain job: ${PRE}"

echo "Submitting Totten VI no_sliding (afterok:${PRE})..."
VI_NO=$(sbatch --parsable --dependency=afterok:"${PRE}" \
    "${EXTRA_SBATCH_ARGS[@]}" slurm/vi_train_vi_only_totten_no_sliding.sbatch)
echo "  no_sliding job: ${VI_NO}"

echo "Submitting Totten VI max_sliding (afterok:${PRE})..."
VI_MAX=$(sbatch --parsable --dependency=afterok:"${PRE}" \
    "${EXTRA_SBATCH_ARGS[@]}" slurm/vi_train_vi_only_totten_max_sliding.sbatch)
echo "  max_sliding job: ${VI_MAX}"

cat <<EOF

Totten pipeline queued:
  pretrain     ${PRE}   -> checkpoints/torch_pretrain/totten/
  no_sliding   ${VI_NO} -> checkpoints/torch_vi_only/totten/no_sliding/
  max_sliding  ${VI_MAX} -> checkpoints/torch_vi_only/totten/max_sliding/

Monitor:
  squeue -u \$USER
  tail -f logs/slurm/vi_pre_totten_${PRE}.out
EOF
