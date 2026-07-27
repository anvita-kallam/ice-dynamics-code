#!/usr/bin/env bash
# Submit normal MISMIP more_sliding VI + eval/predict with unit-corrected η bounds.
# Reuses existing sequential pretrain if present; otherwise queues pretrain first.
#
# Usage (from Archive/):
#   bash slurm/submit_mismip_more_sliding_pipeline.sh

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs/slurm logs checkpoints outputs

EXTRA_SBATCH_ARGS=("$@")
PRETRAIN_CKPT="checkpoints/torch_pretrain/more_sliding_sequential/model_best.pt"

if [[ -f "${PRETRAIN_CKPT}" ]]; then
    echo "Found pretrain ${PRETRAIN_CKPT} — skipping Stage 1."
    DEP_ARGS=()
else
    echo "Submitting MISMIP more_sliding sequential pretrain..."
    PRE=$(sbatch --parsable "${EXTRA_SBATCH_ARGS[@]}" slurm/vi_pretrain_sequential_more_sliding.sbatch)
    echo "  pretrain job: ${PRE}"
    DEP_ARGS=(--dependency=afterok:"${PRE}")
fi

echo "Submitting MISMIP more_sliding optimized VI..."
VI=$(sbatch --parsable "${DEP_ARGS[@]}" \
    "${EXTRA_SBATCH_ARGS[@]}" slurm/vi_train_vi_only_optimized_more_sliding.sbatch)
echo "  VI job: ${VI}"

echo "Submitting evaluate+predict (afterok:${VI})..."
EVAL=$(sbatch --parsable --dependency=afterok:"${VI}" \
    "${EXTRA_SBATCH_ARGS[@]}" slurm/vi_eval_predict_more_sliding_optimized.sbatch)
echo "  eval/predict job: ${EVAL}"

cat <<EOF

MISMIP more_sliding pipeline queued:
  VI           ${VI}   -> checkpoints/torch_vi_only/more_sliding_optimized/
  eval/predict ${EVAL} -> outputs/figures/vi_only/more_sliding_optimized/

eta bounds: eta_min=1e-3, eta_max=1e4 MPa·yr (converted from Pa·yr 1e3 / 1e10)
EOF
