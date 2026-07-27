#!/usr/bin/env bash
# Submit in-domain no_sliding sequential pipeline on DSI:
#   1) PINN pretrain
#   2) raised-prior VI (afterok pretrain)
#   3) evaluate + predict (afterok VI)
#
# Usage (from Archive/):
#   bash slurm/submit_no_sliding_raised_prior_pipeline.sh

set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p logs/slurm

EXTRA_SBATCH_ARGS=("$@")
PRETRAIN_CKPT="checkpoints/torch_pretrain/no_sliding_sequential/model_best.pt"

if [[ -f "${PRETRAIN_CKPT}" ]]; then
    echo "Found pretrain ${PRETRAIN_CKPT} — skipping Stage 1."
    DEP_ARGS=()
    PRE="(existing)"
else
    echo "Submitting no_sliding PINN pretrain..."
    PRE=$(sbatch --parsable "${EXTRA_SBATCH_ARGS[@]}" slurm/vi_pretrain_sequential_no_sliding.sbatch)
    echo "  pretrain job: ${PRE}"
    DEP_ARGS=(--dependency=afterok:"${PRE}")
fi

echo "Submitting no_sliding raised-prior VI..."
VI=$(sbatch --parsable "${DEP_ARGS[@]}" \
    "${EXTRA_SBATCH_ARGS[@]}" slurm/vi_train_vi_only_no_sliding_raised_prior.sbatch)
echo "  VI job: ${VI}"

echo "Submitting evaluate+predict (afterok:${VI})..."
EVAL=$(sbatch --parsable --dependency=afterok:"${VI}" \
    "${EXTRA_SBATCH_ARGS[@]}" slurm/vi_eval_predict_no_sliding_raised_prior.sbatch)
echo "  eval/predict job: ${EVAL}"

cat <<EOF

Queued:
  pretrain     ${PRE} -> checkpoints/torch_pretrain/no_sliding_sequential/
  VI           ${VI}  -> checkpoints/torch_vi_only/no_sliding_raised_prior/
  eval/predict ${EVAL} -> outputs/figures/vi_only/no_sliding_raised_prior/

eta bounds: eta_min=1e-3, eta_max=1e4 MPa·yr (converted from Pa·yr 1e3 / 1e10)
EOF
