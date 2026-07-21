#!/bin/bash
#SBATCH --job-name=eta_bias_collect
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

set -euo pipefail

MANIFEST="${1:?Usage: sbatch vi_only_eta_bias_collect.sbatch MANIFEST}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${SLURM_SUBMIT_DIR:-${HOME}/ice-dynamics/Archive}}"
cd "${ARCHIVE_DIR}"

# afterany intentionally produces a partial table if a trial failed.
python3 scripts/collect_eta_bias_suite.py "${MANIFEST}"
