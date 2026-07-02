#!/usr/bin/env bash
# Run a spin-up notebook headlessly from the terminal (no Jupyter Lab).
#
# Firedrake/PETSc must run under mpiexec. Jupyter Lab spawns a separate kernel
# process that is often *not* under MPI, which can cause segfaults on macOS.
# This script converts the notebook to a .py file, then runs it with mpiexec.
#
# Usage (from anywhere):
#   bash scripts/run_spinup.sh more_sliding
#   bash scripts/run_spinup.sh no_sliding
#
# Optional environment variables:
#   FIREDDRAKE_ENV   Path to firedrake conda env prefix (default: ~/firedrake-env)
#   MPI_RANKS        Number of MPI ranks (default: 1)
#   KEEP_SCRIPT      If set to 1, keep the generated .py in outputs/logs/spinup/
#
# Activate the env first (conda prefix install — there is no bin/activate):
#   conda activate ~/firedrake-env
#
# Sanity check:
#   ~/firedrake-env/bin/mpiexec -n 1 ~/firedrake-env/bin/python -c "import firedrake; print('ok')"
#
# Example:
#   bash scripts/run_spinup.sh more_sliding

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIREDDRAKE_ENV="${FIREDDRAKE_ENV:-$HOME/firedrake-env}"
MPI_RANKS="${MPI_RANKS:-1}"
CASE="${1:-}"

PYTHON="${FIREDDRAKE_ENV}/bin/python"
JUPYTER="${FIREDDRAKE_ENV}/bin/jupyter"
MPIEXEC="${FIREDDRAKE_ENV}/bin/mpiexec"

usage() {
    cat <<EOF
Usage: $(basename "$0") <more_sliding|no_sliding>

Run spinupNewFull-moreSlide.ipynb or spinupNewFull-lessSlide.ipynb via mpiexec.

Environment:
  FIREDDRAKE_ENV  Conda env prefix with firedrake + jupyter (default: ~/firedrake-env)
                  Activate with: conda activate ~/firedrake-env
  MPI_RANKS       MPI process count (default: 1)
  KEEP_SCRIPT     Keep generated runner .py if set to 1

Logs and optional script copy:
  outputs/logs/spinup/
EOF
}

if [[ -z "$CASE" ]]; then
    usage
    exit 1
fi

case "$CASE" in
    more_sliding | more | more-slide | moreSlide)
        NOTEBOOK="${ROOT}/notebooks/spinup/spinupNewFull-moreSlide.ipynb"
        STEM="spinupNewFull-moreSlide"
        ;;
    no_sliding | less | no-slide | lessSlide | no_sliding)
        NOTEBOOK="${ROOT}/notebooks/spinup/spinupNewFull-lessSlide.ipynb"
        STEM="spinupNewFull-lessSlide"
        ;;
    *)
        echo "Unknown case: $CASE" >&2
        usage
        exit 1
        ;;
esac

for bin in "$PYTHON" "$JUPYTER" "$MPIEXEC"; do
    if [[ ! -x "$bin" ]]; then
        echo "Missing executable: $bin" >&2
        echo "Set FIREDDRAKE_ENV to your firedrake virtualenv." >&2
        exit 1
    fi
done

if [[ ! -f "$NOTEBOOK" ]]; then
    echo "Notebook not found: $NOTEBOOK" >&2
    exit 1
fi

LOG_DIR="${ROOT}/outputs/logs/spinup"
mkdir -p "$LOG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LOG_DIR}/${STEM}_${STAMP}"
mkdir -p "$RUN_DIR"

SCRIPT="${RUN_DIR}/${STEM}.py"
LOG="${RUN_DIR}/run.log"

# Headless plotting; avoid GUI backends in terminal runs.
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "Project:   ${ROOT}"
echo "Notebook:  ${NOTEBOOK}"
echo "Python:    ${PYTHON}"
echo "MPI ranks: ${MPI_RANKS}"
echo "Log:       ${LOG}"
echo

echo "Converting notebook to script..."
"$JUPYTER" nbconvert \
    --to script \
    --output-dir "$RUN_DIR" \
    --output "$(basename "${STEM}")" \
    "$NOTEBOOK"

echo "Starting spin-up (this may take hours in production mode)..."
cd "$ROOT"

set -o pipefail
"$MPIEXEC" -n "$MPI_RANKS" "$PYTHON" "$SCRIPT" 2>&1 | tee "$LOG"
STATUS="${PIPESTATUS[0]}"

if [[ "${KEEP_SCRIPT:-0}" != "1" ]]; then
    rm -f "$SCRIPT"
fi

if [[ "$STATUS" -ne 0 ]]; then
    echo "Spin-up failed with exit code ${STATUS}. See ${LOG}." >&2
    exit "$STATUS"
fi

echo "Done. Log: ${LOG}"
if [[ "${KEEP_SCRIPT:-0}" == "1" ]]; then
    echo "Script: ${SCRIPT}"
fi
