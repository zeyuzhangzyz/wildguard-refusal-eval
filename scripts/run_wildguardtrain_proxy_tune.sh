#!/usr/bin/env bash
set -euo pipefail

# MODE=plan WILDGUARD_TRAIN=/data/train.parquet bash scripts/run_wildguardtrain_proxy_tune.sh
# MODE=run CONFIRM_WILDGUARDTRAIN_PROXY_TUNE=1 WILDGUARD_TRAIN=/data/train.parquet RUN_ID=proxy_c_tune_v1 bash scripts/run_wildguardtrain_proxy_tune.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${MODE:-plan}"
WILDGUARD_TRAIN="${WILDGUARD_TRAIN:?Set WILDGUARD_TRAIN}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/wildguardtrain_proxy_tune}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CPU_THREADS="${CPU_THREADS:-1}"
LOGISTIC_CS="${LOGISTIC_CS:-0.05 0.1 0.2 0.5 1 2 5 10}"
CONFIRM_WILDGUARDTRAIN_PROXY_TUNE="${CONFIRM_WILDGUARDTRAIN_PROXY_TUNE:-0}"

cd "${REPO_ROOT}"
[[ "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]] || { echo "CPU_THREADS must be positive." >&2; exit 2; }
for value in ${LOGISTIC_CS}; do
  [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]] && [[ "${value}" != "0" ]] || { echo "Each LOGISTIC_CS value must be positive." >&2; exit 2; }
done
export OMP_NUM_THREADS="${CPU_THREADS}" OPENBLAS_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}" PYTHONUTF8=1
write_provenance() {
  mkdir -p "${RUN_ROOT}"
  [[ ! -e "${RUN_ROOT}/provenance.txt" ]] || return 0
  cat > "${RUN_ROOT}/provenance.txt" <<EOF
run_id=${RUN_ID}
task_type=cpu_only
python=${PYTHON_BIN}
cpu_threads=${CPU_THREADS}
logistic_cs=${LOGISTIC_CS}
wildguard_train=${WILDGUARD_TRAIN}
wildguard_test=N/A (not loaded during tuning)
git_commit=$(git rev-parse HEAD)
git_dirty=$(test -n "$(git status --porcelain)" && echo true || echo false)
command=${RUN_COMMAND_ORIGINAL}
EOF
}
case "${MODE}" in
  plan) "${PYTHON_BIN}" -m wildguard_refusal_eval.benchmark --mode tune-plan --wildguard-train "${WILDGUARD_TRAIN}" --output-dir "${RUN_ROOT}" --tune-logistic-cs ${LOGISTIC_CS} ;;
  run)
    [[ "${CONFIRM_WILDGUARDTRAIN_PROXY_TUNE}" == "1" ]] || { echo "Set CONFIRM_WILDGUARDTRAIN_PROXY_TUNE=1." >&2; exit 3; }
    [[ ! -e "${RUN_ROOT}" ]] || { echo "Refusing to overwrite ${RUN_ROOT}" >&2; exit 3; }
    export RUN_COMMAND_ORIGINAL="MODE=run RUN_ID=${RUN_ID} CPU_THREADS=${CPU_THREADS} LOGISTIC_CS='${LOGISTIC_CS}' WILDGUARD_TRAIN=${WILDGUARD_TRAIN} bash scripts/run_wildguardtrain_proxy_tune.sh"
    write_provenance
    "${PYTHON_BIN}" -m wildguard_refusal_eval.benchmark --mode tune --wildguard-train "${WILDGUARD_TRAIN}" --output-dir "${RUN_ROOT}" --tune-logistic-cs ${LOGISTIC_CS}
    ;;
  *) echo "Unknown MODE=${MODE}; expected plan|run" >&2; exit 2 ;;
esac
