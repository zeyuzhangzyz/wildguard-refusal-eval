#!/usr/bin/env bash
set -euo pipefail

# MODE=plan WILDGUARD_TRAIN=/data/train.parquet WILDGUARD_TEST=/data/test.parquet bash scripts/run_wildguardtest_proxy_eval.sh
# MODE=run CONFIRM_WILDGUARDTEST_PROXY_EVAL=1 WILDGUARD_TRAIN=/data/train.parquet WILDGUARD_TEST=/data/test.parquet RUN_ID=proxy_f1_v1 bash scripts/run_wildguardtest_proxy_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${MODE:-plan}"
WILDGUARD_TRAIN="${WILDGUARD_TRAIN:?Set WILDGUARD_TRAIN}"
WILDGUARD_TEST="${WILDGUARD_TEST:?Set WILDGUARD_TEST}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/wildguardtest_proxy}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CPU_THREADS="${CPU_THREADS:-1}"
LOGISTIC_C="${LOGISTIC_C:-2.0}"
CONFIRM_WILDGUARDTEST_PROXY_EVAL="${CONFIRM_WILDGUARDTEST_PROXY_EVAL:-0}"
BUILD_DIR="${RUN_ROOT}/candidates"
REPORT_DIR="${RUN_ROOT}/report"

cd "${REPO_ROOT}"
[[ "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]] || { echo "CPU_THREADS must be positive." >&2; exit 2; }
[[ "${LOGISTIC_C}" =~ ^[0-9]+([.][0-9]+)?$ ]] && [[ "${LOGISTIC_C}" != "0" ]] || { echo "LOGISTIC_C must be positive." >&2; exit 2; }
export OMP_NUM_THREADS="${CPU_THREADS}" OPENBLAS_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}" PYTHONUTF8=1
build_args=(-m wildguard_refusal_eval.benchmark --wildguard-train "${WILDGUARD_TRAIN}" --wildguard-test "${WILDGUARD_TEST}" --output-dir "${BUILD_DIR}" --logistic-c "${LOGISTIC_C}")
write_provenance() {
  mkdir -p "${RUN_ROOT}"
  [[ ! -e "${RUN_ROOT}/provenance.txt" ]] || return 0
  cat > "${RUN_ROOT}/provenance.txt" <<EOF
run_id=${RUN_ID}
task_type=cpu_only
python=${PYTHON_BIN}
cpu_threads=${CPU_THREADS}
logistic_c=${LOGISTIC_C}
wildguard_train=${WILDGUARD_TRAIN}
wildguard_test=${WILDGUARD_TEST}
git_commit=$(git rev-parse HEAD)
git_dirty=$(test -n "$(git status --porcelain)" && echo true || echo false)
command=${RUN_COMMAND_ORIGINAL}
EOF
}
case "${MODE}" in
  plan) "${PYTHON_BIN}" "${build_args[@]}" --mode plan ;;
  run)
    [[ "${CONFIRM_WILDGUARDTEST_PROXY_EVAL}" == "1" ]] || { echo "Set CONFIRM_WILDGUARDTEST_PROXY_EVAL=1." >&2; exit 3; }
    export RUN_COMMAND_ORIGINAL="MODE=run RUN_ID=${RUN_ID} CPU_THREADS=${CPU_THREADS} LOGISTIC_C=${LOGISTIC_C} WILDGUARD_TRAIN=${WILDGUARD_TRAIN} WILDGUARD_TEST=${WILDGUARD_TEST} bash scripts/run_wildguardtest_proxy_eval.sh"
    write_provenance
    if [[ ! -f "${BUILD_DIR}/candidates.jsonl" ]]; then
      [[ ! -e "${BUILD_DIR}" ]] || { echo "Incomplete candidate directory: ${BUILD_DIR}" >&2; exit 3; }
      "${PYTHON_BIN}" "${build_args[@]}" --mode build
    fi
    "${PYTHON_BIN}" -m wildguard_refusal_eval.proxy_report --candidates "${BUILD_DIR}/candidates.jsonl" --output-dir "${REPORT_DIR}"
    ;;
  report) "${PYTHON_BIN}" -m wildguard_refusal_eval.proxy_report --candidates "${BUILD_DIR}/candidates.jsonl" --output-dir "${REPORT_DIR}" ;;
  *) echo "Unknown MODE=${MODE}; expected plan|run|report" >&2; exit 2 ;;
esac
