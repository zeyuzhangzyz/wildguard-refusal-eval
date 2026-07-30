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
CONFIRM_WILDGUARDTEST_PROXY_EVAL="${CONFIRM_WILDGUARDTEST_PROXY_EVAL:-0}"
BUILD_DIR="${RUN_ROOT}/candidates"
REPORT_DIR="${RUN_ROOT}/report"

cd "${REPO_ROOT}"
build_args=(-m wildguard_refusal_eval.benchmark --wildguard-train "${WILDGUARD_TRAIN}" --wildguard-test "${WILDGUARD_TEST}" --output-dir "${BUILD_DIR}")
case "${MODE}" in
  plan) "${PYTHON_BIN}" "${build_args[@]}" --mode plan ;;
  run)
    [[ "${CONFIRM_WILDGUARDTEST_PROXY_EVAL}" == "1" ]] || { echo "Set CONFIRM_WILDGUARDTEST_PROXY_EVAL=1." >&2; exit 3; }
    if [[ ! -f "${BUILD_DIR}/candidates.jsonl" ]]; then
      [[ ! -e "${BUILD_DIR}" ]] || { echo "Incomplete candidate directory: ${BUILD_DIR}" >&2; exit 3; }
      "${PYTHON_BIN}" "${build_args[@]}" --mode build
    fi
    "${PYTHON_BIN}" -m wildguard_refusal_eval.proxy_report --candidates "${BUILD_DIR}/candidates.jsonl" --output-dir "${REPORT_DIR}"
    ;;
  report) "${PYTHON_BIN}" -m wildguard_refusal_eval.proxy_report --candidates "${BUILD_DIR}/candidates.jsonl" --output-dir "${REPORT_DIR}" ;;
  *) echo "Unknown MODE=${MODE}; expected plan|run|report" >&2; exit 2 ;;
esac
