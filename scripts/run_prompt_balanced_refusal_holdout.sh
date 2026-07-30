#!/usr/bin/env bash
set -euo pipefail

# MODE=plan WILDGUARD_TRAIN=/data/train.parquet bash scripts/run_prompt_balanced_refusal_holdout.sh
# MODE=run CONFIRM_PROMPT_BALANCED_HOLDOUT=1 WILDGUARD_TRAIN=/data/train.parquet RUN_ID=prompt_balanced_v1 bash scripts/run_prompt_balanced_refusal_holdout.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${MODE:-plan}"
WILDGUARD_TRAIN="${WILDGUARD_TRAIN:?Set WILDGUARD_TRAIN}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/prompt_balanced_refusal_holdout}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CPU_THREADS="${CPU_THREADS:-1}"
LOGISTIC_C="${LOGISTIC_C:-10.0}"
PAIRS_PER_HARM="${PAIRS_PER_HARM:-1000}"
CONFIRM_PROMPT_BALANCED_HOLDOUT="${CONFIRM_PROMPT_BALANCED_HOLDOUT:-0}"

cd "${REPO_ROOT}"
[[ "${CPU_THREADS}" =~ ^[1-9][0-9]*$ ]] || { echo "CPU_THREADS must be positive." >&2; exit 2; }
[[ "${PAIRS_PER_HARM}" =~ ^[1-9][0-9]*$ ]] || { echo "PAIRS_PER_HARM must be positive." >&2; exit 2; }
[[ "${LOGISTIC_C}" =~ ^[0-9]+([.][0-9]+)?$ ]] && [[ "${LOGISTIC_C}" != "0" ]] || { echo "LOGISTIC_C must be positive." >&2; exit 2; }
export OMP_NUM_THREADS="${CPU_THREADS}" OPENBLAS_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}" PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=""
args=(-m wildguard_refusal_eval.prompt_balanced_benchmark --wildguard-train "${WILDGUARD_TRAIN}" --output-dir "${RUN_ROOT}/report" --pairs-per-harm "${PAIRS_PER_HARM}" --logistic-c "${LOGISTIC_C}")

write_provenance() {
  mkdir -p "${RUN_ROOT}"
  [[ ! -e "${RUN_ROOT}/provenance.txt" ]] || return 0
  cat > "${RUN_ROOT}/provenance.txt" <<EOF
run_id=${RUN_ID}
task_type=cpu_only
python=${PYTHON_BIN}
cpu_threads=${CPU_THREADS}
logistic_c=${LOGISTIC_C}
pairs_per_harm=${PAIRS_PER_HARM}
wildguard_train=${WILDGUARD_TRAIN}
git_commit=$(git rev-parse HEAD)
git_dirty=$(test -n "$(git status --porcelain)" && echo true || echo false)
command=${RUN_COMMAND_ORIGINAL}
EOF
}

case "${MODE}" in
  plan) "${PYTHON_BIN}" "${args[@]}" --mode plan ;;
  run)
    [[ "${CONFIRM_PROMPT_BALANCED_HOLDOUT}" == "1" ]] || { echo "Set CONFIRM_PROMPT_BALANCED_HOLDOUT=1." >&2; exit 3; }
    export RUN_COMMAND_ORIGINAL="MODE=run RUN_ID=${RUN_ID} CPU_THREADS=${CPU_THREADS} LOGISTIC_C=${LOGISTIC_C} PAIRS_PER_HARM=${PAIRS_PER_HARM} WILDGUARD_TRAIN=${WILDGUARD_TRAIN} bash scripts/run_prompt_balanced_refusal_holdout.sh"
    write_provenance
    "${PYTHON_BIN}" "${args[@]}" --mode run
    ;;
  *) echo "Unknown MODE=${MODE}; expected plan|run" >&2; exit 2 ;;
esac
