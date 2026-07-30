#!/usr/bin/env bash
set -euo pipefail

# Score one previously built full-matrix candidate shard using SGLang DP replicas.
# GPU assignment is explicit; begin/end are carried into all evaluator provenance.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${MODE:-plan}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${PYTHON_BIN}}"
INPUT="${INPUT:?Set INPUT to a candidates.jsonl built by run_build_matrix_candidates.sh}"
OUTPUT_DIR="${OUTPUT_DIR:?Set a fresh OUTPUT_DIR for this judgment shard}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local allenai/wildguard checkpoint}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?Set explicit physical GPU ids}"
SYSTEM_PROMPT_BEGIN_INDEX="${SYSTEM_PROMPT_BEGIN_INDEX:?Set inclusive system-prompt begin index}"
SYSTEM_PROMPT_END_INDEX_EXCLUSIVE="${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE:?Set exclusive system-prompt end index}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-wildguard}"
TP_SIZE="${TP_SIZE:-1}"
DP_SIZE="${DP_SIZE:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.82}"
CONCURRENCY="${CONCURRENCY:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
CONFIRM_WILDGUARD_SGLANG_MATRIX_RUN="${CONFIRM_WILDGUARD_SGLANG_MATRIX_RUN:-0}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES PYTHONUTF8=1
cd "${REPO_ROOT}"
judge_args=(-m wildguard_refusal_eval.judge --input "${INPUT}" --output-dir "${OUTPUT_DIR}"
  --server-url "http://${HOST}:${PORT}/v1" --model "${SERVED_MODEL_NAME}" --engine-label sglang
  --concurrency "${CONCURRENCY}" --max-new-tokens "${MAX_NEW_TOKENS}"
  --system-prompt-begin-index "${SYSTEM_PROMPT_BEGIN_INDEX}"
  --system-prompt-end-index-exclusive "${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE}")

case "${MODE}" in
  plan)
    "${PYTHON_BIN}" "${judge_args[@]}" --mode plan
    printf 'Planned SGLang matrix shard: GPUs=%s TP=%s DP=%s range=[%s,%s)\n' \
      "${CUDA_VISIBLE_DEVICES}" "${TP_SIZE}" "${DP_SIZE}" \
      "${SYSTEM_PROMPT_BEGIN_INDEX}" "${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE}"
    ;;
  run)
    [[ "${CONFIRM_WILDGUARD_SGLANG_MATRIX_RUN}" == "1" ]] || {
      echo "Set CONFIRM_WILDGUARD_SGLANG_MATRIX_RUN=1 for GPU inference." >&2; exit 3;
    }
    [[ ! -e "${OUTPUT_DIR}" ]] || { echo "Output exists; refuse overwrite: ${OUTPUT_DIR}" >&2; exit 3; }
    mkdir -p "${OUTPUT_DIR}"
    cat > "${OUTPUT_DIR}/launcher_provenance.txt" <<EOF
task_type=gpu_multi_data_parallel
engine=sglang
cuda_device_order=${CUDA_DEVICE_ORDER}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
tensor_parallel_size=${TP_SIZE}
data_parallel_size=${DP_SIZE}
model_path=${MODEL_PATH}
input=${INPUT}
system_prompt_begin_index=${SYSTEM_PROMPT_BEGIN_INDEX}
system_prompt_end_index_exclusive=${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE}
command=MODE=run CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} SYSTEM_PROMPT_BEGIN_INDEX=${SYSTEM_PROMPT_BEGIN_INDEX} SYSTEM_PROMPT_END_INDEX_EXCLUSIVE=${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE} bash scripts/run_sglang_matrix_shard.sh
EOF
    "${SGLANG_PYTHON}" -m sglang.launch_server --model-path "${MODEL_PATH}" \
      --served-model-name "${SERVED_MODEL_NAME}" --host "${HOST}" --port "${PORT}" \
      --tp-size "${TP_SIZE}" --dp-size "${DP_SIZE}" --mem-fraction-static "${MEM_FRACTION_STATIC}" \
      >"${OUTPUT_DIR}/sglang_server.log" 2>&1 &
    server_pid=$!
    trap 'kill "${server_pid}" 2>/dev/null || true' EXIT
    for _ in $(seq 1 300); do curl --silent --fail "http://${HOST}:${PORT}/health" >/dev/null && break; sleep 2; done
    curl --silent --fail "http://${HOST}:${PORT}/health" >/dev/null || {
      echo "SGLang health check failed; see ${OUTPUT_DIR}/sglang_server.log" >&2; exit 1;
    }
    "${PYTHON_BIN}" "${judge_args[@]}" --mode run --confirm-full-run 2>&1 | tee "${OUTPUT_DIR}/run.log"
    ;;
  status)
    "${PYTHON_BIN}" "${judge_args[@]}" --mode status
    ;;
  *) echo "Unknown MODE=${MODE}; expected plan|run|status" >&2; exit 2 ;;
esac
