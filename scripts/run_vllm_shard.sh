#!/usr/bin/env bash
set -euo pipefail

# Example:
# MODE=run CONFIRM_WILDGUARD_RUN=1 INPUT=/data/candidates.jsonl MODEL_PATH=/models/WildGuard-7B \
# CUDA_VISIBLE_DEVICES=4,5 SYSTEM_PROMPT_SHARD_SIZE=15 SYSTEM_PROMPT_SHARD_INDEX=2 \
# bash scripts/run_vllm_shard.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${MODE:-plan}"
INPUT="${INPUT:?Set INPUT to a candidate JSONL file}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_PYTHON="${VLLM_PYTHON:-${PYTHON_BIN}}"
MODEL_PATH="${MODEL_PATH:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-wildguard}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
CONCURRENCY="${CONCURRENCY:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
SYSTEM_PROMPT_SHARD_SIZE="${SYSTEM_PROMPT_SHARD_SIZE:-10}"
SYSTEM_PROMPT_SHARD_INDEX="${SYSTEM_PROMPT_SHARD_INDEX:-0}"
SYSTEM_PROMPT_BEGIN_INDEX="${SYSTEM_PROMPT_BEGIN_INDEX:-}"
SYSTEM_PROMPT_END_INDEX_EXCLUSIVE="${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE:-}"
SYSTEM_PROMPT_UNIVERSE_SIZE="${SYSTEM_PROMPT_UNIVERSE_SIZE:-90}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
CONFIRM_WILDGUARD_RUN="${CONFIRM_WILDGUARD_RUN:-0}"

if [[ -n "${SYSTEM_PROMPT_BEGIN_INDEX}" || -n "${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE}" ]]; then
  [[ -n "${SYSTEM_PROMPT_BEGIN_INDEX}" && -n "${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE}" ]] || { echo "Set both explicit range bounds or neither." >&2; exit 2; }
else
  [[ "${SYSTEM_PROMPT_SHARD_SIZE}" =~ ^[1-9][0-9]*$ && "${SYSTEM_PROMPT_SHARD_INDEX}" =~ ^[0-9]+$ ]] || { echo "Invalid shard size/index." >&2; exit 2; }
  SYSTEM_PROMPT_BEGIN_INDEX=$((SYSTEM_PROMPT_SHARD_INDEX * SYSTEM_PROMPT_SHARD_SIZE))
  SYSTEM_PROMPT_END_INDEX_EXCLUSIVE=$((SYSTEM_PROMPT_BEGIN_INDEX + SYSTEM_PROMPT_SHARD_SIZE))
  (( SYSTEM_PROMPT_END_INDEX_EXCLUSIVE <= SYSTEM_PROMPT_UNIVERSE_SIZE )) || SYSTEM_PROMPT_END_INDEX_EXCLUSIVE="${SYSTEM_PROMPT_UNIVERSE_SIZE}"
fi
[[ "${SYSTEM_PROMPT_BEGIN_INDEX}" =~ ^[0-9]+$ && "${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE}" =~ ^[0-9]+$ ]] || { echo "Range bounds must be integers." >&2; exit 2; }
(( SYSTEM_PROMPT_BEGIN_INDEX < SYSTEM_PROMPT_END_INDEX_EXCLUSIVE && SYSTEM_PROMPT_END_INDEX_EXCLUSIVE <= SYSTEM_PROMPT_UNIVERSE_SIZE )) || { echo "Invalid range [${SYSTEM_PROMPT_BEGIN_INDEX},${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE})." >&2; exit 2; }

SHARD_NAME="system_prompts_$(printf '%03d' "${SYSTEM_PROMPT_BEGIN_INDEX}")_to_$(printf '%03d' "${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE}")"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${RUN_ID}/${SHARD_NAME}}"
judge_args=(-m wildguard_refusal_eval.judge --input "${INPUT}" --output-dir "${OUTPUT_DIR}"
  --server-url "http://${HOST}:${PORT}/v1" --model "${SERVED_MODEL_NAME}" --engine-label vllm
  --concurrency "${CONCURRENCY}" --max-new-tokens "${MAX_NEW_TOKENS}"
  --system-prompt-begin-index "${SYSTEM_PROMPT_BEGIN_INDEX}"
  --system-prompt-end-index-exclusive "${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE}")

write_provenance() {
  mkdir -p "${OUTPUT_DIR}"
  [[ ! -e "${OUTPUT_DIR}/launcher_provenance.txt" ]] || return 0
  cat > "${OUTPUT_DIR}/launcher_provenance.txt" <<EOF
run_id=${RUN_ID}
engine=vllm
cuda_device_order=${CUDA_DEVICE_ORDER}
cuda_visible_devices=${CUDA_VISIBLE_DEVICES}
tensor_parallel_size=${TP_SIZE}
model_path=${MODEL_PATH}
input=${INPUT}
system_prompt_begin_index=${SYSTEM_PROMPT_BEGIN_INDEX}
system_prompt_end_index_exclusive=${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE}
command=${RUN_COMMAND_ORIGINAL}
EOF
}

cd "${REPO_ROOT}"
case "${MODE}" in
  plan)
    "${PYTHON_BIN}" "${judge_args[@]}" --mode plan
    ;;
  status)
    "${PYTHON_BIN}" "${judge_args[@]}" --mode status
    ;;
  run)
    [[ "${CONFIRM_WILDGUARD_RUN}" == "1" ]] || { echo "Set CONFIRM_WILDGUARD_RUN=1 for GPU inference." >&2; exit 3; }
    [[ -n "${MODEL_PATH}" && -n "${CUDA_VISIBLE_DEVICES}" ]] || { echo "run requires MODEL_PATH and CUDA_VISIBLE_DEVICES." >&2; exit 2; }
    IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    GPU_COUNT="${#GPU_IDS[@]}"
    TP_SIZE="${TP_SIZE:-${GPU_COUNT}}"
    [[ "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]] && (( TP_SIZE <= GPU_COUNT )) || { echo "TP_SIZE must be in 1..${GPU_COUNT}." >&2; exit 2; }
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES PYTHONUTF8=1
    export RUN_COMMAND_ORIGINAL="MODE=run RUN_ID=${RUN_ID} INPUT=${INPUT} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} SYSTEM_PROMPT_BEGIN_INDEX=${SYSTEM_PROMPT_BEGIN_INDEX} SYSTEM_PROMPT_END_INDEX_EXCLUSIVE=${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE} bash scripts/run_vllm_shard.sh"
    if [[ -e "${OUTPUT_DIR}" && ! -f "${OUTPUT_DIR}/launcher_provenance.txt" ]]; then
      echo "Output directory exists without this launcher's provenance; refusing to overwrite: ${OUTPUT_DIR}" >&2
      exit 3
    fi
    write_provenance
    "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server \
      --model "${MODEL_PATH}" --served-model-name "${SERVED_MODEL_NAME}" --host "${HOST}" --port "${PORT}" \
      --tensor-parallel-size "${TP_SIZE}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs "${MAX_NUM_SEQS}" --enable-prefix-caching \
      >"${OUTPUT_DIR}/vllm_server.log" 2>&1 &
    server_pid=$!
    trap 'kill "${server_pid}" 2>/dev/null || true' EXIT
    for _ in $(seq 1 300); do curl --silent --fail "http://${HOST}:${PORT}/health" >/dev/null && break; sleep 2; done
    curl --silent --fail "http://${HOST}:${PORT}/health" >/dev/null || { echo "vLLM health check failed; see ${OUTPUT_DIR}/vllm_server.log" >&2; exit 1; }
    "${PYTHON_BIN}" "${judge_args[@]}" --mode run --confirm-full-run 2>&1 | tee "${OUTPUT_DIR}/run.log"
    ;;
  *) echo "Unknown MODE=${MODE}; expected plan|run|status" >&2; exit 2 ;;
esac
