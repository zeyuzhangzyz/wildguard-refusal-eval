#!/usr/bin/env bash
set -euo pipefail

# First inspect: MODE=plan WILDGUARD_TRAIN=/data/train.parquet WILDGUARD_TEST=/data/test.parquet bash scripts/run_wildguardtest_benchmark_vllm.sh
# Full GPU run: MODE=run CONFIRM_WILDGUARDTEST_BENCHMARK=1 WILDGUARD_TRAIN=... WILDGUARD_TEST=... MODEL_PATH=... CUDA_VISIBLE_DEVICES=4 bash scripts/run_wildguardtest_benchmark_vllm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${MODE:-plan}"
WILDGUARD_TRAIN="${WILDGUARD_TRAIN:?Set WILDGUARD_TRAIN}"
WILDGUARD_TEST="${WILDGUARD_TEST:?Set WILDGUARD_TEST}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/wildguardtest_benchmark}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_ROOT}/${RUN_ID}}"
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
CONFIRM_WILDGUARDTEST_BENCHMARK="${CONFIRM_WILDGUARDTEST_BENCHMARK:-0}"
BUILD_DIR="${RUN_ROOT}/candidates"
JUDGE_DIR="${RUN_ROOT}/judgments"
REPORT_DIR="${RUN_ROOT}/report"

cd "${REPO_ROOT}"
build_args=(-m wildguard_refusal_eval.benchmark --wildguard-train "${WILDGUARD_TRAIN}" --wildguard-test "${WILDGUARD_TEST}" --output-dir "${BUILD_DIR}")
judge_args=(-m wildguard_refusal_eval.judge --input "${BUILD_DIR}/candidates.jsonl" --output-dir "${JUDGE_DIR}" --server-url "http://${HOST}:${PORT}/v1" --model "${SERVED_MODEL_NAME}" --engine-label vllm)

case "${MODE}" in
  plan) "${PYTHON_BIN}" "${build_args[@]}" --mode plan ;;
  build)
    [[ "${CONFIRM_WILDGUARDTEST_BENCHMARK}" == "1" ]] || { echo "Set CONFIRM_WILDGUARDTEST_BENCHMARK=1." >&2; exit 3; }
    "${PYTHON_BIN}" "${build_args[@]}" --mode build
    ;;
  run)
    [[ "${CONFIRM_WILDGUARDTEST_BENCHMARK}" == "1" ]] || { echo "Set CONFIRM_WILDGUARDTEST_BENCHMARK=1." >&2; exit 3; }
    [[ -n "${MODEL_PATH}" && -n "${CUDA_VISIBLE_DEVICES}" ]] || { echo "run requires MODEL_PATH and CUDA_VISIBLE_DEVICES." >&2; exit 2; }
    if [[ ! -f "${BUILD_DIR}/candidates.jsonl" ]]; then
      [[ ! -e "${BUILD_DIR}" ]] || { echo "Incomplete candidate directory: ${BUILD_DIR}" >&2; exit 3; }
      "${PYTHON_BIN}" "${build_args[@]}" --mode build
    fi
    IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"; GPU_COUNT="${#GPU_IDS[@]}"; TP_SIZE="${TP_SIZE:-${GPU_COUNT}}"
    [[ "${TP_SIZE}" =~ ^[1-9][0-9]*$ ]] && (( TP_SIZE <= GPU_COUNT )) || { echo "Invalid TP_SIZE." >&2; exit 2; }
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES PYTHONUTF8=1
    mkdir -p "${JUDGE_DIR}"
    [[ ! -e "${JUDGE_DIR}/launcher_provenance.txt" ]] || true
    if [[ ! -e "${JUDGE_DIR}/launcher_provenance.txt" ]]; then
      printf 'engine=vllm\ncuda_visible_devices=%s\ntensor_parallel_size=%s\nmodel_path=%s\ninput=%s\n' "${CUDA_VISIBLE_DEVICES}" "${TP_SIZE}" "${MODEL_PATH}" "${BUILD_DIR}/candidates.jsonl" > "${JUDGE_DIR}/launcher_provenance.txt"
    fi
    "${VLLM_PYTHON}" -m vllm.entrypoints.openai.api_server --model "${MODEL_PATH}" --served-model-name "${SERVED_MODEL_NAME}" --host "${HOST}" --port "${PORT}" --tensor-parallel-size "${TP_SIZE}" --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs "${MAX_NUM_SEQS}" --enable-prefix-caching >"${JUDGE_DIR}/vllm_server.log" 2>&1 &
    server_pid=$!; trap 'kill "${server_pid}" 2>/dev/null || true' EXIT
    for _ in $(seq 1 300); do curl --silent --fail "http://${HOST}:${PORT}/health" >/dev/null && break; sleep 2; done
    curl --silent --fail "http://${HOST}:${PORT}/health" >/dev/null || { echo "vLLM health check failed; see ${JUDGE_DIR}/vllm_server.log" >&2; exit 1; }
    "${PYTHON_BIN}" "${judge_args[@]}" --mode run --confirm-full-run 2>&1 | tee "${JUDGE_DIR}/run.log"
    "${PYTHON_BIN}" -m wildguard_refusal_eval.benchmark_report --candidates "${BUILD_DIR}/candidates.jsonl" --judgments "${JUDGE_DIR}/judgments.jsonl" --output-dir "${REPORT_DIR}"
    ;;
  report)
    "${PYTHON_BIN}" -m wildguard_refusal_eval.benchmark_report --candidates "${BUILD_DIR}/candidates.jsonl" --judgments "${JUDGE_DIR}/judgments.jsonl" --output-dir "${REPORT_DIR}"
    ;;
  *) echo "Unknown MODE=${MODE}; expected plan|build|run|report" >&2; exit 2 ;;
esac
