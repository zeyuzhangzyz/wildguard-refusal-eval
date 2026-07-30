#!/usr/bin/env bash
set -euo pipefail

# Downloads weights only. Hugging Face credentials must be supplied by its
# standard secure cache/environment; no token is accepted as an argument.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${MODE:-plan}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ID="${MODEL_ID:-allenai/wildguard}"
MODEL_REVISION="${MODEL_REVISION:-main}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/models/WildGuard-7B}"
CONFIRM_WILDGUARD_MODEL_DOWNLOAD="${CONFIRM_WILDGUARD_MODEL_DOWNLOAD:-0}"

cd "${REPO_ROOT}"
args=(scripts/download_wildguard_model.py --mode "${MODE}" --model-id "${MODEL_ID}"
  --revision "${MODEL_REVISION}" --output-dir "${OUTPUT_DIR}")
case "${MODE}" in
  plan|status) ;;
  run)
    [[ "${CONFIRM_WILDGUARD_MODEL_DOWNLOAD}" == "1" ]] || {
      echo "Set CONFIRM_WILDGUARD_MODEL_DOWNLOAD=1 for model download." >&2; exit 3;
    }
    args+=(--confirm-download)
    ;;
  *) echo "Unknown MODE=${MODE}; expected plan|run|status" >&2; exit 2 ;;
esac
export RUN_COMMAND_ORIGINAL="MODE=${MODE} MODEL_ID=${MODEL_ID} OUTPUT_DIR=${OUTPUT_DIR} bash scripts/run_download_wildguard_model.sh"
"${PYTHON_BIN}" "${args[@]}"
