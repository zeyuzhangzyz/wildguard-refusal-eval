#!/usr/bin/env bash
set -euo pipefail

# Build an isolated serving environment while reusing the host's CUDA-enabled
# torch installation. This is setup only: it never downloads model weights or
# starts an inference server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${MODE:-plan}"
BASE_PYTHON="${BASE_PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv-sglang}"
SGLANG_VERSION="${SGLANG_VERSION:-0.5.10.post1}"
CONFIRM_SGLANG_ENVIRONMENT_SETUP="${CONFIRM_SGLANG_ENVIRONMENT_SETUP:-0}"

case "${MODE}" in
  plan)
    "${BASE_PYTHON}" -c 'import torch; print("python_torch=" + torch.__version__ + " cuda=" + str(torch.version.cuda))'
    printf 'Planned environment: base_python=%s venv=%s sglang=%s\n' \
      "${BASE_PYTHON}" "${VENV_DIR}" "${SGLANG_VERSION}"
    ;;
  run)
    [[ "${CONFIRM_SGLANG_ENVIRONMENT_SETUP}" == "1" ]] || {
      echo "Set CONFIRM_SGLANG_ENVIRONMENT_SETUP=1 for environment setup." >&2; exit 3;
    }
    [[ ! -e "${VENV_DIR}" ]] || { echo "Venv already exists; refusing to modify: ${VENV_DIR}" >&2; exit 3; }
    "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
    "${VENV_DIR}/bin/python" -m pip install --upgrade pip
    "${VENV_DIR}/bin/python" -m pip install "sglang==${SGLANG_VERSION}" "huggingface_hub>=0.24" "openai>=1.30"
    "${VENV_DIR}/bin/python" -m pip install --no-deps -e "${REPO_ROOT}"
    "${VENV_DIR}/bin/python" - <<'PY'
import sglang, torch
from wildguard_refusal_eval import __file__ as evaluator_init
print("sglang=" + sglang.__version__)
print("torch=" + torch.__version__ + " cuda=" + str(torch.version.cuda))
print("evaluator=" + evaluator_init)
PY
    ;;
  *) echo "Unknown MODE=${MODE}; expected plan|run" >&2; exit 2 ;;
esac
