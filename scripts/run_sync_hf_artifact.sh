#!/usr/bin/env bash
set -euo pipefail

# Credentials are resolved only by huggingface_hub from its normal secure cache/env.
# Do not pass a token as an argument or commit one to this repository.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACTION="${ACTION:?Set ACTION=upload or ACTION=download}"
REPO_ID="${REPO_ID:-zeyuzy/LLM_safety_update_reward}"
LOCAL_PATH="${LOCAL_PATH:?Set LOCAL_PATH}"
REPO_PATH="${REPO_PATH:?Set REPO_PATH}"
REVISION="${REVISION:-main}"
CONFIRM_HF_ARTIFACT_TRANSFER="${CONFIRM_HF_ARTIFACT_TRANSFER:-0}"

[[ "${CONFIRM_HF_ARTIFACT_TRANSFER}" == "1" ]] || {
  echo "Set CONFIRM_HF_ARTIFACT_TRANSFER=1 for Hugging Face transfer." >&2; exit 3;
}
[[ "${ACTION}" == "upload" || "${ACTION}" == "download" ]] || {
  echo "ACTION must be upload or download." >&2; exit 2;
}
cd "${REPO_ROOT}"
"${PYTHON_BIN}" scripts/sync_hf_artifact.py --action "${ACTION}" --repo-id "${REPO_ID}" \
  --local-path "${LOCAL_PATH}" --repo-path "${REPO_PATH}" --revision "${REVISION}" --confirm
