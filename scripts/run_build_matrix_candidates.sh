#!/usr/bin/env bash
set -euo pipefail

# Build one backbone x contiguous system-prompt shard from a pre-generated matrix.
# The resulting JSONL is generic evaluator input and contains no router/bandit logic.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${MODE:-plan}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESPONSES="${RESPONSES:?Set RESPONSES to one immutable response-matrix JSON file}"
PROMPTS="${PROMPTS:?Set PROMPTS to the aligned prompt JSON file}"
BACKBONE="${BACKBONE:?Set BACKBONE (for example base or instruct)}"
SYSTEM_PROMPT_BEGIN_INDEX="${SYSTEM_PROMPT_BEGIN_INDEX:?Set inclusive system-prompt begin index}"
SYSTEM_PROMPT_END_INDEX_EXCLUSIVE="${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE:?Set exclusive system-prompt end index}"
EXPECTED_SYSTEM_PROMPT_COUNT="${EXPECTED_SYSTEM_PROMPT_COUNT:-90}"
RESPONSE_FIELD="${RESPONSE_FIELD:-filtered_response}"
OUTPUT_DIR="${OUTPUT_DIR:?Set a fresh OUTPUT_DIR for this candidate shard}"
CONFIRM_MATRIX_CANDIDATE_BUILD="${CONFIRM_MATRIX_CANDIDATE_BUILD:-0}"

cd "${REPO_ROOT}"
args=(-m wildguard_refusal_eval.matrix_candidates --mode "${MODE}"
  --responses "${RESPONSES}" --prompts "${PROMPTS}" --backbone "${BACKBONE}"
  --response-field "${RESPONSE_FIELD}" --output-dir "${OUTPUT_DIR}"
  --system-prompt-begin-index "${SYSTEM_PROMPT_BEGIN_INDEX}"
  --system-prompt-end-index-exclusive "${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE}"
  --expected-system-prompt-count "${EXPECTED_SYSTEM_PROMPT_COUNT}")

if [[ "${MODE}" == "run" ]]; then
  [[ "${CONFIRM_MATRIX_CANDIDATE_BUILD}" == "1" ]] || {
    echo "Set CONFIRM_MATRIX_CANDIDATE_BUILD=1 for a matrix scan." >&2; exit 3;
  }
  args+=(--confirm-build)
elif [[ "${MODE}" != "plan" ]]; then
  echo "Unknown MODE=${MODE}; expected plan|run" >&2; exit 2
fi

export RUN_COMMAND_ORIGINAL="MODE=${MODE} BACKBONE=${BACKBONE} SYSTEM_PROMPT_BEGIN_INDEX=${SYSTEM_PROMPT_BEGIN_INDEX} SYSTEM_PROMPT_END_INDEX_EXCLUSIVE=${SYSTEM_PROMPT_END_INDEX_EXCLUSIVE} bash scripts/run_build_matrix_candidates.sh"
"${PYTHON_BIN}" "${args[@]}"
