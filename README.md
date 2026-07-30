# WildGuard Refusal Eval

Small, standalone evaluator for the released `allenai/wildguard` model. It
scores existing `(prompt, response)` JSONL candidates with the released,
complete three-field template and writes resumable parsed results.

It deliberately does **not** generate model responses or build candidates from
a bandit trace. Those are experiment-specific upstream steps. This repository
only requires a candidate JSONL file and the locally available WildGuard model.

## Complete response matrices

For an already materialized response matrix, build only the contiguous
system-prompt slice that will be scored. The generic builder streams a JSON
array and therefore does not load a multi-GB matrix into memory:

```bash
MODE=run CONFIRM_MATRIX_CANDIDATE_BUILD=1 \
RESPONSES=/data/qwen_base_responses.json PROMPTS=/data/beavertails.json \
BACKBONE=base SYSTEM_PROMPT_BEGIN_INDEX=0 SYSTEM_PROMPT_END_INDEX_EXCLUSIVE=10 \
OUTPUT_DIR=/data/LLM_score/wildguard_matrix/base/candidates/system_prompts_000_to_010 \
bash scripts/run_build_matrix_candidates.sh
```

Then use SGLang with data-parallel replicas. The output records the same
inclusive/exclusive range in its manifest and per-row judgments:

```bash
MODE=run CONFIRM_WILDGUARD_SGLANG_MATRIX_RUN=1 \
INPUT=/data/LLM_score/wildguard_matrix/base/candidates/system_prompts_000_to_010/candidates.jsonl \
OUTPUT_DIR=/data/LLM_score/wildguard_matrix/base/judgments/system_prompts_000_to_010 \
MODEL_PATH=/data/models/WildGuard-7B CUDA_VISIBLE_DEVICES=0,1,5,7 TP_SIZE=1 DP_SIZE=4 \
SYSTEM_PROMPT_BEGIN_INDEX=0 SYSTEM_PROMPT_END_INDEX_EXCLUSIVE=10 \
bash scripts/run_sglang_matrix_shard.sh
```

For large immutable input files, `scripts/run_sync_hf_artifact.sh` uploads or
downloads a specified file through a Hugging Face **dataset** repository. It
accepts no token argument; standard Hugging Face environment/cache credentials
are the only supported authentication path.

## Input contract

Each non-empty JSONL row must contain:

```json
{"example_id":"q0001-a03","prompt":"...","response":"...","arm_id":3}
```

`arm_id` is optional for an unsharded run, but is required when using
system-prompt range sharding. Other fields are preserved only in the input; the
output records `example_id`, optional `query_id`/`arm_id`, raw WildGuard output,
and the three parsed labels.

## New machine: quick start

The full GPU evaluator is intended for Linux with an NVIDIA CUDA GPU; vLLM is
not a supported Windows serving path. Python 3.9+ is supported.

1. Clone and create an environment:

   ```bash
   git clone git@github.com:zeyuzhangzyz/wildguard-refusal-eval.git
   cd wildguard-refusal-eval
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e '.[serve,test]'
   ```

   On a shared server where a CUDA-enabled PyTorch environment is already
   available, use the versioned isolated setup instead; it reuses only that
   environment's site packages and does not alter it:

   ```bash
   MODE=run CONFIRM_SGLANG_ENVIRONMENT_SETUP=1 \
   BASE_PYTHON=/home/amax/miniforge3/envs/cxj/bin/python \
   VENV_DIR=/data1/dxx/LLM_score/envs/wildguard-sglang \
   bash scripts/prepare_sglang_environment.sh
   ```

   On a Windows machine used only for CPU-side planning, set `PYTHONUTF8=1`
   before invoking Python if its site-packages use UTF-8 `.pth` files.

2. Download the gated model after accepting its Hugging Face terms:

   ```bash
   huggingface-cli login
   huggingface-cli download allenai/wildguard --local-dir /models/WildGuard-7B
   ```

3. Place or link your candidate file anywhere, then run one shard. This command
   uses physical GPUs 4 and 5 when `CUDA_DEVICE_ORDER=PCI_BUS_ID` is respected,
   creates `[30,45)`, and uses TP=2 by default:

   ```bash
   MODE=run CONFIRM_WILDGUARD_RUN=1 \
   INPUT=/data/candidates.jsonl MODEL_PATH=/models/WildGuard-7B \
   CUDA_VISIBLE_DEVICES=4,5 SYSTEM_PROMPT_SHARD_SIZE=15 SYSTEM_PROMPT_SHARD_INDEX=2 \
   RUN_ID=wg_25k_vllm bash scripts/run_vllm_shard.sh
   ```

Use `MODE=plan` first to inspect the exact shard without starting a GPU server.
It only needs `INPUT`:

```bash
MODE=plan INPUT=/data/candidates.jsonl \
SYSTEM_PROMPT_SHARD_SIZE=15 SYSTEM_PROMPT_SHARD_INDEX=2 \
bash scripts/run_vllm_shard.sh
```

## Output and resume behavior

For the above range, results are stored under:

```text
outputs/wg_25k_vllm/system_prompts_030_to_045/
  launcher_provenance.txt
  run_manifest.json
  judgments.jsonl
  failed_requests.jsonl
  run.log
  vllm_server.log
```

The manifest and every judgment row record
`system_prompt_begin_index=30` and
`system_prompt_end_index_exclusive=45`. Reusing the same output directory
resumes only missing `example_id`s after checking the candidate-file hash and
range. The launcher refuses to overwrite a completed or incompatible shard.

`end` is intentionally **exclusive**: `[30,45)` means prompt IDs 30 through 44.

## Scope

WildGuard's response-refusal label is binary. It is not a full/partial refusal
taxonomy and should not alone be reported as human-ground-truth over-refusal.

## Reproducible WildGuardTest comparison

The repository also contains a registered, CPU-only benchmark for the frozen
local WildGuardMix TF-IDF/logistic proxy. It uses official
WildGuardTest `response_refusal_label` as ground truth. The proxy is trained on
WildGuardTrain. A deterministic stratified 80/20 Train/validation split selects
the F1-optimal threshold, which is rounded to one reportable decimal. The final
proxy is then refit on all WildGuardTrain rows. All 1,720 labeled WildGuardTest
examples are reserved for the one final report.

Prepare and inspect the benchmark without GPU inference:

```bash
pip install -e '.[serve,test,benchmark]'
MODE=plan WILDGUARD_TRAIN=/data/wildguard_train.parquet \
WILDGUARD_TEST=/data/wildguard_test.parquet \
bash scripts/run_wildguardtest_proxy_eval.sh
```

The CPU report command is:

```bash
MODE=run CONFIRM_WILDGUARDTEST_PROXY_EVAL=1 \
WILDGUARD_TRAIN=/data/wildguard_train.parquet \
WILDGUARD_TEST=/data/wildguard_test.parquet \
RUN_ID=proxy_f1_v1 bash scripts/run_wildguardtest_proxy_eval.sh
```

It writes `report/proxy_report.md`, JSON, and CSV. The optional official
WildGuard-7B comparator remains available through
`scripts/run_wildguardtest_benchmark_vllm.sh`, but it is not required to report
the proxy's held-out F1. See `refine-logs/EXPERIMENT_PLAN.md` for the claim
boundary and split protocol.
