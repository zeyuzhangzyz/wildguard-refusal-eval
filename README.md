# WildGuard Refusal Eval

Small, standalone evaluator for the released `allenai/wildguard` model. It
scores existing `(prompt, response)` JSONL candidates with the released,
complete three-field template and writes resumable parsed results.

It deliberately does **not** generate model responses or build candidates from
a bandit trace. Those are experiment-specific upstream steps. This repository
only requires a candidate JSONL file and the locally available WildGuard model.

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
