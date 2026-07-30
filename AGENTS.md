# Project Notes

This repository is the standalone evaluation layer only. It accepts already
materialized `(prompt, response)` candidates and must not silently regenerate
answers or infer candidate-selection policy. GPU runs use the versioned
`scripts/run_vllm_shard.sh`, require an explicit confirmation flag, and save
range plus input-hash provenance. Never store Hugging Face tokens or API keys.

The reportable proxy comparison is registered in
`scripts/run_wildguardtest_benchmark_vllm.sh` and
`refine-logs/EXPERIMENT_PLAN.md`. It selects the proxy threshold on a
deterministic stratified held-out WildGuardTrain validation partition, rounds it
to one reportable decimal, then refits on all Train rows and evaluates once on
all labeled WildGuardTest rows. Do not launch its GPU mode without a fresh
resource
inspection and explicit configuration approval.

The CPU launcher accepts `LOGISTIC_C` (default `10.0`) and records it in both
provenance and the candidate manifest. Lower `C` means stronger L2 regularization.
The user-selected operational default is `C=10`; its full-Train/full-Test result
is F1 `0.8366` at `p>=0.40`. The C-sensitivity Test comparison is post-hoc and
must not be described as pristine hyperparameter selection.

`scripts/run_prompt_balanced_refusal_holdout.sh` is a Train-derived diagnostic
with exactly 2,000 `unharmful`-request QA pairs and 2,000 `harmful`-request QA
pairs. It holds out prompts—not merely rows—so held-out requests do not enter
either proxy fitting or threshold validation. It is never an external test.

For complete response matrices, use only the generic streaming builder
`scripts/run_build_matrix_candidates.sh` and SGLang scorer
`scripts/run_sglang_matrix_shard.sh`. They preserve an explicit contiguous
`[begin,end)` system-prompt range. Large immutable matrices may move through a
Hugging Face dataset repository only with `scripts/run_sync_hf_artifact.sh`;
credentials must come from Hugging Face's secure environment/cache and may not
be added to code, commands, logs, or documentation.

For `amax-77`, create the isolated SGLang environment only through
`scripts/prepare_sglang_environment.sh`, with the existing CUDA-enabled `cxj`
Python as `BASE_PYTHON`. It must not modify that shared conda environment.
