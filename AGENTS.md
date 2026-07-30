# Project Notes

This repository is the standalone evaluation layer only. It accepts already
materialized `(prompt, response)` candidates and must not silently regenerate
answers or infer candidate-selection policy. GPU runs use the versioned
`scripts/run_vllm_shard.sh`, require an explicit confirmation flag, and save
range plus input-hash provenance. Never store Hugging Face tokens or API keys.

The reportable proxy comparison is registered in
`scripts/run_wildguardtest_benchmark_vllm.sh` and
`refine-logs/EXPERIMENT_PLAN.md`. It uses the fixed `p>=0.6970338` proxy
threshold; the threshold-unseen evaluation half is primary, while all-test
numbers are descriptive. Do not launch its GPU mode without a fresh resource
inspection and explicit configuration approval.
