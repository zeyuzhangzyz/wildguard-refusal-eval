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

The CPU launcher accepts `LOGISTIC_C` (default `5.0`) and records it in both
provenance and the candidate manifest. Lower `C` means stronger L2 regularization.
The current operational default (`C=5`) was selected from the near-optimal
Train-validation plateau and checked in a clearly labeled post-hoc Test
sensitivity analysis; do not describe that Test comparison as pristine
hyperparameter selection.
