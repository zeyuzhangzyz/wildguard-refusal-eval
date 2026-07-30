# Project Notes

This repository is the standalone evaluation layer only. It accepts already
materialized `(prompt, response)` candidates and must not silently regenerate
answers or infer candidate-selection policy. GPU runs use the versioned
`scripts/run_vllm_shard.sh`, require an explicit confirmation flag, and save
range plus input-hash provenance. Never store Hugging Face tokens or API keys.
