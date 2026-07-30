# Progress

## 2026-07-29 - Initial standalone package

Status: implementation_complete_not_executed

- Extracted generic JSONL scoring, official template parsing, resumable output,
  system-prompt range sharding, and vLLM serving into this repository.
- Clean editable installation, Python compilation, two parser/shard tests, Bash
  syntax, and a four-record `MODE=plan` fixture all passed. The range `[30,45)`
  selected exactly its two arm IDs, proving the inclusive/exclusive contract.
- Added and validated the reportable WildGuardTest comparator plan. The pinned
  test parquet has 1,720 response-refusal labels (563 refusals), partitioned by
  the historical deterministic hash into 862 calibration and 858
  threshold-unseen evaluation examples. Three unit tests, compilation, Bash
  syntax, and the real-data no-write plan passed.
- Separated the CPU-only TF-IDF F1 report from the optional GPU WildGuard-7B
  comparator. The primary report needs only official WildGuardTest labels and
  the frozen proxy threshold; it does not wait for 7B inference.
- No model inference or GPU server has been launched from this new repository.
