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
- Completed the formal CPU-only full-test report at
  `outputs/wildguardtest_proxy/full_test_f1_20260730`. On all 1,720 official
  labels, fixed-threshold TF-IDF achieves F1 `0.8669`, precision `0.8342`, and
  recall `0.9023`; all candidate IDs, hashes, threshold, and provenance pass.
  This full-test row includes the threshold-calibration half and is descriptive;
  the 858-row threshold-unseen F1 remains `0.8702`.
- No model inference or GPU server has been launched from this new repository.
