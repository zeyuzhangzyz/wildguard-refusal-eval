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

## 2026-07-30 - Rounded fixed refusal threshold

Status: implementation_complete

- Replaced the historical calibrated `p>=0.6970338` decision boundary with the
  globally fixed, reportable `p>=0.70` boundary in candidate construction and
  both proxy/7B report paths. The report now thresholds stored probabilities at
  report time, avoiding stale thresholded labels.
- Recomputed metrics from the immutable candidate probabilities: full 1,720-row
  F1 is `0.8640` (precision `0.8333`, recall `0.8970`); the deterministic 858-row
  evaluation F1 is unchanged at `0.8702`. The historical `0.6970338` run remains
  recorded unchanged in `RUNS.md`.

## 2026-07-30 - Train-only threshold protocol

Status: implementation_complete_not_executed

- Replaced the prior WildGuardTest calibration/evaluation split. The benchmark
  now reserves all 1,720 labeled WildGuardTest examples for one final report.
- It fits the final proxy on all WildGuardTrain rows, selects the F1-optimal
  threshold from its in-sample Train predictions, rounds it to one reportable
  decimal, and then scores the full test set. Compilation, unit tests, Bash
  syntax, and the real-data `MODE=plan` check pass.
- Formal CPU-only run `train_calibrated_full_test_20260730` completed at commit
  `0d98dd1`: 37,976 full-Train rows selected a raw in-sample F1 threshold of
  `0.3720`, rounded to `0.40`. On all 1,720 WildGuardTest labels, this obtains
  F1 `0.8351`, precision `0.7427`, and recall `0.9538`. The 98.42% in-sample
  Train F1 demonstrates threshold overfitting; this protocol keeps Test clean
  but is not the recommended primary operating-threshold selection procedure.
- User-directed follow-up: restored deterministic stratified Train/validation
  threshold selection for the next primary run. This preserves a Test-only final
  evaluation while avoiding the observed in-sample threshold overfit.
