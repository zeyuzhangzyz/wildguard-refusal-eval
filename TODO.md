# TODO

## Doing

- [ ] Resolve the `amax-77` Hugging Face network route, upload/download the
  immutable Base and Instruct 540,000-response matrices through the authorized
  dataset repository, then execute the range-sharded official WildGuard SGLang
  evaluation. Use only the Git-versioned generic builder/launcher; preserve
  `[begin,end)` provenance and never place credentials in commands or logs.
- [x] Validate editable installation and a fixture `MODE=plan` range.
- [x] Monitor `full_test_f1_20260730` and verify its full-test proxy report,
  exact 1,720-row coverage, provenance, and output metrics.
- [ ] Run and verify the fixed prompt-disjoint 2,000-unharmful + 2,000-harmful
  QA-pair Train-derived refusal-detection holdout.

## Next

- [ ] Diagnose the Train-to-Test calibration/distribution shift before promoting
  a Train-calibrated proxy F1 comparison or re-aggregating 25k refusal rates.
- [ ] If needed after the proxy report, inspect GPU resources and obtain approval
  for the optional official-WildGuard-7B comparator.
- [ ] Before a GPU run, inspect resources and confirm model path, visible GPUs,
  candidate input, output root, and shard range.

## Blocked

- [ ] Full GPU scoring requires a locally available, access-approved
  `allenai/wildguard` checkpoint.

## Done

- [x] Switched the reportable TF-IDF refusal threshold to the fixed, rounded
  `p>=0.70` convention. Recomputing the saved 1,720 probabilities gives full
  F1 `0.8640` and held-out evaluation F1 `0.8702`.
- [x] Ran the requested direct full-Train threshold selection and full-Test
  evaluation. It selected `p>=0.40` but showed a large Train/Test F1 gap
  (`98.42%` / `83.51%`), documenting why it is retained as a protocol diagnostic
  rather than the primary threshold-selection method.
- [x] Tested stronger L2 regularization (`C=0.5` versus `C=2.0`) under the same
  Train-validation/full-Test protocol. It reduced Test F1 from `0.8351` to
  `0.8291`; retain `C=2.0` for this configuration.
- [x] Ran Train-only regularization sweeps and a post-hoc Test sensitivity check.
  Package default is now `C=5`: validation F1 `95.22%` (within 0.04 points of
  the Train-validation maximum) and diagnostic Test F1 `83.88%`.
- [x] User selected the C=10 operating point. Restored the package default to
  `C=10`; full-Train/full-Test F1 is `83.66%` at `p>=0.40`.
