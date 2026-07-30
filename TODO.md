# TODO

## Doing

- [x] Validate editable installation and a fixture `MODE=plan` range.
- [x] Monitor `full_test_f1_20260730` and verify its full-test proxy report,
  exact 1,720-row coverage, provenance, and output metrics.

## Next

- [ ] Run the provenance-recorded Train-calibrated, full-1,720-row WildGuardTest
  proxy report after reporting its CPU configuration. The threshold must be
  selected only on a deterministic WildGuardTrain validation partition.
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
