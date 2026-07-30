# TODO

## Doing

- [x] Validate editable installation and a fixture `MODE=plan` range.

## Next

- [ ] Run the CPU-only fixed-threshold 1,720-row WildGuardTest proxy report after
  reporting its configuration; its 858-row threshold-unseen evaluation split is
  the primary F1 result.
- [ ] If needed after the proxy report, inspect GPU resources and obtain approval
  for the optional official-WildGuard-7B comparator.
- [ ] Before a GPU run, inspect resources and confirm model path, visible GPUs,
  candidate input, output root, and shard range.

## Blocked

- [ ] Full GPU scoring requires a locally available, access-approved
  `allenai/wildguard` checkpoint.
