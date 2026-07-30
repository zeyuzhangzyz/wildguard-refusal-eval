# TODO

## Doing

- [x] Validate editable installation and a fixture `MODE=plan` range.

## Next

- [ ] Run `MODE=plan` for the registered 1,720-row WildGuardTest comparison,
  then inspect GPU resources and obtain approval before `MODE=run`.
- [ ] Before a GPU run, inspect resources and confirm model path, visible GPUs,
  candidate input, output root, and shard range.

## Blocked

- [ ] Full GPU scoring requires a locally available, access-approved
  `allenai/wildguard` checkpoint.
