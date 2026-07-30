# Experiment Plan

**Problem**: determine whether the scalable WildGuardMix-trained TF-IDF refusal
proxy at its frozen F1 threshold agrees with, and has comparable refusal-label
accuracy to, the released WildGuard-7B judge.

**Method Thesis**: the proxy is useful only if its held-out binary-refusal
performance is transparently reported against known labels and separately
compared with the official 7B model on the exact same examples.

**Date**: 2026-07-29

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: the TF-IDF proxy is a measurable scalable binary proxy | The 25k aggregate currently uses it | Fixed threshold, unseen labeled split, official 7B on same rows, exact confusion matrices | B1 |
| C2: disagreement is characterized rather than hidden | A proxy cannot be called semantic ground truth | Agreement/discordance and binary metric table, with scope limit | B2 |

## Experiment Blocks

### B1: Threshold-unseen WildGuardTest benchmark
- Claim tested: C1.
- Dataset / split: all 1,720 WildGuardTest rows with a response-refusal label;
  train TF-IDF only on WildGuardTrain. Threshold `0.6970338` is frozen from the
  deterministic 862-row calibration half. Primary metrics are only the other
  858-row evaluation half.
- Compared systems: frozen TF-IDF proxy versus released `allenai/wildguard` 7B.
- Metrics: accuracy, balanced accuracy, precision, recall, F1, mIoU, confusion
  matrix; paired bootstrap difference for proxy minus 7B.
- Setup: full released three-field template, temperature 0, 32 output tokens;
  one or more CUDA GPUs through vLLM, no response generation.
- Success criterion: a complete 858-row official-7B output, exact ID coverage,
  strict parsing, and a report that labels this as binary refusal evaluation.
- Failure interpretation: incomplete parse or a large metric gap limits the
  proxy to candidate gating/sensitivity rather than primary aggregate scoring.
- Table target: appendix calibration table; one concise scope sentence in rebuttal.
- Priority: MUST-RUN.

### B2: Calibration and all-test descriptive views
- Claim tested: C2.
- Why: detect split-specific behavior and avoid concealing threshold reuse.
- Dataset / split: calibration half and all 1,720 rows.
- Metrics: same as B1, explicitly marked descriptive.
- Priority: NICE-TO-HAVE; emitted automatically once B1 is complete.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | Verify data/split | `MODE=plan` | exactly 1,720 labels, 862/858 split | CPU seconds | source dataset drift |
| M1 | Materialize proxy/candidates | `MODE=build` | manifest/hash written | CPU minutes | train dependency mismatch |
| M2 | Official comparator | `MODE=run` | 1,720 parsed 7B outputs | one 7B GPU pass | gated weights/GPU availability |
| M3 | Report | automatic + `MODE=report` | all IDs and labels join exactly | CPU seconds | no C/P/F interpretation |

## Compute and Data Budget

- GPU: one 24GB GPU is sufficient for bf16 7B; multi-GPU tensor parallelism is
  optional. The 1,720 request workload is substantially larger than the 32-row
  smoke but far smaller than a 540k response matrix.
- Data: local `wildguard_train.parquet` and `wildguard_test.parquet` only.
- Biggest bottleneck: official WildGuard 7B server startup and per-response
  prefill, not TF-IDF training.

## Risks and Mitigations

- Threshold leakage: primary metrics exclude the calibration half.
- Judge scope: report binary refusal only, never full/partial refusal or human
  over-refusal.
- Failure recovery: JSONL judgments resume by example ID after manifest/hash
  compatibility checks.
