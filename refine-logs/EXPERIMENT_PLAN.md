# Experiment Plan

**Problem**: determine whether the scalable WildGuardMix-trained TF-IDF refusal
proxy at a Train-calibrated, reportable threshold agrees with, and has comparable refusal-label
accuracy to, the released WildGuard-7B judge.

**Method Thesis**: the proxy is useful only if its held-out binary-refusal
performance is transparently reported against the dataset's official labels;
the 7B comparison is supplementary rather than a prerequisite.

**Date**: 2026-07-29

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: the TF-IDF proxy is a measurable scalable binary proxy | The 25k aggregate currently uses it | Fixed threshold, unseen official-label split, exact confusion matrices | B1 |
| C2: disagreement with 7B is characterized if needed | A proxy cannot be called semantic ground truth | Optional agreement/discordance table, with scope limit | B2 |

## Experiment Blocks

### B1: Train-calibrated full WildGuardTest proxy benchmark
- Claim tested: C1.
- Dataset / split: all 1,720 WildGuardTest rows with a response-refusal label;
  train TF-IDF only on WildGuardTrain. The final full-Train fit's in-sample
  predictions select the F1-optimal threshold, rounded to one reportable decimal.
  All 1,720 WildGuardTest rows are then evaluated exactly once.
- Compared systems: fixed-threshold TF-IDF proxy against official dataset labels.
- Metrics: accuracy, balanced accuracy, precision, recall, F1, mIoU, and the
  complete confusion matrix.
- Setup: CPU-only re-fit on WildGuardTrain; no 7B server or response generation.
- Success criterion: a complete 1,720-row report with exact ID coverage and a
  clear binary-refusal scope label.
- Failure interpretation: a weak held-out result limits the proxy to candidate
  gating/sensitivity rather than primary aggregate scoring.
- Table target: appendix calibration table; one concise scope sentence in rebuttal.
- Priority: MUST-RUN.

### B2: Optional official WildGuard-7B comparison
- Claim tested: C2.
- Why: characterize proxy/7B disagreements, not establish the proxy F1.
- Dataset / split: all 1,720 rows, primary 858-row split emphasized.
- Metrics: same as B1 plus paired bootstrap differences.
- Priority: NICE-TO-HAVE; needs a separate approved GPU run.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | Verify data/split | `MODE=plan` | exactly 1,720 labels; Train-only threshold protocol | CPU seconds | source dataset drift |
| M1 | Primary proxy report | CPU `MODE=run` | 1,720-row final-test F1 report | CPU minutes | train dependency mismatch |
| M2 | Optional official comparator | vLLM `MODE=run` | 1,720 parsed 7B outputs | one 7B GPU pass | gated weights/GPU availability |
| M3 | Optional comparison report | automatic + `MODE=report` | all IDs and labels join exactly | CPU seconds | no C/P/F interpretation |

## Compute and Data Budget

- Primary report: CPU only. Optional 7B comparator: one 24GB GPU is sufficient
  for bf16 7B; multi-GPU tensor parallelism is optional.
- Data: local `wildguard_train.parquet` and `wildguard_test.parquet` only.
- Biggest bottleneck: official WildGuard 7B server startup and per-response
  prefill, not TF-IDF training.

## Risks and Mitigations

- Threshold protocol: choose the threshold exclusively from full WildGuardTrain
  in-sample predictions; never use WildGuardTest for calibration.
- Judge scope: report binary refusal only, never full/partial refusal or human
  over-refusal.
- Failure recovery: JSONL judgments resume by example ID after manifest/hash
  compatibility checks.
