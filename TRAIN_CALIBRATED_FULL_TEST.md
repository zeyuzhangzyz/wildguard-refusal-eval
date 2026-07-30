# Train-Only Threshold Selection and Full-Test Evaluation

## Protocol

1. Deterministically stratify 37,976 labeled WildGuardTrain examples into 80%
   fitting rows and 20% validation rows.
2. Fit the TF-IDF + class-balanced logistic-regression refusal proxy on the
   fitting rows, choose the validation F1-optimal threshold, and round it to one
   reportable decimal.
3. Refit the proxy on all WildGuardTrain rows and evaluate once on all 1,720
   labeled WildGuardTest examples. No Test labels are used in training or
   threshold selection.

This protocol cleanly holds out WildGuardTest. The validation F1 is used only to
choose an operating threshold; the full-Test metric estimates generalization.

## Selected threshold

| Quantity | Value |
|---|---:|
| Train examples | 37,976 |
| Train validation examples | 7,596 |
| Train fitting examples | 30,380 |
| Raw validation F1-optimal threshold | 0.3567 |
| Reported threshold | `p >= 0.40` |
| Held-out Train-validation F1 at raw threshold | 95.11% |
| Held-out Train-validation F1 at `p >= 0.40` | 94.99% |

## Final WildGuardTest result

| n | Threshold | Accuracy | Balanced accuracy | Precision | Recall | F1 | mIoU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,720 | `p >= 0.40` | 87.67% | 89.65% | 74.27% | 95.38% | 83.51% | 76.89% |

Confusion matrix (`[[TN, FP], [FN, TP]]`): `[[971, 186], [26, 537]]`.

## Interpretation

The held-out Train-validation split independently selects the same rounded
threshold (`0.40`) as the earlier in-sample diagnostic, but full-Test F1 remains
83.51%. Therefore this is not explained solely by in-sample threshold overfit.
It indicates a **WildGuardTrain-to-WildGuardTest probability-calibration or
distribution shift**: the training validation set favors a recall-heavy,
permissive operating point that yields 186 false positives on Test.

The 25k `p >= 0.70` rate table should remain a sensitivity view, not a
Train-calibrated primary threshold claim. A future improvement would require
source-aware Train validation or a separate calibration dataset representative of
the intended response distribution; simply increasing or removing L2
regularization would not establish that transfer.

## Provenance

- Run: `outputs/wildguardtest_proxy/trainval_calibrated_full_test_20260730/`.
- Code commit: `72cd879`.
- Output metrics: `report/proxy_metrics.json`.
- Full command and input hashes: `provenance.txt` and
  `candidates/benchmark_manifest.json`.
