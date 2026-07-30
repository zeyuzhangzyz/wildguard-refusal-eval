# Full-Train Threshold Selection and Full-Test Evaluation

## Protocol

1. Fit the TF-IDF + class-balanced logistic-regression refusal proxy on all
   37,976 labeled WildGuardTrain examples.
2. Use the fitted model's **in-sample WildGuardTrain probabilities** to choose
   the F1-optimal decision threshold; round it to one reportable decimal.
3. Evaluate once on all 1,720 labeled WildGuardTest examples. No Test labels are
   used in training or threshold selection.

This protocol cleanly holds out WildGuardTest, but its threshold-selection F1 is
an in-sample calibration quantity, not a generalization estimate.

## Selected threshold

| Quantity | Value |
|---|---:|
| Train examples | 37,976 |
| Raw Train F1-optimal threshold | 0.3720 |
| Reported threshold | `p >= 0.40` |
| In-sample Train F1 at raw threshold | 98.47% |
| In-sample Train F1 at `p >= 0.40` | 98.42% |

## Final WildGuardTest result

| n | Threshold | Accuracy | Balanced accuracy | Precision | Recall | F1 | mIoU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,720 | `p >= 0.40` | 87.67% | 89.65% | 74.27% | 95.38% | 83.51% | 76.89% |

Confusion matrix (`[[TN, FP], [FN, TP]]`): `[[971, 186], [26, 537]]`.

## Interpretation

The large gap between the 98.42% in-sample Train F1 and 83.51% full-Test F1
shows that selecting the threshold from predictions of the same fitted model is
optimistic. It keeps the Test set clean, but it selects an overly permissive
threshold (`0.40`) and is not an appropriate basis for claiming a high-quality
binary refusal judge or for replacing the existing 25k `p >= 0.70` rate table.

If a Test-independent high-quality operating threshold is required, threshold
selection should instead use held-out or out-of-fold WildGuardTrain predictions;
the final model can still be trained on all WildGuardTrain rows afterward.

## Provenance

- Run: `outputs/wildguardtest_proxy/train_calibrated_full_test_20260730/`.
- Code commit: `0d98dd17263b6794a011ea81420e69adee2f964d`.
- Output metrics: `report/proxy_metrics.json`.
- Full command and input hashes: `provenance.txt` and
  `candidates/benchmark_manifest.json`.
