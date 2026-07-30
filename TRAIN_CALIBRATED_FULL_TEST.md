# Train-Only Regularization Selection and Full-Test Sensitivity

## Protocol

1. Deterministically stratify 37,976 labeled WildGuardTrain examples into 80%
   fitting rows and 20% validation rows.
2. Fit the TF-IDF + class-balanced logistic-regression refusal proxy on the
   fitting rows. Search `C` only on the held-out Train validation rows, choose
   the validation F1 operating threshold for each `C`, and round it to one
   reportable decimal.
3. Refit each diagnostic configuration on all WildGuardTrain rows and evaluate
   it on all 1,720 labeled WildGuardTest examples.

The initial regularization sweep cleanly holds out WildGuardTest: it never loads
or inspects Test rows. The subsequent Test rows below are a post-hoc local
sensitivity diagnostic requested to avoid choosing `C` solely from one
validation fold. They must not be presented as a pristine Test-set
hyperparameter-selection protocol.

## Train-only regularization sweep

| Quantity | Value |
|---|---:|
| Train examples | 37,976 |
| Train validation examples | 7,596 |
| Train fitting examples | 30,380 |
| Candidate `C` values | 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100 |
| Validation-optimal `C` | 10 |
| `C=10` Train-validation F1 at `p >= 0.40` | 95.26% |
| Operational `C` | **5** |
| `C=5` Train-validation F1 at `p >= 0.40` | 95.22% |

`C=5` is within 0.04 percentage points of the validation maximum while retaining
twice as much L2 regularization as `C=10`; it is therefore the conservative
near-optimal configuration used as the package default.

## Full-Test regularization sensitivity (post-hoc diagnostic)

All rows use a model refit on all 37,976 WildGuardTrain examples and its own
Train-validation-derived threshold; every selected threshold rounds to
`p >= 0.40`.

| Logistic `C` | Raw validation threshold | Validation F1 | Test Precision | Test Recall | Test F1 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.4403 | 94.60% | 73.42% | 95.20% | 82.91% |
| 2.0 | 0.3567 | 94.99% | 74.27% | 95.38% | 83.51% |
| **5.0** | 0.3820 | 95.22% | **74.97%** | **95.20%** | **83.88%** |
| 10.0 | 0.3804 | **95.26%** | 74.72% | 95.03% | 83.66% |
| 20.0 | 0.4293 | 95.19% | 74.34% | 94.67% | 83.28% |

The Test curve is locally stable but non-monotonic: the weakly regularized
`C=5` setting is best in this diagnostic, whereas both stronger (`C=0.5`) and
weaker (`C=20`) regularization are worse. This supports the conservative `C=5`
near-optimal choice, but the Test comparison remains diagnostic rather than a
claim that `C` was selected on Test.

## Interpretation

The held-out Train-validation split consistently selects the same rounded
threshold (`0.40`), while Test F1 remains around 83--84%. Therefore the
performance gap is not explained solely by in-sample threshold overfit. It
indicates a **WildGuardTrain-to-WildGuardTest probability-calibration or
distribution shift**.

Regularization matters modestly but does not erase the transfer gap, so it
should not be attributed solely to excessive classifier flexibility.

The 25k `p >= 0.70` rate table should remain a sensitivity view, not a
Train-calibrated primary threshold claim. A future improvement would require
source-aware Train validation or a separate calibration dataset representative of
the intended response distribution.

## Provenance

- Train-only sweeps: `outputs/wildguardtrain_proxy_tune/trainval_c_sweep_20260730_rerun/`
  and `outputs/wildguardtrain_proxy_tune/trainval_c_sweep_extended_20260730/`.
- Full-Test outputs: `outputs/wildguardtest_proxy/trainval_c05_full_test_20260730/`,
  `trainval_calibrated_full_test_20260730/`, `trainval_c5_full_test_20260730/`,
  `trainval_c10_full_test_20260730/`, and `trainval_c20_full_test_20260730/`.
- Each output contains `provenance.txt`, input hashes, the candidate manifest,
  and `report/proxy_metrics.json`.
