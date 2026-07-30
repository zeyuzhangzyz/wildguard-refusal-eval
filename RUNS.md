# Runs

## 2026-07-30 - full_test_f1_20260730

Status: completed

- Task type: `cpu_only`
- Machine: local Windows workstation (amax-77 preflight was rejected because
  its load was approximately 893 on 64 logical CPUs)
- GPU: N/A (`CUDA_VISIBLE_DEVICES` cleared)
- Branch / commit: `main` / `ea6611b`
- Script: `scripts/run_wildguardtest_proxy_eval.sh`
- Input: `F:/code/SafetyLLM/LLM_safety_data/datasets/wildguardmix/train/wildguard_train.parquet` and `.../test/wildguard_test.parquet`
- Output: `outputs/wildguardtest_proxy/full_test_f1_20260730`
- Log: FastCtx job `j-41hg9t`, `C:/Users/lingxueyi/.fastctx/jobs/j-41hg9t/output.log`
- Configuration: fixed TF-IDF threshold `p>=0.6970338`, full 1,720-label
  WildGuardTest descriptive summary, CPU thread count 1, no model/API/GPU use.
- Result: complete-test F1 `0.8669`, precision `0.8342`, recall `0.9023`;
  confusion matrix `[[1056,101],[55,508]]`. The threshold-unseen 858-row
  evaluation split remains the primary result (`F1=0.8702`).

## 2026-07-30 - train_calibrated_full_test_20260730

Status: completed

- Task type: `cpu_only`
- Machine: local Windows workstation
- GPU: N/A (`CUDA_VISIBLE_DEVICES` cleared)
- Branch / commit: `main` / `0d98dd1`
- Script: `scripts/run_wildguardtest_proxy_eval.sh`
- Input: `F:/code/SafetyLLM/LLM_safety_data/datasets/wildguardmix/train/wildguard_train.parquet` and `.../test/wildguard_test.parquet`
- Output: `outputs/wildguardtest_proxy/train_calibrated_full_test_20260730`
- Log: FastCtx job `j-rbshym`, `C:/Users/lingxueyi/.fastctx/jobs/j-rbshym/output.log`
- Configuration: full WildGuardTrain fit; threshold chosen from the fitted
  model's in-sample Train predictions; one CPU thread; no model/API/GPU use.
- Result: raw Train F1 optimum `0.3720`, reported threshold `p>=0.40`; full
  1,720-label WildGuardTest F1 `0.8351`, precision `0.7427`, recall `0.9538`.
  The in-sample Train F1 (`0.9842`) is much higher, so do not use this direct
  in-sample selection protocol as the primary refusal-threshold claim.

## 2026-07-30 - trainval_calibrated_full_test_20260730

Status: completed

- Task type: `cpu_only`
- Machine: local Windows workstation
- GPU: N/A (`CUDA_VISIBLE_DEVICES` cleared)
- Branch / commit: `main` / `72cd879`
- Script: `scripts/run_wildguardtest_proxy_eval.sh`
- Input: `F:/code/SafetyLLM/LLM_safety_data/datasets/wildguardmix/train/wildguard_train.parquet` and `.../test/wildguard_test.parquet`
- Output: `outputs/wildguardtest_proxy/trainval_calibrated_full_test_20260730`
- Log: FastCtx job `j-s8de07`, `C:/Users/lingxueyi/.fastctx/jobs/j-s8de07/output.log`
- Configuration: deterministic stratified 80/20 WildGuardTrain fitting/validation
  split selects the threshold; final TF-IDF proxy refit on all Train rows; one
  CPU thread; no model/API/GPU use.
- Result: validation selects raw `0.3567`, reported `p>=0.40`, with validation
  F1 `0.9499`; full 1,720-label Test F1 `0.8351`, precision `0.7427`, recall
  `0.9538`. The matching `0.40` threshold in Train validation and the earlier
  direct-Train run indicates Train-to-Test calibration/distribution shift.
