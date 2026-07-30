# Runs

## 2026-07-30 - full_matrix_sglang_remote_preflight

Status: prepared_not_started

- Task type: `gpu_multi_data_parallel` (planned)
- Target: Base and Instruct Qwen3-0.6B BeaverTails matrices, each 540,000
  existing responses, split by contiguous system-prompt index ranges.
- Planned machine: `amax-77`; current free candidate physical GPUs are 0, 1, 5,
  and 7. GPUs 2/3 remain reserved. `amax-78` has only 87 GiB free on `/data1`,
  no usable `/data2/dxx` directory, and GPUs 0/1 are occupied, so it is not a
  launch target unless that state changes.
- Input source: local immutable JSON matrices
  `qwen3_0.6b_base_beavertails_results.json` (2.4 GiB) and
  `qwen3_0.6b_instruct_beavertails_results.json` (1.8 GiB), plus the 6,000-row
  `beavertails.json` prompt file.
- Transport: planned Hugging Face dataset repository
  `zeyuzy/LLM_safety_update_reward`, using `scripts/run_sync_hf_artifact.sh` and
  standard secure Hugging Face authentication only.
- Status: `amax-77` has Hugging Face cache credentials and client packages, but
  a Hub API request timed out; no artifact, model, candidate, or judgment output
  has been created.

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

## 2026-07-30 - trainval_c05_full_test_20260730

Status: completed

- Task type: `cpu_only`
- Machine: local Windows workstation
- GPU: N/A (`CUDA_VISIBLE_DEVICES` cleared)
- Branch / commit: `main` / `90fe18e`
- Script: `scripts/run_wildguardtest_proxy_eval.sh`
- Input: `F:/code/SafetyLLM/LLM_safety_data/datasets/wildguardmix/train/wildguard_train.parquet` and `.../test/wildguard_test.parquet`
- Output: `outputs/wildguardtest_proxy/trainval_c05_full_test_20260730`
- Log: FastCtx job `j-j3wu3l`, `C:/Users/lingxueyi/.fastctx/jobs/j-j3wu3l/output.log`
- Configuration: deterministic stratified 80/20 Train-validation protocol,
  final full-Train refit, `LOGISTIC_C=0.5` (four times stronger L2 penalty),
  CPU-only one thread.
- Result: raw validation threshold `0.4403`, reported `p>=0.40`, validation F1
  `0.9460`; full-Test F1 `0.8291`, precision `0.7342`, recall `0.9520`.
  This is worse than the matched `C=2.0` full-Test F1 `0.8351`.

## 2026-07-30 - trainval_c_sweep_20260730_rerun and extended sweep

Status: completed

- Task type: `cpu_only`; machine: local Windows workstation; GPU: N/A.
- Script: `scripts/run_wildguardtrain_proxy_tune.sh` at commit `e111f56`.
- Input: `F:/code/SafetyLLM/LLM_safety_data/datasets/wildguardmix/train/wildguard_train.parquet`.
- Output: `outputs/wildguardtrain_proxy_tune/trainval_c_sweep_20260730_rerun/`
  and `outputs/wildguardtrain_proxy_tune/trainval_c_sweep_extended_20260730/`.
- Configuration: deterministic Train 80/20 stratified split, shared sparse
  fitting-fold TF-IDF matrix, candidate `C={0.05,0.1,0.2,0.5,1,2,5,10,20,50,100}`;
  WildGuardTest was not loaded.
- Result: validation peak `C=10`, F1 `0.9526`; `C=5` F1 `0.9522`.

## 2026-07-30 - post-hoc C sensitivity on full WildGuardTest

Status: completed

- Task type: `cpu_only`; machine: local Windows workstation; GPU: N/A.
- Script: `scripts/run_wildguardtest_proxy_eval.sh` at commit `e111f56`.
- Input: full WildGuardTrain and all 1,720 labeled WildGuardTest rows.
- Outputs: `outputs/wildguardtest_proxy/trainval_c5_full_test_20260730/`,
  `trainval_c10_full_test_20260730/`, and `trainval_c20_full_test_20260730/`.
- Result: C=5 / 10 / 20 obtains Test F1 `0.8388 / 0.8366 / 0.8328`, respectively.
  Together with earlier C=0.5 / 2.0 diagnostics, this is a post-hoc sensitivity
  table, not a Test-selected hyperparameter claim. Operational default set to
  C=5 due to its near-optimal Train-validation F1 and stronger L2 than C=10.
