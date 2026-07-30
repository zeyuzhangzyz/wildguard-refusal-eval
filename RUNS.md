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
