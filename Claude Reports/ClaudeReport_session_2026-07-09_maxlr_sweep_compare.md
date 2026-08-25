# Claude Report - 2026-07-09 - MAX_LR sweep comparison

**Date:** 2026-07-09
**Work done for:** Comparing the completed 9-point MAX_LR sweep (warmup->MultiStepLR) of gpt2_fmha_ddp_sched.cpp
**Workspace:** BluTrain_BestPrecisionEnv/BluTrain
**Files:** compare_sched_fmha.py (ALL-CSV mode), maxlr_sweep_logs/train_log_maxlr_*.csv, compare_maxlr_*.png

## Sweep status

All 9 runs complete, 6767 steps each (44.35M params, 10 tok/param, GLOBAL_BATCH 65536).

## Command

python3 compare_sched_fmha.py --csvs maxlr_sweep_logs/train_log_maxlr_{...}.csv --out-prefix compare_maxlr

Outputs: compare_maxlr_{loss,throughput,grad_norm,lr,hellaswag}_overlap.png,
compare_maxlr_loss_vs_elapsed.png, compare_maxlr_loss_vs_tokens.png

## Final metrics (step 6766)

| MAX_LR  | val loss | hellaswag |
|---------|----------|-----------|
| 1.5e-3  | 4.1185   | 0.2538    |
| 1.99e-3 | 4.1209   | 0.2525    |
| 2e-3    | 4.1309   | 0.2551    |
| 2.4e-3  | 4.1383   | 0.2525    |
| 2.8e-3  | 4.4549   | 0.2481    |
| 3e-3    | 4.2137   | 0.2512    |
| 4e-3    | 4.5678   | 0.2448    |
| 5e-3    | 4.5845   | 0.2465    |
| 6e-3    | 6.1416   | 0.2425    |

Best val loss: 1.5e-3 (flat plateau through ~2.4e-3, degrading above; 6e-3 diverged/unstable).
Compute-to-threshold table also shows 1.5e-3 fastest to every loss level.
