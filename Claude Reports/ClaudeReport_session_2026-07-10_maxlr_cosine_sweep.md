# Claude Report - 2026-07-10 - Cosine MAX_LR sweep comparison

**Date:** 2026-07-10
**Work done for:** Comparing the 9-point MAX_LR sweep with warmup->CosineAnnealingLR (MIN_LR = 0.1*MAX_LR) in gpt2_fmha_ddp_sched.cpp
**Workspace:** BluTrain_BestPrecisionEnv/BluTrain
**Files:** maxlr_cosine_sweep_logs/train_log_maxlr_*.csv, compare_maxlr_cosine_*.png

## Final metrics (step 6766, cosine schedule)

| MAX_LR  | val loss | hellaswag |
|---------|----------|-----------|
| 1.5e-3  | 4.1509   | 0.2529    |
| 1.99e-3 | 4.1741   | 0.2563    |
| 2e-3    | 4.1627   | 0.2547    |
| 2.4e-3  | 4.1673   | 0.2557    |
| 2.8e-3  | 4.5275   | 0.2463    |
| 3e-3    | 4.3101   | 0.2536    |
| 4e-3    | 4.3722   | 0.2500    |
| 5e-3    | 4.6523   | 0.2472    |
| 6e-3    | 5.9683   | 0.2423    |

## vs multistep (same MAX_LR, same steps)

Multistep basin (1.5e-3..2.4e-3) final val 4.118-4.138; cosine basin 4.151-4.174 —
multistep ~0.03-0.04 better at every basin point. Same shape: flat basin through
2.4e-3, degradation above, 6e-3 unstable. HellaSwag best: cosine 1.99e-3 (0.2563)
vs multistep 2e-3 (0.2551); differences within eval noise.

Plots: compare_maxlr_cosine_{loss,lr,hellaswag,grad_norm,throughput}_overlap.png,
compare_maxlr_cosine_loss_vs_{elapsed,tokens}.png
