# Claude Report - 2026-07-08 - MAX_LR sweep for gpt2_fmha_ddp_sched.cpp

**Date:** 2026-07-08
**Work done for:** Sweeping peak learning rate over the warmup->MultiStepLR schedule
**Workspace:** BluTrain_BestPrecisionEnv/BluTrain
**Files:** gpt2_fmha_ddp_sched.cpp, run_maxlr_sweep.sh

## Changes

1. gpt2_fmha_ddp_sched.cpp (line ~774): MAX_LR was hardcoded to 6e-4f. Added a MAX_LR
   environment-variable override (same pattern as WARMUP_FRAC / START_FACTOR). Default
   behavior unchanged when MAX_LR env is unset. MIN_LR remains MAX_LR * 0.1.

2. run_maxlr_sweep.sh (new): sweeps
   MAX_LR in {1.5e-3, 1.99e-3, 2e-3, 2.4e-3, 2.8e-3, 3e-3, 4e-3, 5e-3, 6e-3}
   Each point runs, with INIT_FROM_BIN exported:
   USE_PACKED_SDPA=1 make run-mpi FILE=gpt2_fmha_ddp_sched.cpp WITH_BLUBLAS=1 NP=2
   Per-point CSV via OUT_CSV -> maxlr_sweep_logs/train_log_maxlr_<lr>.csv, stdout tee'd
   to maxlr_sweep_logs/run_maxlr_<lr>.log. A failed point is reported and the sweep continues.

## Usage

    ./run_maxlr_sweep.sh
