# Claude Report - 2026-07-08 - MAX_LR sweep environment fixes

**Date:** 2026-07-08
**Work done for:** Getting the MAX_LR sweep running on blubridge25-MS-7E06 (ported from blubridge-041 setup)
**Workspace:** BluTrain_BestPrecisionEnv/BluTrain
**Files:** gpt2_fmha_ddp_sched.cpp, Makefile, run_maxlr_sweep.sh

## Failures found and fixed (in order)

1. Link failure (all 9 sweep points): prebuilt dist/lib/libdist.so needed GLIBCXX_3.4.31/32
   (GCC 13), machine has g++ 11.4. Fix: make clean-dist && make -C dist all (rebuilt with
   local toolchain).

2. INIT_FROM_BIN pointed at /home/blubridge-041/... (other machine). Fix: run_maxlr_sweep.sh
   now uses $PWD/gpt2_init_44M.bin (exists locally).

3. Crash at startup (filesystem_error): data_root hardcoded to
   /home/blubridge-041/.../data/data in gpt2_fmha_ddp_sched.cpp. Fix: now "data/data"
   (relative), overridable via DATA_ROOT env.

4. CUDA OOM at step 0: both ranks came up as rank 0 on GPU 0. Root cause: Makefile had
   MPICXX := /usr/bin/mpic++ (OpenMPI 4.1.2) while mpirun in PATH and the rebuilt libdist
   are OpenMPI 5.0.9 (~/.local). Binary linked against mismatched libmpi ran as two
   singletons. Fix: Makefile MPICXX := /home/blu-bridge25/.local/bin/mpic++.

## Verification

- Full smoke: MAX_LR=2.8e-3 ran to step 198, ~102k tok/sec, loss 10.91 -> 6.6, ranks on GPU 0/1.
- All 9 sweep LRs smoke-run 75s each: every point reaches step 0 with
  lr(step0) = 0.36 * MAX_LR (START_FACTOR default), no OOM/errors:
  1.5e-3->5.40e-4, 1.99e-3->7.164e-4, 2e-3->7.20e-4, 2.4e-3->8.64e-4, 2.8e-3->1.008e-3,
  3e-3->1.08e-3, 4e-3->1.44e-3, 5e-3->1.80e-3, 6e-3->2.16e-3.

Sweep is ready: ./run_maxlr_sweep.sh
