# HD=64 Scalar Bypass + Repro Mismatch Fix

**Date:** 2026-05-08
**Project:** TensorParallelismBeta
**Branch:** `_adhi_`
**Files modified:** `DTensor/gpt2_cp_test/context_parallel/AttentionForward.cu`, `DTensor/gpt2_cp_test/context_parallel/ContextParallel.h`

## Summary

Continued from the HD=64 kernel bug investigation. Confirmed that the scalar fallback path is correct for T_q != T_k (PARTIAL ring step), and added a targeted bypass of the WMMA TC kernel for this configuration. Also fixed a bug in the binary repro instrumentation that was causing false divergence readings.

## What was done

### 1. Scalar bypass added — `AttentionForward.cu`

In `mem_efficient_attn_forward_tc_strided`, just before `launch_fwd_tc_kernel_dispatch`, added:

```cpp
if (hd == 64 && T_q != T_k) {
    ::OwnTensor::cp::launch_fwd_kernel_dispatch(params, hd, grid_y);
    return;
}
```

This routes HD=64 PARTIAL ring steps (T_q=256, T_k=512) to the scalar kernel, bypassing the WMMA TF32 kernel that produces wrong output.

### 2. Repro instrumentation fix — `ContextParallel.h`

**Root cause of false divergence:** The binary dump code wrote Q/K/V only on the first occurrence (`_binary_dumped` gate) but wrote `kernel_repro_out_cpp.bin` on EVERY training step at (rank_==0, g_block_idx==0, ring step i==1). After the first training step, the output bin was from step N while Q/K/V were from step 0 — mismatched inputs/output.

**Fix:** Moved the output dump inside the `!_binary_dumped` gate, immediately after the forward call on the first occurrence. Added `continue` to skip the duplicate forward that would otherwise follow.

### 3. Scalar kernel confirmed correct

After fixing the repro:

```
CPP kernel out      abs_sum=6.7332e+03  std=2.1415e-02
PT manual SDPA      abs_sum=6.7332e+03  std=2.1415e-02
PT F.SDPA           abs_sum=6.7332e+03  std=2.1415e-02

CPP vs PT-manual:  rel_norm=2.9506e-07   ← bit-accurate (FP32 noise only)
CPP vs PT-F.SDPA:  rel_norm=2.9957e-07   ← bit-accurate
```

The scalar path matches PT at floating-point noise level.

### 4. CP-LB causal correctness test: all PASS

```
=> PASS  (×4)
```

## What the TC kernel bug is (known, not fixed)

`fused_attn_forward_kernel_tc<64>` (WMMA TF32 path) produces wrong output for HD=64 when T_q != T_k. The bug is somewhere in the TC kernel — candidates are tile indexing in the score GEMM, smem layout at HD=64 boundaries, or P×V WMMA fragment mapping. The bypass avoids it for the PARTIAL ring step which is the only place T_q != T_k occurs in training.

## Status

- HD=64 PARTIAL step: **fixed** via scalar bypass — scalar matches PT at 3e-7.
- WMMA TC kernel bug: **bypassed** (not root-cause fixed). Safe to leave until needed for performance.
- All correctness tests: **PASS**.
- Training curve impact: pending re-run.
