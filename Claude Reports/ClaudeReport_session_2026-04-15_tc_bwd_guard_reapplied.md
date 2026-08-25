# Claude Report: TC Backward Guard Re-Applied
**Date:** 2026-04-15
**Project:** TensorParallelismBeta / DTensor Context-Parallel GPT-2
**Branch:** `_adhi_`

---

## Problem

Gradient norm exploding identically to 2026-04-04 incident (547K+ norm by step 511). Previous session had identified BOT_HALF TC forward/backward mismatch and applied a fix, but explosion persisted.

## Root Cause

The `if (false)` guard disabling the TC backward kernel in `launch_flash_attn_bwd_f32` (applied 2026-04-04) had been removed in a subsequent session. The TC backward path was live again:

```cpp
// FusedSDPABackwardKernel.cu line 1118 — was:
if (D % 16 == 0 && D <= 64) {
```

For D=64 (head_dim of this model), this triggers `flash_attn_bwd_unified_qparallel_tc` and `flash_attn_bwd_dq_kernel_tc`, which produce wrong gradients.

## Fix

Re-applied the `if (false)` guard:

```cpp
// TC backward disabled: WMMA backward kernel produces wrong gradients (547K norm explosion).
// Scalar FP32 backward below is correct. Do not re-enable until kernel is fixed and validated.
if (false && D % 16 == 0 && D <= 64) {
```

## File Modified

- `DTensor/gpt2_cp_test/context_parallel/FusedSDPABackwardKernel.cu` line 1118

## Next Steps

- Recompile and run training (log 183)
- Expect grad_norm stable past step 200 (matches 55.5k working state)
- BOT_HALF TC forward fix (FusedSDPAOp.h) from this session is also in place — both fixes should be compiled together
