# Claude Report - 2026-04-04
## CP Training Grad Norm Explosion Fix

**Time:** 2026-04-04  
**Workspace:** TensorParallelismBeta / DTensor  
**Files Modified:** FusedSDPABackwardKernel.cu, FusedSDPAKernel.cu

---

## Problem
Context Parallel GPT-2 training (gpt2_cp_test) had grad_norm ~547,000 at every step (should be ~3-4). Loss stayed flat, not converging. PyTorch CP reference converged normally.

## Root Cause Analysis

Two bugs found (both fixed):

### Bug 1: Stack Buffer Overflow in Generic CUDA Kernels (Minor)
- **File:** FusedSDPAKernel.cu, FusedSDPABackwardKernel.cu
- **Issue:** `MAX_D = 256` and `MAX_D_BWD = 256` in generic kernel register arrays
- **When triggered:** n_heads=1, n_embd=384 → head_dim=384 > 256
- **Fix:** Increased both to 512, added `cudaFuncSetAttribute` for extended shared memory
- **Impact:** Affects single-head configs only (log14 config with n_heads=1)

### Bug 2: TF32 WMMA TC Backward Kernel Produces Wrong Gradients (Primary/Critical)
- **File:** FusedSDPABackwardKernel.cu — `flash_attn_bwd_unified_qparallel_tc`
- **When triggered:** D % 16 == 0 && D <= 64 && T_q == T_k (hits for D=64, T_local=512)
- **Issue:** The unified Q-parallel TC kernel (recently added in "after fused sdpaop fwd+bwd" commit) produces incorrect backward gradients that cause the 547K norm explosion
- **Root cause of TC bug:** Not fully diagnosed (WMMA indexing or fragment load/store error in the unified kernel); the separate dQ and dKdV TC kernels may be correct but the unified kernel has a bug
- **Fix:** Disabled the TC path entirely with `if (false)`, falling back to the scalar FP32 path
- **Impact:** Backward is ~40% slower (6.4ms vs 4.6ms per ring step) but correct

## Verification
- cp_sdpa_compare_test: FORWARD MATCH (max_diff=6e-8), BACKWARD MATCH (max_diff<1e-3)
- gpt2_cp_test norm at step 0: 3.67 (was 547K)
- gpt2_cp_test loss: 10.904 → 10.808 in 8 steps (converging normally)

## What Was Not Fixed
- The TC backward kernel bug is NOT diagnosed at the WMMA instruction level. The `flash_attn_bwd_unified_qparallel_tc` kernel needs investigation before re-enabling. Possible issues: wrong fragment layout (row_major vs col_major), wrong smem pointer arithmetic with HP padding, atomic accumulation race condition.

## Throughput Impact
- Before (broken TC path): 2062 ms/step (wrong gradients)
- After (scalar fallback): 2338 ms/step (correct gradients)
- Backward: 4.6ms → 6.4ms per ring step

## Config Tested
- B=4, T=1024, n_heads=6, n_embd=384, n_layers=3, world_size=2 (CP)
- head_dim=64, T_local=512
