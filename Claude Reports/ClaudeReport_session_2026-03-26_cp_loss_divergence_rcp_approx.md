# CP Loss Divergence Root Cause Analysis

**Date:** 2026-03-26
**Workspace:** TensorParallelismBeta
**Files:** FusedKernels.cu, SDPAOp.h, SDPAMerger.h, ActivationKernels.cu

## Summary

Investigated why Context Parallel (CP) training loss plateaus at ~6.1-6.2 while the normal run continues decreasing to ~5.2 by step 1099.

## Root Cause

The `rcp.approx.f32` instruction in `fused_tril_softmax_kernel` (FusedKernels.cu:106) creates a numerical inconsistency with the separately computed LSE (log-sum-exp) in SDPAOp.h.

### The Inconsistency

In SDPAOp.h, the CP forward pass computes:
1. **LSE** (lines 73-80): Full-precision ops (reduce_max, exp, reduce_sum, log)
2. **P_local** (line 85): fused_tril_softmax with rcp.approx.f32

The LSE is precise, but P_local's probabilities are scaled by an approximate reciprocal. When the backward pass (line 186-187) computes:

```
weight = exp(lse_diff)       // precise (from precise LSE)
P_global = P_local * weight  // inherits rcp.approx error
```

P_global has a systematic scale error because P_local and LSE were computed with different precision.

### Why Normal Path Is Unaffected

The normal softmax backward is self-consistent: it only uses the output probabilities (which contain the rcp.approx error), and the error cancels out in the gradient formula `grad[i] = out[i] * (grad_out[i] - dot)`.

### Evidence

- Loss divergence begins at step ~500-600 (>5% gap)
- CP loss plateaus at 6.1-6.2 from step 500 onward
- Normal loss continues decreasing steadily
- Pattern is consistent with cumulative systematic gradient bias

## Recommended Fix

Replace `rcp.approx.f32` with full-precision division in FusedKernels.cu line 106:

```cuda
// Before:
float inv_sum = fast_rcp(s_sum);

// After:
float inv_sum = 1.0f / s_sum;
```

Alternative (retains speed with Newton-Raphson refinement):
```cuda
float inv_sum = fast_rcp(s_sum);
inv_sum = inv_sum * (2.0f - s_sum * inv_sum);
```
