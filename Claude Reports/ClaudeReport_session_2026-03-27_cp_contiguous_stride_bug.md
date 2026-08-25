# Claude Report: Context Parallel contiguous() Stride Bug

**Date:** 2026-03-27
**Workspace:** TensorParallelismBeta
**Files Modified:** DTensor/gpt2_cp_test/context_parallel/ContextParallel.h, DTensor/gpt2_cp_test/test_attn_discrepancy.cpp

## Summary

Fixed a bug in Context Parallel (ring attention) where `.contiguous()` produced corrupt data when called on a shard of a transposed tensor, causing SDPA output to diverge from standard attention by up to 1.82 max error.

## Root Cause

The Q, K, V tensors entering `forward_cp` come from `autograd::transpose(..., 1, 2)`, which swaps dims 1 and 2 **without copying data** -- it only swaps strides. This produces a tensor with shape `[B, H, T, D]` but non-standard strides `[24576, 64, 384, 1]` (stride[1] < stride[2]).

When `make_shards_inplace_axis` creates a view along dim 2 (T), the resulting shard inherits these non-standard strides. The subsequent `.contiguous()` call fails to correctly copy the strided data -- it appears to assume strides are in decreasing order and effectively does a flat memcpy, scrambling data across heads and sequence positions.

**Specifically:** `local_q[b=0, h=1, t=0]` would contain data from the wrong location (neither the correct h=1,t=0 nor a simple flat-offset match), corrupting all SDPA computations beyond h=0's first few positions.

## Fix

In `ContextParallel.h`, added `.contiguous()` on Q, K, V **before** sharding:

```cpp
// BEFORE (buggy):
Tensor q_work = q;           // still non-contiguous (transposed strides)
// ... shard q_work ...
// ... shard.contiguous()    // broken: can't handle non-standard stride order

// AFTER (fixed):
Tensor q_work = q.contiguous();  // forces proper strided copy first
// ... shard q_work ...           // shard now has standard decreasing strides
// ... shard.contiguous()         // works correctly
```

## Diagnostic Approach

1. Added intermediate value captures (LN output, Q/K/V, SDPA output, c_proj) to both StdAttention and CPAttention
2. Layered comparison revealed SDPA output (Layer 2) as the divergence point (max error 1.82)
3. Position-specific prints showed position 0 matched (trivial: only self-attends) but position 31 diverged
4. Key test: sharding a contiguous Q then calling `.contiguous()` gave correct data; sharding a transposed Q then calling `.contiguous()` gave wrong data
5. This confirmed `.contiguous()` cannot handle non-standard stride ordering on shard views

## Results

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| SDPA Max Error | 1.823 | 4.25e-4 |
| Final Max Error | 0.257 | 3.86e-5 |
| Final Mean Error | 0.0158 | 1.79e-6 |

Remaining error is expected FP precision from decomposed ring attention vs single-pass SDPA.

## Affected Files

- **DTensor/gpt2_cp_test/context_parallel/ContextParallel.h** -- Fix applied (lines 110-112)
- **DTensor/gpt2_cp_test/gpt2_cp_test.cpp** -- Production CP test, inherits fix via header
- **DTensor/gpt2_cp_test/test_attn_discrepancy.cpp** -- Diagnostic test, enhanced with layered comparisons

## Note

The underlying `.contiguous()` implementation in the tensor library has a latent bug: it does not correctly handle tensors where strides are not in decreasing order (i.e., transposed views). The fix works around this by ensuring contiguity before the problematic shard+contiguous pattern. A proper fix to the tensor library's `contiguous()` would make this workaround unnecessary.
