2026-03-24 - Context Parallel backward Di correction debug - TensorParallelismBeta / DTensor/gpt2_cp_test/context_parallel/

## Problem
GPT-2 CP training with `np 2` plateaus at loss ~5.8, while `np 1` reaches ~4.8. A ~0.7-1.0 loss gap.

## Root Cause Identified
The softmax backward Di term in `sdpa_backward_op` is incorrect for merged ring attention:
- Autograd computes: `Di = rowsum(weighted_grad * out_i)` (per-step output)
- Correct value: `Di = w_i * D_global` where `D_global = rowsum(grad_local * merged_out)`
- With 1 GPU, `w_0 = 1` and `out_0 == merged_out`, so Di is correct
- With 2+ GPUs, `out_i != merged_out`, causing systematic gradient error in dQ and dK

## What Was Verified
- Rank 0 and rank 1 gradients are identical in 2-GPU run (CP backward all_gather works)
- 2 GPU vs 1 GPU gradients differ (the Di issue)
- `clip_grad_norm_` vs `clip_grad_norm_dtensor_nccl` is NOT the issue (replicated params, same data)
- Standalone test (`test_sdpa_backward.cpp`) verified `sdpa_backward_manual` is bit-exact with autograd backward when given the same Di

## Attempts Made (All Reverted)
1. **External post-correction**: Recomputed softmax with raw ops, computed Di correction, subtracted from autograd grads. Failed because raw softmax didn't match `fused_tril_softmax` numerically, causing training instability.
2. **Full manual backward (`sdpa_backward_manual`)**: Replaced autograd backward entirely with raw ops + injected Di. Verified correct in isolation test. But caused worse training instability (loss spikes to 12.7). Root cause of integration failure not yet identified.

## Current State
All files reverted to original working state (stable ~5.8 plateau). The test file `tests/test_sdpa_backward.cpp` and Makefile target remain for future use.

## Files Touched (All Reverted)
- DTensor/gpt2_cp_test/context_parallel/SDPAOp.h
- DTensor/gpt2_cp_test/context_parallel/ContextParallelBackward.h
- DTensor/gpt2_cp_test/context_parallel/ContextParallel.h

## Files Added
- DTensor/tests/test_sdpa_backward.cpp (standalone SDPA backward comparison test)
- DTensor/Makefile (added test_sdpa_backward target)

## Next Steps
1. Debug why `sdpa_backward_manual` (verified correct in isolation) causes instability when integrated into CP backward. Suspect: tensor lifetime/view issue, or the detach() breaking something in the outer autograd graph.
2. Alternative approach: modify the `fused_tril_softmax_backward_cuda` kernel to accept an external `dot` value override, avoiding the need for a full manual backward.
3. Consider whether the ~0.8 loss gap is acceptable for an initial CP implementation and focus on correctness of the forward path first.