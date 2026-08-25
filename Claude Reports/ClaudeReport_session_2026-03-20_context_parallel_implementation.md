# Claude Report: Context Parallelism Implementation
**Date:** 2026-03-20
**Workspace:** TensorParallelismBeta
**Files Created:**
- DTensor/gpt2_cp_test/context_parallel/RingRotator.h
- DTensor/gpt2_cp_test/context_parallel/SDPAOp.h
- DTensor/gpt2_cp_test/context_parallel/SDPAMerger.h
- DTensor/gpt2_cp_test/context_parallel/ContextParallel.h
- DTensor/gpt2_cp_test/context_parallel/ContextParallelBackward.h

## Summary

Implemented the full Context Parallelism (Ring Attention) mechanism for the OwnTensor custom framework. This distributes attention computation across multiple ranks by sharding the sequence dimension and rotating K,V buffers in a ring pattern.

## Components Created

### 1. RingRotator.h
Three rotator classes inheriting from `RingRotatorBase`:
- **P2PRingRotator**: Uses `ncclSend`/`ncclRecv` with even/odd deadlock avoidance
- **AlltoAllRingRotator**: Uses `sendrecv_async` for ring shift
- **AllGatherRingRotator**: Single `all_gather` call, then index by iteration

All expose `exchange_buffers(curr_buffer)` and `next_buffer()` interface.

### 2. SDPAOp.h
- `sdpa_forward(q, k, v, is_causal, scale)` -> `SDPAResult{out, lse}`
  - Uses existing `autograd::matmul` and `autograd::fused_tril_softmax`
  - Computes LSE via raw `reduce_max` + `reduce_sum` + `exp` + `log` (not autograd-tracked, only for merger)
- `sdpa_backward_op(q, k, v, grad_output, is_causal, scale)` -> `{grad_q, grad_k, grad_v}`
  - Recomputes forward with autograd, then runs engine backward

### 3. SDPAMerger.h
Online softmax merger using the numerically stable formula:
```
out = out - sigmoid(block_lse - lse) * (out - block_out)
lse = lse - log(sigmoid(lse - block_lse))
```
Uses existing `autograd::sigmoid` and `autograd::log`. Supports f32 conversion for numerical stability.

### 4. ContextParallel.h
`ContextParallel : DModule` with `forward_cp(q, k, v)`:
- Phase 1: Shard Q along seq dim (with HeadTail load balance for Q only)
- Phase 2: Ring loop -- rotate K,V, compute SDPA, merge results
- Phase 3: Get merged output from SDPAMerger
- Phase 4: All-gather output, unshard, undo load balance
- Registers `ContextParallelBackward` as autograd grad_fn on output

Causal masking: skips ring steps where K,V chunk is entirely "future" relative to Q chunk.

### 5. ContextParallelBackward.h
`ContextParallelBackward : Node` with `apply(grads)`:
- Phase 1: Init gradient buffers (grad_q, grad_k_accum, grad_v_accum)
- Phase 2: Ring loop backward -- recompute SDPA backward per saved chunk, accumulate grads
- Phase 3: Communicate dK/dV gradients back to source ranks via sendrecv
- Phase 4: All-gather full gradients, undo load balance on grad_q

## Existing APIs Used (no modifications)
- `autograd::matmul`, `autograd::fused_tril_softmax`, `autograd::sigmoid`, `autograd::log`
- `autograd::transpose`, `autograd::mul`, `autograd::softmax`
- `ProcessGroupNCCL::send_async`, `recieve_async`, `sendrecv_async`, `all_gather`
- `Tensor::make_shards_inplace_axis`, `Tensor::flatten`, `Tensor::flatten_concat`, `Tensor::narrow`, `Tensor::reshape`
- `reduce_max`, `reduce_sum`, `OwnTensor::exp`, `OwnTensor::log`
- `HeadTail` load balancer, `DModule` base class

## No existing files were modified.

## Next Steps
- Integration test: write a test in gpt2_cp_test that creates a multi-rank attention forward+backward pass
- Makefile update: add the new files to the build system
- Performance: fuse the SDPA merger into a single kernel, overlap compute with communication
