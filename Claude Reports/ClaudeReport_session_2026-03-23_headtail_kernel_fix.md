2026-03-23 - HeadTail permutation rewrite and CP correctness fixes - TensorParallelismBeta

## Summary

Diagnosed and fixed the root cause of loss divergence when using load balancing in context parallel training.

## Root Cause

The `HeadTail::loadbalance` function used a reshape+transpose permutation that produced a stride interleave pattern `[0,2,4,...,1022,1,3,5,...,1023]` instead of the correct head-tail pattern `[0,1023,1,1022,2,1021,...]`. This meant rank 0 got ALL even positions and rank 1 got ALL odd positions -- no actual load balancing for causal attention.

## Changes Made

### 1. New CUDA kernel for HeadTail permutation
- **tensor/headtail_kernel.cu** -- Two kernels: `headtail_loadbalance_kernel` and `headtail_unloadbalance_kernel`
- **tensor/headtail_kernel.cuh** -- Header with launcher declarations
- Permutation: `output[2k] = input[k]`, `output[2k+1] = input[T-1-k]`
- One thread per element, operates on arbitrary [outer, T, inner] layout

### 2. Updated dtensor.cpp
- `HeadTail::loadbalance` now calls `launch_headtail_loadbalance` kernel
- `HeadTail::unloadbalance` now calls `launch_headtail_unloadbalance` kernel
- Removed old reshape+transpose logic

### 3. Backward correctness fixes (from previous session, verified)
- **ContextParallelBackward.h**: `.clone()` on saved tensors to break view aliasing
- **ContextParallelBackward.h**: `Node(3)` instead of `Node(1)` for q/k/v edges
- **ContextParallel.h**: Added `set_next_edge` for k (edge 1) and v (edge 2)

### 4. Makefile
- Added `tensor/headtail_kernel.cu` to `CUDA_SRCS`

## Remaining Issue

The causal skip logic (`source_rank > rank -> skip`) still assumes contiguous partitioning. With correct HeadTail, each chunk has positions spanning the full range, so no chunk is "purely future". The cross-chunk SDPA needs a custom causal mask (not simple tril), which the current `fused_tril_softmax` doesn't support.

## Files Modified
- DTensor/tensor/headtail_kernel.cu (NEW)
- DTensor/tensor/headtail_kernel.cuh (NEW)
- DTensor/tensor/dtensor.cpp
- DTensor/gpt2_cp_test/context_parallel/ContextParallel.h
- DTensor/gpt2_cp_test/context_parallel/ContextParallelBackward.h
- DTensor/Makefile
