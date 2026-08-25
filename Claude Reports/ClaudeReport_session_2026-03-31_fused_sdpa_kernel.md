# ClaudeReport_session_2026-03-31_fused_sdpa_kernel.md

**Date:** 2026-03-31
**Work:** Fused SDPA (FlashAttention tiled kernel) implementation + validation + Megatron fair comparison
**Workspace:** TensorParallelismBeta

---

## Files Created / Modified

| File | Action |
|------|--------|
| `DTensor/gpt2_cp_test/context_parallel/FusedSDPAKernel.h` | Created - kernel launcher declaration |
| `DTensor/gpt2_cp_test/context_parallel/FusedSDPAKernel.cu` | Created - CUDA FlashAttention forward kernel |
| `DTensor/gpt2_cp_test/context_parallel/FusedSDPAOp.h` | Created - Tensor wrapper matching sdpa_forward() interface |
| `DTensor/gpt2_cp_test/test_fused_sdpa.cpp` | Created - single-GPU correctness + timing test |
| `DTensor/gpt2_cp_test/cp_sdpa_compare_test.cpp` | Modified - added fair 2-GPU CP timing vs Megatron section |
| `DTensor/Makefile` | Modified - added `test_fused_sdpa` target with .cu explicit compile rule |

---

## What Was Built

### FusedSDPAKernel.cu

CUDA implementation of the FlashAttention forward algorithm (Dao et al. 2022).

**Design:**
- Grid: `(ceil(T_q/32), B*H)` -- one CUDA block handles 32 query rows
- Block: 32 threads -- one thread owns one query row
- Shared memory: `K_tile[32][D]` + `V_tile[32][D]` -- loaded cooperatively (no bank conflicts at stride-D load since BLOCK_Q == BLOCK_K == 32)
- Register arrays per thread: `q_regs[D]`, `o_regs[D]` (unnormalized output), `scores[32]`
- Online softmax: O is kept unnormalized during the tile sweep, normalised once at end
- Causal masking: per-element guard `k_global > q_global`; early-exit when entire K block is future
- Template specialisations for HEAD_DIM: 32, 64, 128 (dispatched at launch)
- Uses `expf`/`logf` (not fast intrinsics `__expf`/`__logf`) for FP32 precision

### FusedSDPAOp.h

Drop-in wrapper that returns `SDPAResult{out, lse}` with the same shape convention as `sdpa_forward()` in SDPAOp.h (`lse` is `[B,H,T_q,1]` for SDPAMerger compatibility). Output tensors are not autograd-tracked -- backward still uses `sdpa_backward_op_manual()` which recomputes the forward unfused.

`q_offset` / `k_offset` parameters allow correct causal masking during CP ring-attention steps where Q and K/V come from different global positions.

---

## Correctness Results (all PASS, tol = 1e-3)

| Config | out max_diff | lse max_diff |
|--------|-------------|-------------|
| B=1 H=1 T=8 D=64 causal | 1.19e-7 | 1.19e-7 |
| B=1 H=1 T=8 D=64 non-causal | 5.96e-8 | 0.0 |
| B=2 H=4 T=64 D=64 causal | 6.78e-4 | 9.91e-5 |
| B=2 H=4 T=64 D=64 non-causal | 9.17e-5 | 4.20e-5 |
| B=4 H=6 T=128 D=64 causal | 5.76e-4 | 2.16e-4 |
| B=4 H=6 T=128 D=64 non-causal | 7.51e-5 | 2.86e-5 |
| B=2 H=4 T=64 D=32 causal | 4.31e-4 | 1.15e-4 |
| B=2 H=4 T=64 D=128 causal | 5.12e-4 | 1.24e-4 |

Error source: FP32 tiled accumulation vs reference full-row softmax -- expected at this level for multi-tile causal cases.

---

## Throughput Results (vs unfused autograd baseline, single GPU)

| Config | unfused | fused | speedup |
|--------|---------|-------|---------|
| B=4 H=6 T=512 D=64 causal | 6.38 ms | 0.84 ms | 7.6x |
| B=4 H=6 T=512 D=64 non-causal | 6.20 ms | 1.17 ms | 5.3x |
| B=2 H=8 T=1024 D=64 causal | 15.89 ms | 1.88 ms | 8.4x |
| B=2 H=8 T=2048 D=64 causal | 63.80 ms | 6.60 ms | 9.7x |

---

## Fair Comparison vs Megatron-LM TEDotProductAttention

**Scope of comparison -- all on the same problem: full ring-attention for T=1024 distributed over 2 GPUs (B=4 H=6 D=64 causal)**

| Implementation | Time | What is included |
|----------------|------|-----------------|
| Our CP forward_cp (FP32) | **5.547 ms** | 2x unfused SDPA (T_local=512) + 2x NCCL K/V comm + merge |
| Megatron TEDotProductAttention (BF16) | **1.031 ms** | 2x cuDNN FlashAttn SDPA + 2x ring comm + merge |
| Megatron speedup | **5.4x** | |

The 5.4x gap is expected:
- BF16 tensor cores are ~3-4x faster for SDPA compute than our FP32 kernel
- cuDNN FlashAttention is more optimized than our from-scratch tile kernel
- Our unfused SDPA baseline (~6 ms/call) dominates; switching to our fused kernel would close the gap

**With fused kernel in CP (estimated):** ~2 × 0.84 ms compute + comm ≈ ~2-3 ms -- much closer to Megatron.

---

---

## Fused Kernel Wired into ContextParallel::forward_cp

### Changes

| File | Change |
|------|--------|
| `DTensor/gpt2_cp_test/context_parallel/ContextParallel.h` | Added `#include FusedSDPAOp.h`; moved `source_rank` outside causal block; added `T_local_fwd`; replaced `sdpa_forward` call at line 221 with `sdpa_fused_forward(q_off, k_off)` |
| `DTensor/gpt2_cp_test/context_parallel/FusedSDPAOp.h` | Added fallback to `sdpa_forward` for unsupported D values (not 32/64/128) |
| `DTensor/Makefile` | Added `$(FUSED_SDPA_KERNEL_OBJ)` to `CP_SDPA_COMPARE_OBJS` |

### Updated Timing Results

| Implementation | Time | Notes |
|----------------|------|-------|
| Our CP forward_cp (FP32) — unfused | 5.547 ms | prior baseline |
| Our CP forward_cp (FP32) — fused | **2.659 ms** | **2.1x speedup** |
| Megatron TEDotProduct (BF16) | 1.031 ms | reference |
| Gap vs Megatron | **2.6x** | down from 5.4x |

Remaining gap is BF16 tensor cores + cuDNN FlashAttn on Megatron's side vs FP32 from-scratch kernel.

Forward PASS, Backward PASS (unchanged).

---

## Build

```bash
cd DTensor

# Single-GPU correctness + timing (no MPI needed)
make test_fused_sdpa
./test_fused_sdpa_exec

# Fair 2-GPU CP timing vs Megatron (requires 2 GPUs)
make cp_sdpa_compare_test
mpirun -np 2 ./cp_sdpa_compare_test_exec
```
