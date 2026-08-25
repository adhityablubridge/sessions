# ClaudeReport — 2026-04-01 — Component Optimizations (FusedAdamW, VectorizedLayerNorm, FusedLMHead)
**Workspace:** TensorParallelismBeta / DTensor

---

## Summary

Implemented three new dnn/ optimization kernels + three test files to close the performance gap vs MegatronLM. All tests pass.

---

## Files Created

### Kernels
- `DTensor/dnn/FusedAdamWKernel.h` / `.cu`
- `DTensor/dnn/VectorizedLayerNormKernel.h` / `.cu`
- `DTensor/dnn/FusedLMHeadKernel.h` / `.cu`

### Tests
- `DTensor/gpt2_cp_test/test_fused_adamw.cpp`
- `DTensor/gpt2_cp_test/test_vectorized_layernorm.cpp`
- `DTensor/gpt2_cp_test/test_lm_head_opt.cpp`

### Modified
- `DTensor/Makefile` — added compile rules + test targets for all three components

---

## Component Details

### 1. FusedAdamW (dnn/FusedAdamWKernel.cu)

Fuses FP32 grad unscale (inv_scale * clip_coeff), AdamW m/v update, and parameter update into a single multi-tensor CUDA kernel. Same TensorInfo struct and binary-search offset layout as existing MultiTensorKernels.cu. No changes to Optim.h or MultiTensorKernels.cu.

**Results (50 tensors x 4096 elements):**
- Correctness: param diff=2.76e-07, m diff=9.31e-10, v diff=5.82e-11 vs reference
- Speedup: **7.86x** (0.227ms → 0.029ms per step)

### 2. VectorizedLayerNorm (dnn/VectorizedLayerNormKernel.cu)

Float4 vectorized forward + backward for float32, half2 for fp16, nv_bfloat162 for bf16. Warp-shuffle block reduction with scalar fallback when cols % 4 != 0.

**Critical bug fixed:** In the backward dx kernel, vln_block_reduce uses smem[0..nwarps-1] for warp leaders. Saving total1 to smem[1] was overwritten by warp-1's leader in the second reduce call. Fix: save total1 to smem[32] (beyond all 32 possible warp-leader slots). smem size set to 34*sizeof(float) for backward kernels.

**Results (512x384):**
- f32 fwd max diff: 1.79e-07, bwd dx max diff: 5.96e-08 vs reference
- Speedup: fwd **1.89x**, bwd **1.75x**

### 3. FusedLMHead (dnn/FusedLMHeadKernel.cu)

Uses `cublasSetMathMode(CUBLAS_TF32_TENSOR_OP_MATH)` on a persistent handle to route cublasSgemm through TF32 Tensor Cores on sm_86 (Ampere). FP32 inputs/outputs, no dtype casting. No external deps beyond -lcublas (already in LIBS).

Megatron context: megatronCP.py uses plain nn.Linear (no VocabParallelCrossEntropy). Model is FP32; only Q/K/V are cast to BF16 for TEDotProductAttention.

**Results (BT=2048, n_embd=384, vocab=50304):**
- Forward diff=0.0000, backward diff=0.0000 (TF32 bit-identical to FP32 at this scale)
- Speedup: fwd **1.62x**, bwd **1.38x**

---

## cuBLAS GEMM Convention (row-major)

For row-major C = A @ B^T (forward):
```
cublasSgemm(OP_T, OP_N, vocab_size, BT, n_embd, alpha, weight, n_embd, hidden, n_embd, beta, logits, vocab_size)
```

For backward d_hidden = d_logits @ weight:
```
cublasSgemm(OP_N, OP_N, n_embd, BT, vocab_size, alpha, weight, n_embd, d_logits, vocab_size, beta0, d_hidden, n_embd)
```

For backward d_weight = d_logits^T @ hidden (accumulated, beta=1):
```
cublasSgemm(OP_N, OP_T, n_embd, vocab_size, BT, alpha, hidden, n_embd, d_logits, vocab_size, beta1, d_weight, n_embd)
```

---

## Attention BF16 Path

Deferred. The Megatron gap is dtype-based (Megatron runs BF16 end-to-end). Adding BF16 SDPA while the rest of the model is FP32 would require round-trip casts that likely cancel the benefit.
