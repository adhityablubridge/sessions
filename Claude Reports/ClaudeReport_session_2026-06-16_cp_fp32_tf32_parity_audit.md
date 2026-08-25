2026-06-16 - Audit: does C++ CP script use FP32/TF32 in the same places as PyTorch CP script - OldPush / gpt2_cp_test.cpp, gpt2_cp_attnstyle_fp32.py

# CP FP32/TF32 Precision-Placement Parity Audit

**Question:** Confirm whether the C++ context-parallel script
(`TensorParallelismBeta/DTensor/gpt2_cp_test/gpt2_cp_test.cpp`) uses FP32 and TF32
in the same places the PyTorch script
(`TensorParallelismBeta/DTensor/Pytorch/gpt2_cp_attnstyle_fp32.py`) does, for
PyTorch 2.9.0+cu128 on an RTX 3060 (Ampere, sm86).

**Verdict:** NO. They match on all non-matmul ops (LayerNorm, GELU, softmax,
cross-entropy, embeddings, online-softmax accumulation — all FP32 in both), but
DIFFER on the two matmul-bearing operations:

## Mismatches

1. Linear GEMMs (c_attn, c_proj, c_fc, lm_head) + backward
   - C++: TF32, `CUBLAS_COMPUTE_32F_FAST_TF32`, HARDCODED (no env gate).
     Sites: Tensor-Implementations/src/Kernels/cuda/matmul/GenMatmul.cu:3532,
     3545, 3379, 3393. Comment GenMatmul.cu:3278 confirms intent.
   - PyTorch: FP32 (IEEE) by default — `allow_tf32` defaults False in 2.9
     (now legacy alias of `fp32_precision="ieee"` / `float32_matmul_precision="highest"`).
     `PT_TF32=1` flips this to TF32.

2. Attention SDPA (QK^T, P·V forward; dQ/dK/dV backward)
   - C++: TF32 WMMA by default (round-to-nearest corrected, +0x1000 ULP).
     Fwd AttentionForward.cu:659,757; Bwd AttentionBackward.cu:409,458,473,498.
     `ATTN_FP32=1` (FusedSDPAOp.h:130) switches to a scalar-FP32 kernel.
   - PyTorch: FP32. Flash backend REJECTS fp32 in 2.9.0
     (`check_dtypes_low_precision` allows only half/bf16 on SM80+); mem-efficient
     backend runs fp32 with fixed FP32 accumulate. SDPA internal matmul precision
     is INDEPENDENT of `allow_tf32` — `PT_TF32=1` does NOT change SDPA.
   - Ring-attention (`_templated_ring_attention`) issues ordinary per-step SDPA;
     no backend/precision change vs plain SDPA.

## Matches (both FP32)
- Online-softmax accumulation (max/sum/exp/LSE): FP32 scalar both sides.
- LayerNorm: C++ dnn/VectorizedLayerNormKernel.cu:69-127 float accum, no TF32;
  PT acc_type=float. (Build uses DTensor/dnn/, per DTensor/Makefile:283-285,687.)
- GELU(tanh), softmax, cross-entropy, embeddings: FP32, no tensor cores, both sides.

## Toggle asymmetry (key takeaway)
- `PT_TF32=1` affects ONLY PT Linear GEMMs (-> TF32). Does NOT touch SDPA.
- `ATTN_FP32=1` affects ONLY C++ attention (-> FP32). Does NOT touch C++ Linear (still TF32).

Therefore:
- Full FP32 parity: PT default + C++ `ATTN_FP32=1` + C++ Linear GEMM edited to
  `CUBLAS_COMPUTE_32F` (code change; no env gate exists). Achievable.
- Full TF32 parity: IMPOSSIBLE — PT SDPA has no TF32 path.

## Cross-reference
Consistent with prior sessions (Claude Logs 64-69): TF32 round-to-nearest backport
had no effect, PT_TF32 had zero effect on SDPA, precision ruled out as divergence
source (structural; cosine ~1.0 after the c_proj square-Linear transpose fix).

## Authoritative build tree
Tensor library: /home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/Tensor-Implementations
(gpt2_cp_test/Makefile TENSOR_LIB_DIR). LayerNorm from DTensor/dnn/. CP attention
kernels from gpt2_cp_test/context_parallel/ (local).
