# 2026-06-16 - PyTorch 2.9.0+cu128 TF32 vs FP32 precision research (FP32 GPT-2 on RTX 3060 sm86) - TensorParallelismBeta/DTensor (research only, no files modified)

Authoritative answers verified against pytorch/pytorch @ v2.9.0 tag source, corroborated by prior session empirical results (Claude Logs lines 67-69).

## Q1 - nn.Linear / matmul / addmm float32 GEMM precision
- 2.9.0 DEFAULT: `torch.backends.cuda.matmul.allow_tf32 == False`. GEMM runs TRUE FP32 (IEEE).
- New API: `torch.backends.cuda.matmul.fp32_precision` default `"ieee"`; `float32_matmul_precision` default `"highest"`. In Context.cpp `allowTF32CuBLAS()` returns `float32Precision("cuda","matmul") == "tf32"`. allow_tf32 is now a legacy alias of fp32_precision, deprecated post-2.9 (reading it emits a deprecation warning).
- PT_TF32=1 (sets allow_tf32=True, == fp32_precision "tf32", == float32_matmul_precision "high"): cuBLAS GEMM switches to TF32 (TF32 inputs, FP32 accumulate). So: default FP32, PT_TF32=1 -> TF32.

## Q2 - SDPA with FP32 q/k/v on sm86
- Flash (SDPBackend.FLASH_ATTENTION): does NOT support fp32 in 2.9.0. Allowed dtypes `{kHalf, kBFloat16}` on SM80+ (`check_dtypes_low_precision`). FP32 is rejected for flash.
- Mem-efficient: allowed `{kHalf, kFloat, kBFloat16}` on SM80+ -> supports fp32. On sm86 with FP32, mem-efficient is the selected fused backend (math is fallback).
- SDPA internal matmul precision is INDEPENDENT of `torch.backends.cuda.matmul.allow_tf32`. The fused attention kernels do not consult cuBLAS allow_tf32. Setting allow_tf32=True does NOT change what SDPA computes. (Matches log 67/68: "PT_TF32 didn't fix bc allow_tf32 != SDPA backend".) The math backend decomposes into ops that CAN pick up matmul precision, but the fused mem-efficient path runs its own fixed-precision (fp32 accumulate) regardless.

## Q3 - Context-parallel ring attention
- `torch.distributed.tensor.experimental._attention._templated_ring_attention` calls the same SDPA op per ring step on local Q/K shards. It does NOT change backend selection or precision vs plain SDPA - each step is an ordinary SDPA call (same sm86 dtype-driven backend choice). It only adds the online-softmax LSE rescale/merge across ring steps. Precision per-step is identical to standalone SDPA.

## Q4 - nn.LayerNorm float32
- Accumulation uses acc_type<float> = float (NOT double) on CUDA for fp32 input; Welford mean/var in fp32. No TF32, no tensor cores (elementwise/reduction kernels, not GEMM).

## Q5 - GELU(tanh), softmax, cross_entropy float32
- All elementwise/reduction CUDA kernels in fp32. No GEMM, no tensor cores, no TF32 involvement.

## Net callouts vs "naive FP32 everywhere"
- By default everything IS true FP32 (Linear GEMM is IEEE fp32 because allow_tf32 defaults False in 2.9).
- The only TF32 entry point is PT_TF32=1, and it ONLY affects cuBLAS GEMM (Linear/matmul), NOT SDPA, LayerNorm, GELU, softmax, or cross_entropy.
- Flash backend never runs here (fp32 unsupported); mem-efficient handles fp32 on sm86.

## Sources
- pytorch/pytorch v2.9.0: torch/backends/cuda/__init__.py, aten/src/ATen/Context.cpp, aten/src/ATen/native/transformers/cuda/sdp_utils.cpp
- PyTorch docs torch.set_float32_matmul_precision; GitHub issues #153195, #161022
- Prior session: ClaudeReport_session_2026-06-05_cp_grad_parity_harness_bug.md; Claude Logs lines 67-69
