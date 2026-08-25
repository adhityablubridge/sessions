# CP Attention Kernel Divergence at HD=64 with T_q != T_k

**Date:** 2026-05-08
**Project:** TensorParallelismBeta
**Branch:** `_adhi_`
**Files investigated:** `DTensor/gpt2_cp_test/context_parallel/AttentionForward.cu`, `ContextParallel.h`, `FusedSDPAOp.h`

## Symptom

After F1–F4 forward refactor + dK/dV ring fix, training loss curve still overlaps with the previous run instead of closing the gap to PyTorch CP. Activation checksums show ~11% std-rel divergence at `b0_attn_sdpa_merged` (post-attention), compounding through MLP and subsequent blocks.

## Root cause (forward direction)

`fused_attn_forward_kernel_tc` at **HD=64** with **T_q != T_k** (PARTIAL ring step) produces output that is **uncorrelated with PyTorch's SDPA reference** for the same Q/K/V inputs.

Verified via direct binary repro (saved Q/K/V from training step 1, ran PyTorch `F.scaled_dot_product_attention` and manual SDPA on the same tensors, compared against our kernel's actual output `kernel_repro_out_cpp.bin`):

```
shape:  Q=(4, 6, 256, 64)  K=V=(4, 6, 512, 64)
PT manual SDPA:    abs_sum = 6.7332e+03   std = 2.1415e-02
PT F.SDPA:         abs_sum = 6.7332e+03   std = 2.1415e-02   (matches manual at 3.7e-7)
CPP kernel out:    abs_sum = 7.2743e+03   std = 2.3282e-02

Diff CPP vs PT-manual:
  abs_sum ratio CPP/PT = 1.080
  rel_norm             = 0.987   ← essentially orthogonal
```

Per-(b,h) breakdown shows **uniform divergence** across all 24 (b, h) pairs (rel_norm 0.7–1.4). Magnitudes are ~right, values are scrambled.

LSE matches PT at ~1e-3 (TF32 noise), so softmax denominator is correct. But P·V output is wrong despite matching probabilities — kernel must be reading/writing data with wrong indexing somewhere.

## Why the correctness test ("cp_lb_causal_correctness_test") passes at 1.984e-06

Both the CP-LB ring path AND the single-shot reference inside the test use the **same kernel**. They agree with each other; neither validates against PT. The test's pass is internal-consistency only — it does not catch any bug shared by both code paths.

## Why this didn't show up before

The bug shape requires **HD=64 + T_q != T_k**. The earlier correctness test config used HD=128 (default GPT-2). The training config at [gpt2_cp_test.cpp:622-624](DTensor/gpt2_cp_test/gpt2_cp_test.cpp#L622-L624) overrides to `n_embd=384`, `n_heads=6` → HD=64. Training PARTIAL ring step is the only place this kernel path is exercised against PT.

Even after the test was reconfigured to HD=64 (same B=4, H=6, T=1024, D=64 as training), it still passes — because, again, the test's own reference uses our kernel.

## Stride context (relevant but not the actual bug)

Training step 0 sees non-contiguous Q (BTHD storage, BHTD logical view) — `q_sM=384`, `q_sH=64` — falling to the **scalar fallback** load path. Training step 1 PARTIAL has `q_halves[1].contiguous()` so Q is contig and uses the vectorized cp.async path.

The repro that diverges from PT was step 1 (vectorized), so the scalar-vs-vectorized-path question is orthogonal to the kernel bug.

## What I inspected (no smoking gun on paper)

For HD=64:
- `SCORE_K_TILES = 8`, `PV_K_TILES = 4`, `PV_N_TILES = 4`, `PV_TOTAL = 8`, `PV_PASSES = 1`
- 8 warps active in P×V single-pass
- WMMA fragment dims: M=16, N=16, K=8 (TF32)
- smem layout: 5 × [BQ × HD_PAD] + [BQ × BK] + 2×BQ = 11968 floats (47872 bytes) — within 48KB default

Score GEMM tile decomposition, smem load patterns, PV WMMA tile decomposition, and writeback all check out by inspection.

## Repro artifacts (kept for follow-up)

- [kernel_repro_q.bin](../../../TP/TensorParallelismBeta/DTensor/kernel_repro_q.bin), `_k.bin`, `_v.bin`, `_out_cpp.bin` — exact training step 1 inputs/outputs
- [kernel_repro_compare.py](../../../TP/TensorParallelismBeta/DTensor/kernel_repro_compare.py) — basic CPP-vs-PT
- [kernel_repro_compare2.py](../../../TP/TensorParallelismBeta/DTensor/kernel_repro_compare2.py) — manual-SDPA + PT.SDPA + layout sanity
- [kernel_repro_compare3.py](../../../TP/TensorParallelismBeta/DTensor/kernel_repro_compare3.py) — per-(b,h) correlation matrix
- [kernel_repro_compare4.py](../../../TP/TensorParallelismBeta/DTensor/kernel_repro_compare4.py) — per-(b,h) magnitude/rel-norm

## Next steps (recommended order)

1. **Force scalar fallback at HD=64** — make `mem_efficient_attn_forward_tc_strided` always dispatch to `launch_fwd_kernel_dispatch` for HD=64, rerun the kernel-repro compare. If scalar matches PT → confirms WMMA TF32 path is the buggy one.
2. **Write a standalone C++ harness** that loads `kernel_repro_*.bin`, calls our kernel, compares against a CPU reference. With `cuda-memcheck` and printf in suspect lines, pinpoint where data goes sideways.
3. **Bisect the WMMA path** — disable Phase A (use scalar score) but keep Phase B WMMA, or vice versa. Whichever phase, when scalar-substituted, makes output match PT identifies the buggy GEMM.
4. **Compare against `Tensor-Implementations/.../AttentionForward.cu` directly** — the TI kernel is the source of our copy. Diff our cp/ kernel vs TI kernel; any subtle line that differs is a candidate.

## Status

- Forward F1–F4 refactor: **shipped**, semantically identical, validated.
- Backward dK/dV ring accumulation: **shipped**, fixed (from 2.7e-3 FAIL to 1.6e-7 PASS).
- HD=64 + T_q != T_k kernel bug: **identified, not yet fixed**. Blocks closing PT parity gap in training.
