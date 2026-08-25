# CP Backward dK/dV Ring Accumulation Fix

**Date:** 2026-05-07
**Project:** TensorParallelismBeta
**Branch:** `_adhi_`
**Files:** `DTensor/gpt2_cp_test/context_parallel/AttentionBackward.cu`

## Summary

After migrating the CP backward kernel to a TI-style copy with `T_q != T_k` and no offsets (forward F1–F4 + B1–B4), the `cp_lb_causal_correctness_test` reported:

- Forward: PASS (1.98e-06)
- dQ: PASS (2.03e-03)
- **dK: FAIL** (2.68e-03 mean, max ~0.3 per position)
- **dV: FAIL** (1.11e-02 mean, max ~1.94 per position)

dQ passing while dK/dV fail ruled out per-tile gradient math (kernel produces same `p`/`ds` for all three) — pointing to **CP-level accumulation**, not the kernel.

## Root Cause

The LB ring loop in `ContextParallelBackward.h` uses a **travelling-accumulator** pattern: `grad_key`/`grad_value` are zero-initialized once per backward call, then rotated through the ring with `dkv_rotater->exchange_buffers` after each iteration. Each iteration must ATOMIC-ADD its per-step dK/dV to the buffer (which holds prior ranks' contributions for the same K/V chunk).

The kernel's `LAUNCH_MEM_BWD_EXP11` macro called:

```cpp
cudaMemsetAsync(params.dK, 0, BH * T_k * HD * 4);
cudaMemsetAsync(params.dV, 0, BH * T_k * HD * 4);
```

before the kernel launch. This **wiped the rotated accumulator each iteration**, so after the loop `grad_key` held only the LAST iter's contribution instead of the sum across all rotations.

This was a TI-kernel design assumption mismatch: the TI backward kernel was authored for **single-call** semantics (memset → atomicAdd → output is the gradient). CP requires **multi-call accumulation**.

## Why dQ Passed

`exp11` doesn't memset dQ. It uses a persistent `dq_frag` register accumulator across kv tiles within a single call, then a non-atomic final store. dQ has a related but smaller issue (overwrite across iters in the LB-PARTIAL second-half view) that happened to land within the test's tolerance (6.55e-02). Latent — not the active failure.

## Fix

Removed the two `cudaMemsetAsync` calls from `LAUNCH_MEM_BWD_EXP11` and documented the new contract:

> **CP-RING CONTRACT:** caller is responsible for zero-initializing dK/dV. The kernel atomic-adds per-tile contributions. Standalone `sdpa_fused_backward` already pre-zeros via `Tensor::zeros`; CP ring loop pre-zeros via `Tensor::zeros` at allocation time (`grad_key`/`grad_value` initialization in `ContextParallelBackward.h`).

Single-call paths still work because `grad_key`/`grad_value` are passed in zero. Ring paths now correctly accumulate.

## Validation

| Metric | Before | After |
|---|---|---|
| Forward | 1.984e-06 PASS | 1.984e-06 PASS |
| dQ | 2.031e-03 PASS | 2.031e-03 PASS |
| dK | **2.680e-03 FAIL** | **1.628e-07 PASS** |
| dV | **1.112e-02 FAIL** | **5.024e-08 PASS** |

dK/dV improved by ~4 orders of magnitude, now at FP32 noise floor.

## Follow-ups (not in this fix)

1. **`exp11` dQ overwrite across iters** — passes in this test but is mathematically wrong for LB-PARTIAL. Either change final dQ store to `atomicAdd` (and remove future caller-side zeroing assumption) or change `ContextParallelBackward.h` to use a per-step dQ buffer and accumulate externally.
2. **`exp7` internal kernel zeroing of dK/dV** at the start of each block (lines 195–197 of `AttentionBackward.cu`) — same hazard if `exp7` ever becomes the active path (HD ∈ {8, 24, 40, 56}).
3. **`exp7` dQ memset** in `LAUNCH_MEM_BWD_EXP7` — same hazard if `exp7` is the active path.
