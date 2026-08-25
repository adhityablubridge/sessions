# Claude Report: SDPA Load-Balance Forward+Backward Parity with PyTorch
**Date:** 2026-04-16
**Project:** TensorParallelismBeta / DTensor Context-Parallel GPT-2
**Branch:** `_adhi_`

---

## Summary

Implemented 3-way dispatch for load-balanced causal ring attention in both forward and backward, matching PyTorch's `_templated_ring_attention` and `_templated_ring_attention_backward` protocols. Also replaced the batch alltoall gradient routing with a pipelined `dkv_rotater` protocol.

---

## Changes Made

### Phase 1: SDPAMerger Partial Support
**File:** `DTensor/gpt2_cp_test/context_parallel/SDPAMerger.h`

- Added `bool partial = false` parameter to `step()`
- When `partial=true`: block_out/block_lse arrive as `[B,H,T/2,D]` (pre-sliced by forward)
- Accumulator's 2nd half is extracted via `narrow_view` (view, no copy)
- Merged result written back via `copy_` (in-place, no allocation)
- Replaced `autograd::log` with raw `OwnTensor::log` (no autograd graph node)
- `autograd::sigmoid` retained (no raw alternative available)
- Bug 1 fix: block input is NOT re-sliced; accumulator IS sliced

### Phase 2: Forward 3-Way Dispatch
**File:** `DTensor/gpt2_cp_test/context_parallel/ContextParallel.h`

- Enabled load balance for causal: `bool lb_active = load_balance_` (removed `&& !is_causal_`)
- Added `recompute_k` parameter (default=false, save K at every rotation)
- 3-way dispatch in ring loop matching PyTorch lines 451-471:
  - `i == 0`: full Q, K, V, `is_causal=true`, `partial=false`
  - `i <= rank_` with LB: full Q, `K[:T/2]`, `V[:T/2]`, `is_causal=false`, `partial=false`
  - `i > rank_` with LB: `Q[T/2:]`, full K, V, `is_causal=false`, `partial=true`
- Sub-chunking via `make_shards_inplace_axis(2, seq_dim)` before SDPA call
- No cross-chunk causal offsets needed with LB (sub-chunking handles masking)
- Saved `partial_flags` vector passed to backward

### Phase 3: Backward 3-Way Dispatch + Pipelined dkv_rotater
**File:** `DTensor/gpt2_cp_test/context_parallel/ContextParallelBackward.h`

- Replaced per-step `grad_k_accum[i]` + batch alltoall with pipelined `dkv_rotater`
- `dkv_rotater` uses AlltoAllRingRotator (always, matching PyTorch)
- Travelling accumulator protocol (matches PyTorch lines 588-627):
  - `i==0`: add grad_k/v directly to travelling accumulator
  - `i>0`: receive previous grad_key/value from ring, add current contribution, send onward
  - After loop: one final `next_buffer()` to get completed result
- 3-way backward dispatch mirroring forward:
  - `i <= rank_` with LB: backward with K[:T/2], V[:T/2]; pad dK/dV to full-T before adding
  - `i > rank_` with LB: backward with Q[T/2:], out[T/2:], grad_out[T/2:], lse[T/2:]; dQ added to 2nd half only
- `recompute_k_` flag: when true, uses kv_rotater to recompute K/V rotations in backward
- Bug 2 fix: all slicing uses seq_dim=2 consistently
- Bug 3 fix: partial dK/dV padded to full-T before entering rotater
- Error 4 fix: no double accumulation (dQ local, dK/dV only via rotater)

### Phase 4: TC Backward
- TC backward is already enabled (line 1078 of FusedSDPABackwardKernel.cu)
- No `if(false)` guard present in reverted codebase
- Must monitor grad norms during training; re-apply guard if explosion occurs

### Phase 5: Validation Strategy
Validation requires (not yet implemented):
1. Forward parity: seeded inputs, compare our output vs PyTorch's ring_flash_attention
2. Backward parity: compare dQ, dK, dV with looser tolerance (1e-4 atol)
3. Training loss curve: 300 steps, should differ <1% from PyTorch by step 100
4. TC backward check: compare TC vs scalar backward on small input before training
5. Edge cases: world_size=4, odd T, dropout=0

---

## Bug Fixes Applied (from Critical Bugs Document)

| Bug | Description | Fix |
|-----|------------|-----|
| Bug 1 | Partial merger re-slices already-half-T input | Slice ACCUMULATOR, not input; use narrow_view + copy_ |
| Bug 2 | Wrong dim sliced for K/V gradient (dim-2 = heads, not time) | Use seq_dim=2 consistently with make_shards_inplace_axis |
| Bug 3 | dkv_rotater receives inconsistent tensor sizes | Pad T/2 gradients to full-T with zeros before sending |
| Error 4 | Double-accumulation of grad_k_step | Removed direct accumulation; only via travelling rotater |
| Error 5 | is_past != i <= rank in comments | Code uses i <= rank_ (correct), comments updated |
| Error 6 | Unreachable else branch when lb_active=true | Restructured control flow |
| Issue 7 | O(world_size) memory for saved K | Added recompute_k flag (default: save K) |

---

## Runtime Bug Fixes (Post-Implementation)

### Fix 1: narrow_view contiguity (SDPAMerger.h + ContextParallelBackward.h)
- `narrow_view().copy_()` fails with "destination must be contiguous" because narrow_view along a middle dimension creates non-contiguous strides
- **SDPAMerger.h**: Partial merge now uses `clone() + Tensor::cat()` to reconstruct tensor instead of in-place copy
- **ContextParallelBackward.h line 240-244**: dK/dV zero-padding now uses `Tensor::cat({grad_k_step, zeros}, dim)` instead of `narrow_view().copy_()`
- **ContextParallelBackward.h line 201-209**: dQ 2nd-half accumulation now uses `Tensor::cat({first_half.clone(), second_half.clone() + grad_q_step}, dim)`

### Fix 2: Generation hang (ContextParallel.h)
- Load balance was enabled for generation mode (pre_sharded=false) after changing `lb_active = load_balance_` (was `load_balance_ && !is_causal_`)
- Generation uses full [B,H,T,D] tensors, HeadTail applied before sharding -- different ordering than training path causing hang
- **Fix**: `lb_active = load_balance_ && pre_sharded` -- disables LB during generation, keeps it for training

### Fix 3: TC backward guard (FusedSDPABackwardKernel.cu)
- Re-applied `if(false)` guard at line 1078 to force scalar FP32 backward fallback
- TC forward remains enabled; only backward is guarded
- NOTE: This alone did NOT fix the explosion -- scalar backward also explodes at step ~147

### Fix 4: Merger autograd graph corruption (ContextParallel.h)
- **Root cause**: SDPAMerger uses `autograd::sigmoid` which creates autograd graph nodes inside the forward pass
- When `unshard=false` (training), `output_tensor = merged_out` carries this autograd history
- `set_grad_fn(ContextParallelBackward)` overwrites the grad_fn, but the internal autograd connections from sigmoid/log/cat persist
- The autograd engine differentiates through BOTH the merger's graph AND ContextParallelBackward, causing double/incorrect gradient computation that slowly accumulates until explosion
- **Fix**: `merged_out = merged_out_raw.detach()` and `merged_lse = merged_lse_raw.detach()` immediately after `merger.results()` -- severs the merger's autograd graph
- Also removed debug printf from scalar backward kernel

---

## Files Modified

- `DTensor/gpt2_cp_test/context_parallel/SDPAMerger.h`
- `DTensor/gpt2_cp_test/context_parallel/ContextParallel.h`
- `DTensor/gpt2_cp_test/context_parallel/ContextParallelBackward.h`
- `DTensor/gpt2_cp_test/context_parallel/FusedSDPABackwardKernel.cu`

---

## Next Steps

1. Rebuild and rerun with TC backward disabled -- verify grad norms stay stable past step 200
2. Compare loss curve with PyTorch CP baseline (gpt2_cp_headtail_fp32.py) over 300 steps
3. If parity achieved: run to 300+ steps for throughput measurement
4. Investigate TC backward WMMA bug separately (incorrect gradient accumulation in register tiles)
