# Claude Report — AllGather Forward Stale-Gather Bug + Fix

- **Date/Time:** 2026-06-22 13:54
- **Work:** Diagnose and fix the residual AllGather-only training-loss mismatch (noisier/higher than PyTorch and the P2P/AlltoAll rotators) that remained after the backward dK/dV fix.
- **Workspace / File:** OldPush / `TensorParallelismBeta/DTensor/gpt2_cp_test/context_parallel/ContextParallel.h`

---

## Context / how we got here

Earlier this session: fixed an AllGather-only **backward** bug (the dK/dV travelling accumulator used the gather-once AllGather rotator; forced it to AlltoAll). After rebuilding with that fix, the gross monotonic divergence was gone, but AllGather's training loss was still **noisier and slightly higher** than PyTorch / P2P / AlltoAll.

Key correction from the user that redirected the diagnosis: precision (TF32/FP32) is identical across all three rotators, and P2P/AlltoAll match PyTorch — so the residual gap is **AllGather-specific**, not a precision (ATTN_FP32) issue. (My ATTN_FP32 hypothesis was wrong and retracted.)

## Root cause (forward path)

The forward keeps a **persistent** rotator (`ContextParallel.h:352-358`) created once (keyed on `persistent_kv_numel_`, which is constant during training) and reused on every forward call:

```cpp
if (persistent_kv_numel_ != kv_numel) { ...; kv_rotator_persistent_ = create_rotator(); persistent_kv_numel_ = kv_numel; }
kv_rotator = kv_rotator_persistent_.get();
```

`AllGatherRingRotator` (`RingRotator.h:303-376`) gathers all ranks' K/V **once** on its first `exchange_buffers` (`if (!aggregated_buffer_.is_valid())`) and never re-gathers; `idx_` only increments; there is no `reset()`. So with a persistent instance reused across steps:

1. **Stale KV** — it gathers on the first forward (step 0, untrained weights) and serves that cached buffer forever; every later step attends to step-0 K/V for non-local shards.
2. **Drifting shard index** — `idx_` keeps incrementing across steps, so `next_buffer()`'s `source_rank = (rank - idx_) mod N` selects the wrong shard each step (ws=2: alternates peer-shard / own-shard).

P2P and AlltoAll re-communicate fresh data on every `exchange_buffers`, so reusing the persistent instance only reuses buffer storage — they are immune. AllGather also can't overlap (single blocking all_gather), so persistence gives it nothing.

### Why this matched every symptom
- **AllGather-specific** (P2P/AlltoAll immune).
- **Step-0 matches** all rotators (gather is fresh on the first forward).
- **Grows / noisier over training** (cached KV gets staler; the fresh local shard only partially compensates -> loss descends but sits higher/noisier).
- **Survived the backward dkv fix** (this is forward; backward rotators are created fresh per call).

## Fix

Recreate the AllGather rotator each forward (fresh `idx_=0`, invalid buffer -> re-gathers the current step's K/V). Localized; P2P/AlltoAll behavior unchanged.

```cpp
if (rotator_type_ == RotatorType::AllGather) {
  kv_rotator_persistent_ = create_rotator();
}
kv_rotator = kv_rotator_persistent_.get();
```

## Verification / follow-ups (NOT done here)

- Not rebuilt/rerun this session. Next: rebuild C++ and re-run the AllGather config; expect AllGather to collapse onto P2P/AlltoAll/PyTorch (noise + elevation gone).
- The separately-noted per-100-step grad_norm/loss spikes (validation + token-generation steps) may also shrink, since those extra forwards previously perturbed the persistent AllGather state; re-check after rebuild.
- Both AllGather fixes now in place: backward dkv -> AlltoAll (`ContextParallelBackward.h:156`), forward -> recreate per call (`ContextParallel.h:352`).
