# ClaudeReport — CP vs Normal wpe Gradient Mismatch

**Date:** 2026-03-28
**Workspace:** TensorParallelismBeta
**Files:** DTensor/gpt2_cp_test/gpt2_cp_test.cpp, gpt2/gpt2_attn_fixed.cpp
**Symptom:** Gradient values mismatch between CP (world_size=2) and normal (world_size=1) for some parameters
**Iterations:** 2

---

## Setup Confirmed

Both runs use identical configuration:
- B=4, T=1024, n_heads=6, n_embd=384, n_layers=3
- grad_accum_steps=16, global_batch=65536, grad_scale=1/16
- Same data (DataLoaderLite rank=0, world_size=1 for both)
- Same model seed (1234)
- weight_tying=false (both runs have independent lm_head)
- Loss nearly identical: 10.904800 vs 10.904799

---

## Differential Analysis

| Parameter | CP grad rows 0-2 | Normal grad rows 0-2 | Match |
|-----------|-----------------|---------------------|-------|
| c_proj weight | `[[0.0007, 0.0006, ...]]` | `[[0.0007, 0.0006, ...]]` | ✓ |
| ln bias (384) | `[-0.0135, 0.0094, ...]` | `[-0.0135, 0.0094, ...]` | ✓ |
| lm_head (384,50304) | `[[-0.0000, 0.0001, ...]]` | `[[-0.0000, 0.0001, ...]]` | ✓ |
| wte (50304,384) | tiny differences | tiny differences | ~✓ |
| **wpe (1024,384) early rows** | **~1e-4 magnitude** | **~1e-3 magnitude** | **X (10-17x smaller in CP)** |
| wpe tail rows (positions ~1020-1023) | identical | identical | ✓ |
| ln_f gamma, ln_f beta | identical | identical | ✓ |

---

## Root Cause — Phase 3 sendrecv in ContextParallelBackward

**Why early positions (0..511) specifically?**

In a causal transformer with world_size=2:
- Rank 0 handles Q, K, V for positions 0..511
- Rank 1 handles Q, K, V for positions 512..1023

Position i's wpe gradient includes contributions from:
1. Q[i]'s backward (dQ at position i)
2. K[i]'s backward: K[i] is used by Q[i..T-1] (all queries that can attend to K[i])
3. V[i]'s backward: V[i] is used by Q[i..T-1]

For positions 0..511 (rank 0):
- K[0..511] is used by Q[0..511] (rank 0, step i=0 — local, correct)
- K[0..511] is ALSO used by Q[512..1023] (rank 1, step i=1)
- dK contribution from Q[512..1023] attending K[0..511] is computed on rank 1
- **This must be sent back to rank 0 via Phase 3 sendrecv**

For positions 512..1023 (rank 1):
- K[512..1023] is only used by Q[512..1023] (rank 1's own causal step)
- No cross-rank communication needed
- Gradient is fully correct locally

**The observed pattern (tail matches, early positions wrong) is the exact fingerprint of Phase 3 sendrecv failing.**

**Phase 3 code (ContextParallelBackward.h:150-169):**
```cpp
auto work = pg_->sendrecv_async(
    dkv_concat.data<float>(),   // rank 0 sends zeros (step i=1 was skipped)
    recv_buf.data<float>(),     // should receive dK[0..511] from rank 1
    dest_rank,   // = 1
    recv_from,   // = 1
    count,
    dkv_concat.dtype());

if (work) {
    work->wait();   // SKIPPED if work == nullptr
}

Tensor recv_flat = recv_buf.flatten();
local_grad_k = local_grad_k + recv_dk;   // += zeros if recv failed
```

**If `pg_->sendrecv_async()` returns nullptr:**
- `if (work)` is false
- wait is never called
- `recv_buf` remains uninitialized or zeroed
- `recv_dk` = zeros
- Rank 0's `local_grad_k` = only contribution from Q[0..511] (missing Q[512..1023]'s contribution)
- After all_gather: full_grad_k for positions 0..511 is incomplete
- Flows back through TransposeBackward → ReshapeBackward → c_attn backward → h_grad → ln_grad → x_grad → pos_emb_grad → wpe.weight.grad
- wpe.weight.grad[0..511] = too small (missing K/V backward from rank 1's queries)
- wpe.weight.grad[512..1023] = correct (rank 1 computes fully locally)

This matches EXACTLY what the terminals show.

---

## Forward is Correct (Expected)

The forward pass is not affected because Phase 3 is only in the backward. The forward correctly gathers K/V and computes full-context attention. Loss values match (10.904800 vs 10.904799). ✓

---

## What to Investigate

1. Check whether `pg_->sendrecv_async()` can return nullptr in the AlltoAll context
2. Check the `ProcessGroupNCCL::sendrecv_async` implementation — does it require being inside an NCCL group communicator call? AlltoAll ring might initialize NCCL differently from P2P.
3. Verify whether `work->wait()` actually blocks until recv completion, or only until send completion
4. Add a debug assertion: `assert(work != nullptr)` after sendrecv_async in Phase 3
5. Add `cudaStreamSynchronize(0)` after `work->wait()` to ensure GPU completion

---

## Note on Previous Bug (World Size >= 3)

The aliasing bug (saved_k_chunks[i] aliasing recv_buffer_) is separate and only activates for world_size >= 3. For world_size=2 there is only one recv step so the buffer is not overwritten. The Phase 3 sendrecv bug is the active issue for world_size=2.

---

## Fix Directions

**Option A:** If `sendrecv_async` can return nullptr legitimately, add fallback:
```cpp
auto work = pg_->sendrecv_async(...);
if (work) {
    work->wait();
} else {
    // Fallback: use blocking send/recv or different collective
}
```

**Option B:** Replace Phase 3 sendrecv with `all_reduce` (simpler, correct for any world_size):
```cpp
// Each rank holds full local_grad_k [B,H,T/2,D] containing local contribution
// all_reduce sums across ranks, but that's wrong since each rank holds different positions
// Instead: scatter-reduce (reduce-scatter) would work, or explicit P2P per source rank
```

**Option C:** Use all_gather + local summation instead of sendrecv:
```cpp
// all_gather all ranks' grad_k_accum[i] contributions to each rank
// then each rank sums the contributions relevant to its K positions
```
