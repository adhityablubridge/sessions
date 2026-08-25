# ClaudeReport — CP Backward Wrong Gradients Debug

**Date:** 2026-03-28
**Workspace:** TensorParallelismBeta
**Files analyzed:** ContextParallelBackward.h, SDPAOp.h, ContextParallel.h, RingRotator.h
**Symptom:** Wrong gradients in context parallel backward pass
**Severity filter:** Critical + High

---

## Summary

Full investigation across all four CP files. Earlier hypotheses about sdpa_backward_op_manual were wrong — the math is correct. Root cause is a buffer aliasing bug in the forward pass: saved K/V chunks share storage with the rotator's recv buffer, which is overwritten each ring step.

---

## Root Cause — CONFIRMED CRITICAL

### Forward saves views into a reused recv buffer (ContextParallel.h:216-217)

```cpp
// In the forward ring loop (i > 0):
Tensor next_kv = kv_rotator->next_buffer();       // returns recv_buffer_ (same object every time)
Tensor kv_flat  = next_kv.flatten();
curr_k = kv_flat.narrow(0, 0, k_numel).reshape(local_k.shape()); // VIEW into recv_buffer_
curr_v = kv_flat.narrow(0, k_numel, k_numel).reshape(local_v.shape());

saved_k_chunks[i] = curr_k;   // saves VIEW -- shares storage with recv_buffer_
saved_v_chunks[i] = curr_v;
```

P2PRingRotator (RingRotator.h:58-61) pre-allocates recv_buffer_ **once** and reuses it:
```cpp
if (!buffer_allocated_) {
    recv_buffer_ = Tensor::empty(curr_buffer.shape(), curr_buffer.opts());
    buffer_allocated_ = true;
}
// next_buffer() returns recv_buffer_ — same object, same memory, every step
```

AlltoAllRingRotator has the same structure (recv_buffer_ allocated once, returned every call).

**Effect on saved state:**
- After step i=1: `saved_k_chunks[1]` → recv_buffer_ (has step-1 K)
- After step i=2: NCCL overwrites recv_buffer_ with step-2 K. `saved_k_chunks[1]` and `saved_k_chunks[2]` both point to recv_buffer_, which now holds step-2 K.
- After step i=N-1: all `saved_k_chunks[1..N-1]` hold the last step's K.

**Backward consequence:** Every ring step i>0 recomputes SDPA with the wrong K/V. dQ, dK, dV are wrong for all non-local steps.

**Affected rotators:**
- P2PRingRotator — CONFIRMED broken
- AlltoAllRingRotator — CONFIRMED broken
- AllGatherRingRotator — NOT affected (allocates fresh Tensor per call in next_buffer())

**Fix (one line change in ContextParallel.h):**
```cpp
// Line 216-217 — clone to own the data before the rotator buffer is overwritten:
saved_k_chunks[i] = curr_k.clone();
saved_v_chunks[i] = curr_v.clone();
```

---

## Other Finding

### HIGH — K/V gradient buffers use Q's shape (ContextParallelBackward.h:88-89)

```cpp
grad_k_accum[i] = Tensor::zeros(saved_q_.shape(), saved_q_.opts());
grad_v_accum[i] = Tensor::zeros(saved_q_.shape(), saved_q_.opts());
```

K and V shapes may differ from Q (GQA, MQA). Incorrect shape for zero buffer — wrong for grouped-query attention. Does not affect standard MHA where Q/K/V share shape.

**Fix:**
```cpp
grad_k_accum[i] = Tensor::zeros(saved_k_chunks_[i].shape(), saved_k_chunks_[i].opts());
grad_v_accum[i] = Tensor::zeros(saved_v_chunks_[i].shape(), saved_v_chunks_[i].opts());
```

---

## Retracted Findings

The following bugs reported in the initial analysis were incorrect after reading SDPAOp.h:

- **CRITICAL-1 (merged_out_ for D_global):** RETRACTED. D_global = rowsum(dO * merged_out) is mathematically correct — merged_out IS the full attention output O, and D = rowsum(dO * O) is the correct global D term per flash attention backward.

- **CRITICAL-3 (unrescaled gradient):** RETRACTED. sdpa_backward_op_manual correctly captures the per-step weighting through P_global = P_local * exp(lse_diff). The raw grad_output does not need pre-rescaling.

- **HIGH-1 (no saved per-step outputs):** RETRACTED. Per-step outputs are not needed because merged_out provides the correct D_global.

- **HIGH-2 (step-0 validity check):** LOW risk. step 0 (own chunk) is never invalid in practice — the causal logic only skips future chunks (i > 0 steps for rank 0), and step 0 is always rank_'s own K which is always valid.

---

## Final Bug Table

| ID | Severity | File | Line | Status |
|----|----------|------|------|--------|
| ROOT CAUSE | **Critical** | ContextParallel.h | 216-217 | CONFIRMED — view into reused recv_buffer_ corrupts saved K/V for all steps i>0 (P2P and AlltoAll rotators) |
| BUG-2 | High | ContextParallelBackward.h | 88-89 | CONFIRMED — K/V buffer shaped from Q, wrong for GQA |

---

## Fix

In **ContextParallel.h**, lines 216-217:

```cpp
// Before (broken):
saved_k_chunks[i] = curr_k;
saved_v_chunks[i] = curr_v;

// After (correct):
saved_k_chunks[i] = curr_k.clone();
saved_v_chunks[i] = curr_v.clone();
```

This clones the data into independent storage before the rotator overwrites recv_buffer_ in the next ring step. The clone at backward time (ContextParallelBackward.h:101-102) is too late — the data is already corrupted during the forward.
