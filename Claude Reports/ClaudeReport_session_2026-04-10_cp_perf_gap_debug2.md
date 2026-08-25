# ClaudeReport — 2026-04-10 19:25 — CP Throughput Gap Debug (55.5k vs 64k PyTorch)
**Workspace:** TensorParallelismBeta
**Branch:** _adhi_

---

## 1. Primary Request and Intent

Close performance gap: custom CP 55.5k tok/sec vs PyTorch 64k tok/sec at world_size=2.
- Debug why `attn_cp` remains ~180-196ms while PyTorch gets ~120ms
- Extend TC kernel (WMMA) to support T_q != T_k (separate Q and K sequence lengths)
- Enable TC for LEFT_HALF and BOT_HALF ring steps via sub-chunk dispatch
- Fix incorrect sub-chunk dimension in future ring step (BOT_HALF should be Q_back x K_full)
- User requested caveman response mode throughout

---

## 2. Key Technical Concepts

- **CP ring attention**: ring of K/V across ranks, Q stays local, world_size=2
- **WMMA TC kernel** (`mem_efficient_attn_forward_tc`): single T param extended to T_q, T_k, K_head_stride
- **K_head_stride**: per-BH stride in float elements for K tensor; allows non-contiguous K slices (BOT_HALF offset, LEFT_HALF prefix)
- **MaskType dispatch**: CAUSAL=self-step, LEFT_HALF=past-step, BOT_HALF=future-step, NONE=unconstrained
- **PyTorch CP sub-chunk pattern**: future step uses Q_back[T/2] x K_full (partial=True), past step uses Q_full x K_front[T/2]
- **BOT_HALF bug**: our implementation used Q_full x K_back[T/2] (2x grid blocks, wrong attention pattern), PyTorch uses Q_back[T/2] x K_full (half grid blocks)
- **Two-merger approach**: split CAUSAL/LEFT_HALF results into front/back halves, feed merger_front and merger_back; future step feeds only merger_back; Phase 3 cats results
- **-inf padding approach**: keep single merger, pad future step result to full T with -inf LSE for front rows
- **autograd::contiguous() overhead**: creates device-to-device copy; splitting every step adds latency

---

## 3. Files and Code Sections

### DTensor/dnn/AttentionKernels.cu
TC kernel signature extended with T_q/T_k/K_head_stride.
- `fused_attn_forward_kernel_tc<HD>`: changed from single `int64_t T` to `int64_t T_q, int64_t T_k, int64_t K_head_stride`
- K_bnh/V_bnh use `bnh * K_head_stride`, Q_bnh/O_bnh/LSE_bnh use T_q
- max_kj: non-causal = T_k, causal = min(q_offset + qi_block + actual_q - k_offset, T_k)
- `launch_fwd_tc_kernel`: added `int64_t T_k = -1, int64_t K_head_stride = -1` params; resolves defaults
- LAUNCH_FWD_TC macro: replaced `auto* kernel` with explicit function pointer type to fix type deduction error:
```cpp
void (*kernel)(const float*, const float*, const float*,
               float*, float*,
               int64_t, int64_t, int64_t,
               float, bool, float, const float*, int, int)
    = fused_attn_forward_kernel_tc<HD>;
```

### DTensor/dnn/AttentionKernels.h
Declaration updated:
```cpp
void mem_efficient_attn_forward_tc(
    const float* query, const float* key, const float* value,
    float* output, float* lse,
    int64_t B, int64_t nh, int64_t T_q, int64_t hd,
    bool is_causal,
    float dropout_p = 0.0f,
    const float* dropout_mask = nullptr,
    int q_offset = 0,
    int k_offset = 0,
    int64_t T_k = -1,
    int64_t K_head_stride = -1
);
```

### DTensor/gpt2_cp_test/context_parallel/FusedSDPAOp.h
TC dispatch for LEFT_HALF/BOT_HALF (currently active):
```cpp
bool tc_mask_ok = (mask_type == MaskType::CAUSAL || mask_type == MaskType::NONE);
bool tc_half_ok = ((mask_type == MaskType::LEFT_HALF ||
                    mask_type == MaskType::BOT_HALF) && (T_k % 2 == 0));
bool use_tc = (D <= 128) && (tc_mask_ok || tc_half_ok);
if (use_tc) {
  if (tc_mask_ok) {
    // direct TC call
  } else if (mask_type == MaskType::LEFT_HALF) {
    // Q_full x K_front[T_k/2], K_head_stride = T_k * D
    mem_efficient_attn_forward_tc(..., T_k / 2, T_k * D);
  } else {
    // BOT_HALF: offset K/V ptr into second half of BH=0 slice
    const int64_t half_offset = (T_k / 2) * D;
    mem_efficient_attn_forward_tc(Q_ptr, K_ptr + half_offset, V_ptr + half_offset,
                                  ..., k_offset + T_k/2, T_k / 2, T_k * D);
  }
}
```

### DTensor/gpt2_cp_test/context_parallel/ContextParallel.h
**REVERTED to original state.**
- `rr_active = false` (hard-coded)
- BOT_HALF masking on full tensors for future steps
- Single merger_front
- Two-merger attempt caused regression to 196ms attn_cp (was 180ms) due to `autograd::contiguous()` overhead on every ring step split

### DTensor/gpt2_cp_test/context_parallel/ContextParallelBackward.h
Sub-chunk backward path updated (dead code since rr_active=false):
- Changed `is_sub_chunk=true` path to use full K/V (no zero-padding for dK/dV)
- `saved_k_chunks_[i]` and `saved_v_chunks_[i]` accumulated directly

**Identified waste (not yet fixed):** `step_q = saved_q_.clone()`, `step_k = saved_k_chunks_[i].clone()`, `step_v = saved_v_chunks_[i].clone()` in full-tensor backward loop — all unnecessary clones since kernel only reads these tensors.

---

## 4. Problem Solving

### Solved
- TC kernel extended to support T_q != T_k via K_head_stride parameter
- FusedSDPAOp.h LEFT_HALF/BOT_HALF now dispatch to TC kernel
- Build error fixed: `auto* kernel` type deduction after signature change — replaced with explicit function pointer

### Failed attempt
- **Two-merger ContextParallel approach**: split Q front/back every step, two mergers, cat at Phase 3
  - Root cause regression: `autograd::contiguous()` called 4x per full-Q step + 2x for sub-chunk split
  - attn_cp went 180ms → 196ms
  - Reverted immediately

### Unsolved / Current gap
- 55.5k vs 64k remains open
- Forward attn_cp: ~180ms (PyTorch ~120ms)
- Backward: ~650ms (suspected bottleneck, not yet profiled vs PyTorch)
- **Root cause of BOT_HALF inefficiency**: our grid dispatches 2x Q tiles vs PyTorch's Q_back-only dispatch
- **Unnecessary backward clones**: `saved_q_.clone()`, `saved_k_chunks_[i].clone()`, `saved_v_chunks_[i].clone()` in each backward ring step — wasted D2D copies each step

---

## 5. Pending Tasks

- Remove unnecessary `.clone()` calls in ContextParallelBackward.h backward loop (lines ~175-177)
- Profile backward timing to confirm whether bwd or fwd is the dominant bottleneck
- Investigate PyTorch's `partial=True` sub-chunk approach for future ring step in forward without split overhead
- Verify if LEFT_HALF TC dispatch is firing and reducing attn_cp

---

## 6. Current Work

Session ended mid-attempt to remove `.clone()` calls in ContextParallelBackward.h. User denied the edit. The three redundant clones in the full-tensor backward path are:

```cpp
// ContextParallelBackward.h ~line 175
Tensor step_q = saved_q_.clone();           // unnecessary — kernel is read-only
Tensor step_k = saved_k_chunks_[i].clone(); // unnecessary
Tensor step_v = saved_v_chunks_[i].clone(); // unnecessary
```

These should be replaced by passing `saved_q_`, `saved_k_chunks_[i]`, `saved_v_chunks_[i]` directly to `sdpa_fused_backward`.

---

## 7. Optional Next Step

Remove unnecessary clones in backward loop. Then profile to confirm bwd time vs fwd time split.

**Verbatim from last exchange:**
- User: "the throughput is still at 55.5k vs 64k pytorch"
- Assistant was about to edit ContextParallelBackward.h line 175 to remove `step_q = saved_q_.clone()` etc.
- User interrupted the edit tool call
