# CP Tiling + Round-Robin Sub-Chunking Plan
**Date:** 2026-04-10  
**Workspace:** TensorParallelismBeta  
**Files:** FusedSDPAKernel.cu, ContextParallel.h, ContextParallelBackward.h, gpt2_cp_test.cpp

---

## Context

PyTorch's Efficient Attention is ~100ms faster than our C++ SDPA per ring step.  
Two causes identified:
1. **Tiling** — our kernel already tiles (Flash Attention style, BLOCK_K=32). Gap is tile size.  
2. **Round-Robin sub-chunking** — PyTorch slices Q/K/V to half-size before kernel launch for `source_rank > rank_` steps, giving 4x fewer FLOPs. We currently use `BOT_HALF` masking which still runs the full kernel.

---

## Part 1: Tiling — FusedSDPAKernel.cu

### What to change
- `BLOCK_K`: 32 → 64  
- Add `cudaFuncSetAttribute` (max smem) for all template specializations (not just generic path)

### Why
- Current smem at D=64, BLOCK_K=32: 16 KB (well under 48 KB limit — leaving headroom)  
- At BLOCK_K=64, D=64: smem = 32 KB — still fits, halves K-sweep iterations  
- At BLOCK_K=64, D=128: smem = 64 KB — needs extended smem attribute set  

### Code change
```cpp
// FusedSDPAKernel.cu
static constexpr int BLOCK_Q = 32;
static constexpr int BLOCK_K = 64;   // was 32
```

In dispatcher, add before each template kernel launch:
```cpp
cudaFuncSetAttribute(flash_attn_fwd_kernel<32>,
    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
// same for <64>, <128>
```

---

## Part 2: Round-Robin Sub-Chunking — ContextParallel.h

### Key insight

For `source_rank > rank_` (future chunk, load-balanced):

| Approach | Tensors passed | FLOPs |
|---|---|---|
| Old (BOT_HALF mask) | Q[T_q × D], K[T_k × D] | T_q × T_k (half wasted) |
| Round-robin | Q_back[T_q/2 × D], K_front[T_k/2 × D] | (T_q/2) × (T_k/2) = **4x less** |

### Step 2.1 — Add flag to constructor

```cpp
ContextParallel(..., bool use_roundrobin = true)
    : ..., use_roundrobin_(use_roundrobin) {}
```

Add `bool use_roundrobin_;` to private members.

### Step 2.2 — Pre-split local_q at start of forward_cp

After `local_q` is set (~line 138):

```cpp
Tensor local_q_front, local_q_back;
if (use_roundrobin_ && lb_active) {
    std::vector<Tensor> q_halves = local_q.make_shards_inplace_axis(2, 2);
    local_q_front = autograd::contiguous(q_halves[0]);  // [B, H, T/2, D]
    local_q_back  = autograd::contiguous(q_halves[1]);  // [B, H, T/2, D]
}
```

### Step 2.3 — Dual mergers

```cpp
SDPAMerger merger_front(/*convert_to_f32=*/true);  // accumulates q_front steps
SDPAMerger merger_back (/*convert_to_f32=*/true);  // accumulates q_back steps
```

(Replace the existing single `merger`.)

### Step 2.4 — Modified ring loop mask dispatch

Only change the `source_rank > rank_` branch:

```cpp
bool use_sub_chunk = false;
if (is_causal_ && lb_active) {
    if (source_rank == rank_) {
        mask_type = MaskType::CAUSAL;
    } else if (source_rank < rank_) {
        mask_type = MaskType::LEFT_HALF;   // unchanged
    } else {
        // future chunk
        if (use_roundrobin_) {
            use_sub_chunk = true;
            mask_type = MaskType::NONE;
        } else {
            mask_type = MaskType::BOT_HALF;  // fallback
        }
    }
}
```

### Step 2.5 — Sub-chunk SDPA call in ring loop body

Replace the single `sdpa_fused_forward` call with:

```cpp
if (use_sub_chunk) {
    std::vector<Tensor> k_halves = curr_k.make_shards_inplace_axis(2, 2);
    std::vector<Tensor> v_halves = curr_v.make_shards_inplace_axis(2, 2);
    Tensor k_front_h = autograd::contiguous(k_halves[0]);
    Tensor v_front_h = autograd::contiguous(v_halves[0]);

    int q_off_back   = rank_        * static_cast<int>(T_local_fwd) + static_cast<int>(T_local_fwd / 2);
    int k_off_front  = source_rank  * static_cast<int>(T_local_fwd);

    SDPAResult result = sdpa_fused_forward(
        local_q_back, k_front_h, v_front_h,
        MaskType::NONE, attn_scale_, q_off_back, k_off_front);

    saved_k_chunks[i] = k_front_h.clone();
    saved_v_chunks[i] = v_front_h.clone();
    saved_lse_per_step[i] = result.lse;
    merger_back.step(result.out, result.lse);
    // merger_front gets no update — q_front doesn't attend to future K
} else {
    int q_off = rank_       * static_cast<int>(T_local_fwd);
    int k_off = source_rank * static_cast<int>(T_local_fwd);
    SDPAResult result = sdpa_fused_forward(
        local_q, curr_k, curr_v, mask_type, attn_scale_, q_off, k_off);
    saved_lse_per_step[i] = result.lse;
    merger_front.step(result.out, result.lse);  // accumulates full-Q steps
}
```

**Note:** For the non-sub-chunk steps (`source_rank == rank_` and `source_rank < rank_`), `merger_front` accumulates full-Q results. For sub-chunk steps, `merger_back` accumulates q_back-only results. At the end, `merger_front` contains contributions from all past+self steps, and `merger_back` contains contributions from all future steps (only q_back rows).

This means `merger_front` already has the right T_q rows but `merger_back` only has T_q/2 rows. When we cat them along dim=2, the combined output is correct.

Wait — there's a subtle issue: `merger_front` accumulates full-Q (T_q rows) across multiple steps. The diagonal step and past steps both write to it. The future steps write q_back only to `merger_back`. So at the end:

- `merger_front.result` = merged output of [local_q × K_self] and [local_q × K_past] = shape [B,H,T_q,D]
- `merger_back.result` = merged output of [q_back × K_future_front] = shape [B,H,T_q/2,D]

We need to combine these: for q_front rows, use merger_front result; for q_back rows, use the merger_front q_back result PLUS merger_back result (re-merged).

This is the tricky part — **q_back rows appear in both mergers** (from non-sub-chunk steps they go into merger_front, from sub-chunk steps they go into merger_back).

**Correct approach:** Use THREE accumulators or use a partial-index merger.

**Simplest correct approach:** For non-sub-chunk steps, feed q_front and q_back into separate mergers explicitly:

```cpp
// For source_rank <= rank_ steps (non-sub-chunk):
std::vector<Tensor> q_result_halves = result.out.make_shards_inplace_axis(2, 2);
std::vector<Tensor> lse_result_halves = result.lse.make_shards_inplace_axis(2, 2);
merger_front.step(autograd::contiguous(q_result_halves[0]),
                  autograd::contiguous(lse_result_halves[0]));
merger_back.step(autograd::contiguous(q_result_halves[1]),
                 autograd::contiguous(lse_result_halves[1]));
```

Then at end:
```cpp
auto [out_front, lse_front] = merger_front.results();  // [B,H,T/2,D]
auto [out_back,  lse_back ] = merger_back.results();   // [B,H,T/2,D]
Tensor merged_out = Tensor::cat({out_front, out_back}, 2);  // [B,H,T,D]
Tensor merged_lse = Tensor::cat({lse_front, lse_back}, 2);  // [B,H,T,1]
```

This is clean and correct: front-half rows only accumulate non-sub-chunk steps, back-half rows accumulate non-sub-chunk steps + sub-chunk steps.

### Step 2.6 — Backward: ContextParallelBackward.h

Add to saved state:
```cpp
std::vector<bool> saved_use_sub_chunk_;  // per ring step
```

Pass from ContextParallel.h to ContextParallelBackward constructor.

In backward body, for sub-chunk steps:
```cpp
if (saved_use_sub_chunk_[i]) {
    // grad_output for this step only covers q_back
    // slice grad to q_back, run sdpa_fused_backward on (q_back, k_front, v_front)
    // scatter dQ_back back into full dQ
}
```

---

## Files to Change

| File | Change |
|---|---|
| `DTensor/gpt2_cp_test/context_parallel/FusedSDPAKernel.cu` | BLOCK_K 32→64; smem attribute for template paths |
| `DTensor/gpt2_cp_test/context_parallel/ContextParallel.h` | use_roundrobin_ flag; pre-split Q; dual-merger; sub-chunk call |
| `DTensor/gpt2_cp_test/context_parallel/ContextParallelBackward.h` | saved_use_sub_chunk; split backward for sub-chunk steps |
| `DTensor/gpt2_cp_test/gpt2_cp_test.cpp` | Pass use_roundrobin flag to ContextParallel |

---

## Implementation Order

1. `FusedSDPAKernel.cu` — isolated tile size change, validate output unchanged  
2. `ContextParallel.h` forward only — add flag, dual-merger, sub-chunk call  
3. Validate: loss curve with `use_roundrobin=false` vs `true` must match  
4. `ContextParallelBackward.h` — sub-chunk backward  
5. `gpt2_cp_test.cpp` — toggle flag for throughput comparison  

---

## Fallback

All masking provisions (`BOT_HALF`, `LEFT_HALF`) remain in the code.  
Pass `use_roundrobin=false` to revert to old behavior at any time.
