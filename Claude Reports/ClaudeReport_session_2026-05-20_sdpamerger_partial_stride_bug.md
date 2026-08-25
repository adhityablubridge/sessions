2026-05-20 - SDPAMerger partial-merge narrow_view stride bug fix - TensorParallelismBeta / SDPAMerger.h

# Summary

Found and fixed the LB-true training divergence root cause: `SDPAMerger::step` partial-merge path was producing wrong outputs whenever a head index > 0 was involved, because the tensor library's element-wise binary ops treat a `narrow_view` on a non-leading axis as a contiguous flat slab and ignore per-axis strides.

# Investigation Path

1. Built isolated CP backward parity test (PT + C++) that loads identical Q/K/V/dY .bin files and runs one forward+backward with `pre_sharded=true, load_balance=true`.
2. Confirmed perm parity: PT's `_generate_round_robin_indices` produces the same `Q_local` as C++'s `shard_sequence_pre_embed` chunk-level HeadTail formula (bit-exact).
3. Confirmed Tensor library ops (narrow_view+clone+cat) work correctly in isolated test (`tensor_lib_partial_update_test`).
4. With `CP_LB=0` (load balance off): forward and backward match within fp32 noise. With `CP_LB=1`: rank 0 diverges massively in backward.
5. Added `DUMP_CP_DEEP=1` instrumentation in `ContextParallelBackward.h`: pinpointed bug entry at step 0 grad_q_step on rank 0 (rel_max 16.5%), rank 1 clean. Since SAME backward kernel different rank's data → different result, the bug must be UPSTREAM (in forward outputs feeding backward).
6. Added `DUMP_CP_DEEP_FWD=1` in `ContextParallel.h::forward_cp`: dumped per-ring-step `block_out`/`block_lse` (raw SDPA results) and `merged_out`/`merged_lse` (after merger.step). Result: SDPA results within noise on both ranks. Merger after rank 0 step 1 (the partial-merge call) blew up to 14.3% rel_max on `merged_out`. Rank 1 step 1 (full merge) stayed within noise.
7. Added `DUMP_CP_MERGE=1` in `SDPAMerger.h` to dump all intermediate tensors (`accum_out`, `accum_lse`, `lse_diff`, `sig`, `out_diff`, `correction`, `new_out`, `neg_lse_diff`, `sig_neg`, `log_sig`, `new_lse`). Mirrored in PT via monkey-patch on `_SDPAMerger._merge_one`.
8. Found: `accum_lse` (post `to_cpu()`) matches PT bit-for-bit at all positions, but `lse_diff = block_lse - accum_lse` differs ENORMOUSLY at head 1 (5x bigger) while head 0 matches. Impossible under correct stride-aware subtraction.
9. Hypothesis confirmed bit-exactly: C++ binary op reads `accum_lse` as `lse_.flatten()[half_T : half_T + total_elements]`, ignoring per-row stride. For head 0 this coincidentally maps to head 0's second half (correct), but for head 1 it reads head 1's FIRST half (garbage).

# The Bug

Location: `TensorParallelismBeta/DTensor/gpt2_cp_test/context_parallel/SDPAMerger.h`, partial-merge branch.

```cpp
// Before fix:
accum_out = out_.narrow_view(seq_dim, half_T, half_T);   // strided view
accum_lse = lse_.narrow_view(lse_seq_dim, half_T, half_T); // strided view
...
Tensor lse_diff = block_lse - accum_lse;  // BUG: op flat-fast-paths and reads wrong physical memory
```

`narrow_view` on dim 2 of a 4D `[B,H,T,1]` tensor returns a view with logical shape `[B,H,T/2,1]` but underlying-storage offset `T/2` and stride pattern that's NOT flat-contiguous (it has a per-row gap because each head spans `T*1` flat elements but the view only wants the latter half of each head's T).

The C++ tensor library's binary op (`operator-`) appears to use a flat-pointer fast path that treats the view's contiguous-start-offset as the slab origin and iterates `numel()` elements flat. For head 0 this happens to be correct (head 0's second half lives exactly at flat offset T/2). For head 1 the iteration reads from `T/2 + T = 3T/2` which lands in head 1's FIRST half — completely wrong data.

# The Fix

```cpp
// After fix:
accum_out = out_.narrow_view(seq_dim, half_T, half_T).clone();
accum_lse = lse_.narrow_view(lse_seq_dim, half_T, half_T).clone();
```

`.clone()` materializes a fresh contiguous tensor so the binary op's flat-fast-path becomes correct.

This is a workaround. The proper fix is to make the tensor library's element-wise binary ops honor per-axis strides for non-contiguous views. That's a bigger change with wider blast radius; the narrow_view + clone pattern in the merger is a surgical local fix.

# Verification

Forward output divergence (PT vs C++, rank 0):

| Tensor | Before | After |
|---|---|---|
| merged_out rel_max | 14.3% | 0.063% (fp32 noise) |
| merged_lse rel_max | 8.9% | 0.029% (fp32 noise) |

Backward (full ContextParallelBackward chain, isolated test, both ranks):
- dQ/dK/dV rel_max ≤ 0.3% (was 16-25% on rank 0)
- cos_sim = 0.9999997 (was 0.85 on rank 0)
- max_abs_diff ≤ 5e-5

Rank 1 unchanged (it was already clean — doesn't hit the partial-merge code path).

# Files Modified

- `TensorParallelismBeta/DTensor/gpt2_cp_test/context_parallel/SDPAMerger.h` — `.clone()` fix on accum views in partial-merge branch; added `DUMP_CP_MERGE=1` gated intermediate dumps; added `merge_call_counter_` member.
- `TensorParallelismBeta/DTensor/gpt2_cp_test/context_parallel/ContextParallel.h` — added `DUMP_CP_DEEP_FWD=1` gated per-step forward dumps (block_out/block_lse + merged_out/merged_lse after each merger.step) and final merged_out/merged_lse.
- `TensorParallelismBeta/DTensor/Pytorch/cp_bwd_isolated_test.py` — added DUMP_CP_DEEP_FWD and DUMP_CP_MERGE monkey-patches on `_SDPAMerger.step` / `_merge_one`.
- `TensorParallelismBeta/DTensor/Pytorch/cp_bwd_diff_deep.py` — extended diff script to handle per-fwdstep and forward-final dumps.

# Follow-ups / Risk

1. **Wider tensor library bug.** Other call sites that combine `narrow_view` on non-leading axes with binary ops likely have the same latent issue. Recommend grepping `narrow_view\(.*,.*,.*\)` and auditing each for downstream element-wise use without an intervening `.clone()`. Long-term, fix the binary op kernels to honor strides.
2. **Backward instrumentation still in place.** `DUMP_CP_DEEP=1`, `DUMP_CP_DEEP_FWD=1`, `DUMP_CP_MERGE=1` are all env-gated and inert when off. Safe to leave; remove if no longer needed.
3. **Confirm in training loop.** Isolated test proves the fix correct under one forward+backward. User should rerun the full GPT-2 CP training (where grad-norm exploded at step 33 with LB on) to confirm the step-33 explosion is resolved.
