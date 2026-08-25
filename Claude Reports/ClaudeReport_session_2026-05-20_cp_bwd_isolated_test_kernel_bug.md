2026-05-20 - Built isolated CP backward parity test (PT + C++), pinned LB-true training divergence to exp11 SDPA backward kernel producing ~1.35x over-amplified dQ at head index >= 1 in chunk_3 tail region - Workspace: OldPush/TensorParallelismBeta/DTensor, Files: gpt2_cp_test/cp_bwd_isolated_test.cpp, Pytorch/cp_bwd_isolated_test.py, tensor_lib_partial_update_test.cpp, Pytorch/cp_bwd_diff_finals.py, ContextParallelBackward.h, Makefile

## Goal
Localize the LB-true training divergence (norm explosion at step 33) to a single component by building an isolated CP-backward parity test that exercises the full ring-attention backward outside the model.

## What got built

### 1. PT isolated test: `Pytorch/cp_bwd_isolated_test.py`
- 2-rank torchrun script.
- Generates Q/K/V/dY (seed 1234) on rank 0, saves to /tmp/cp_bwd_test/*.bin so C++ can load identical bytes.
- Applies HeadTail (round-robin) permutation via `_generate_round_robin_indices`, shards to per-rank.
- Calls `_templated_ring_attention` (efficient backend) forward → out, lse.
- Copies + instruments `_templated_ring_attention_backward` to dump per-step grad_q_step / grad_k_step / grad_v_step and the in-loop accumulator state.
- Test sizes B=1, H=2, T=128, D=64 (T_local=64; T_local/2=32 to avoid efficient-attention LSE padding to multiple of 32; D=64 to hit same dispatch path as training).

### 2. C++ isolated test: `gpt2_cp_test/cp_bwd_isolated_test.cpp` (+ Makefile target)
- Loads Q/K/V/dY from same bin files.
- Runs `ContextParallel::forward_cp(...)` with `pre_sharded=true, unshard=false`.
- Triggers `autograd::backward(output, &dY_local)` to run the live `ContextParallelBackward::apply`.
- Reuses existing `DUMP_CP_STEPS=1` per-step instrumentation in `ContextParallelBackward.h`.
- Clears stale `step_bw_rank{r}.md` at start (the existing dump opens in append mode).
- Saves final dQ/dK/dV bins for offline diff.

### 3. Tensor-library sanity test: `gpt2_cp_test/tensor_lib_partial_update_test.cpp`
- Standalone single-GPU test exercising `narrow_view + clone + cat` and `cat({step, zeros}) + grad_key` patterns against hand-built expected outputs.
- All three subtests (partial dQ pattern, full step-0-add + step-1-partial-update, head-half pad-then-add) returned `max_abs_diff = 0`.
- Conclusion: Tensor library ops are bit-exact correct; bug is NOT in the accumulation primitives.

### 4. Final-grad diff: `Pytorch/cp_bwd_diff_finals.py`
- Loads PT-side and C++-side dQ/dK/dV bins, reports max/mean abs diff, cosine similarity, argmax position, and per-T-position mean-abs-diff profile (head vs tail).

## Findings

### Per-step kernel outputs match within ~0.2-0.3% at sampled positions
- step 0 (causal full) grad_q_step / grad_k_step / grad_v_step: PT vs C++ match to ~0.1% at H=0 chunk_0 and H=0 chunk_3 start.
- step 1 (partial for rank 0, head-half for rank 1): match to ~0.1-0.3%.
- dK and dV final values match PT closely on both ranks (cos_sim 0.989-0.998).

### Final dQ on rank 0 has localized error
- rank 0 dQ: max_abs_diff=4.28e-3, cos_sim=0.9920, **62.77x more error in chunk_3 tail than chunk_0 head**.
- rank 0 dQ argmax at [B=0, H=1, T=44, D=51]: PT=-5.6e-3, C++=-9.9e-3.
- rank 1 dQ: cos_sim=0.99999971 (essentially bit-exact). rank 1 only uses full-add for dQ (head-half path).

### Pinpointed: H=1 chunk_3 grad_q_step at step 0 is systematically 1.35x of PT
Sampled 4 positions on grad_q_step at step 0 (causal full):
- H0_chunk0_start: both PT (~1e-9) and C++ (~1e-6) at noise floor (algebraically-zero positions).
- **H0_chunk3_start: PT vs C++ match to 0.1%.**
- H1_chunk0_start: both at noise floor.
- **H1_chunk3_start: C++ values are 1.32x-1.39x of PT, consistently, across all 16 sampled dims.**

dK and dV at H1_chunk3 match PT correctly. Only dQ is amplified.

## Conclusion

The training-time LB-true divergence at step 33 is caused by `mem_efficient_bwd_unified_kernel_exp11` (the WMMA/TF32 backward kernel dispatched for D=64) producing systematically over-amplified dQ at head index >= 1 specifically in the Q chunk_3 (tail half under HeadTail layout) region.

The error is in the dQ accumulation path. Suspect location: lines 510-518 of `AttentionBackward.cu` — the per-warp `dq_frag` is stored to `tile_st` AFTER the KV loop (which reuses `tile_st` for dK and dV stores throughout the loop), then atomicAdded to global `dQ`. There may be a register / shared-memory layout issue when the head index changes that doesn't affect dK/dV (which atomicAdd inside the loop with fresh state each iteration).

## LB-OFF comparison RUN (CP_LB=0 added to both scripts) — kernel is innocent
After adding a CP_LB=0 toggle to both PT and C++ tests and rerunning:
- **LB-OFF: every dQ/dK/dV matches PT within ~1e-5 (fp32 TF32-WMMA noise floor), cos_sim 0.99999970-0.99999980 on all six per-rank tensors.**
- LB-ON: errors 100x larger and localized to the LB-specific accumulation paths.

Mapping which path broke which output:

| | rank 0 dQ | rank 0 dK/dV | rank 1 dQ | rank 1 dK/dV |
|---|---|---|---|---|
| LB-only path it uses | partial dQ accum (cat-clone-add) | full-add | full-add | head-half pad-then-add |
| LB-ON cos_sim | 0.992 ✗ | 0.997-0.998 | 0.99999971 ✓ | 0.989-0.992 ✗ |

The earlier "H=1 chunk_3 grad_q_step is 1.35x of PT at step 0" observation was a downstream symptom of LB-specific accumulation, not a kernel bug. The kernel itself is correct (LB-OFF case proves it).

The Tensor library partial-update sanity test (`tensor_lib_partial_update_test.cpp`) already verified the `narrow_view + clone + cat` and `cat({step, zeros}) + grad_key` patterns produce bit-exact results in isolation. So the bug isn't in the ops themselves — it's in how they're invoked in the live `ContextParallelBackward::apply` (some condition the sanity test doesn't replicate: input tensors carrying grad_fn, non-default storage layout, async kernel pipelining, etc.).

## Next session actions
1. **Inspect LIVE invocation of the partial dQ accumulation** (`ContextParallelBackward.h` lines ~232-237) and the head-half pad-then-add (lines ~320-327). Suspect subtle interactions:
   - Input `grad_q` carries an autograd grad_fn from the prior `grad_q = grad_q + grad_q_step` at step 0. Maybe `narrow_view + clone` on a tensor-with-grad_fn behaves differently than the sanity test's grad-less tensor.
   - The K/V received via dkv_rotater at step 1 — verify the rotated bytes match expectations (PT and C++ should both have rank-1's local K/V at rank 0's step 1).
   - `__syncthreads` / cudaStream coordination: the partial dQ accumulation involves multiple intermediate tensors (`gq_1st.clone()`, `gq_2nd.clone() + grad_q_step`, `cat(...)`). If any of those alloc/launch on different streams or skip a sync, we could read stale memory.
2. **Add an in-CPB.h sanity check** before the partial accumulation: clone `grad_q` to a "shadow" tensor, do `_partial_update` on shadow via the same chunk-cat pattern in a different code path, diff against the production cat-clone-add result. If they differ → invocation-specific bug pinned.
3. After fix, rerun training to confirm norm-explosion at step 33 is resolved.

## Files Modified / Added
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/Pytorch/cp_bwd_isolated_test.py` (new)
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/Pytorch/cp_bwd_diff_finals.py` (new)
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/gpt2_cp_test/cp_bwd_isolated_test.cpp` (new)
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/gpt2_cp_test/tensor_lib_partial_update_test.cpp` (new)
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/Makefile` (added cp_bwd_isolated_test and tensor_lib_partial_update_test targets)
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/gpt2_cp_test/context_parallel/ContextParallelBackward.h` (extended DUMP_CP_STEPS instrumentation to 4 slices: H0/H1 × chunk_0/chunk_3)

## How to run
```bash
# PT side
cd /home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/Pytorch
torchrun --standalone --nnodes=1 --nproc-per-node=2 cp_bwd_isolated_test.py

# C++ side
cd /home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor
DUMP_CP_STEPS=1 mpirun -np 2 ./cp_bwd_isolated_test_exec

# Final dQ/dK/dV summary diff
python3 Pytorch/cp_bwd_diff_finals.py

# Per-step diff (focus on H1_chunk3_start lines for step_i=0 grad_q_step)
diff /tmp/cp_bwd_test/pt_step_bw_rank0.md step_bw_rank0.md
```
