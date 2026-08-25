2026-05-16 - HeadTail CUDA kernel rewritten to chunk-level (PyTorch parity, Option 2 of plan i-want-the-shard-transient-whisper) - Workspace: OldPush/TensorParallelismBeta/DTensor, Files: tensor/headtail_kernel.cu, headtail_kernel.cuh, dtensor.cpp, gpt2_cp_test/context_parallel/ContextParallel.h

## Summary
Replaced the element-level HeadTail permutation (`out[2k]=in[k], out[2k+1]=in[T-1-k]`) with PyTorch's chunk-level semantics so the C++ codebase now has exactly ONE HeadTail definition across all paths (pre-embed shard helper + internal CP kernel).

## Step 0 audit (PREREQUISITE GATE) — PASSED
- `apply_lb_rule` in the original plan was a mislabel. The actual generic functions are `DTensor::context_parallel_shard` / `context_parallel_unshard` at dtensor.cpp:1546,1576.
- `grep` across `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta` confirms these are **defined but never called** from C++ (only a Python import reference in `gpt2_context_parallel.py`).
- All active `LoadBalancer::loadbalance/unloadbalance` callers are CP-pipeline internal:
  - `ContextParallel.h:245-247` (forward Q/K/V loadbalance, guarded by `!external_balanced`)
  - `ContextParallel.h:458` (post-allgather unloadbalance, guarded by `load_balance_ && !external_balanced`)
  - `ContextParallelBackward.h:395-397` (backward grad unloadbalance, same gate)
- None require element-level semantics; all expect PyTorch parity. Gate cleared.

## Changes

### Step 1+2 — `tensor/headtail_kernel.cu` (full rewrite)
- `headtail_loadbalance_kernel`: new `world_size` parameter. Replaces `seq_idx % 2 == 0` math with:
  - `t_local = seq_len / world_size; chunk_sz = t_local / 2`
  - `r = seq_idx / t_local; i = seq_idx % t_local`
  - `src = (i < chunk_sz) ? r*chunk_sz + i : (2N-1-r)*chunk_sz + (i - chunk_sz)`
- `headtail_unloadbalance_kernel`: matching inverse.
  - `c = dst_seq / chunk_sz; o = dst_seq % chunk_sz`
  - `if c<N: r=c, src = r*t_local + o; else: r=2N-1-c, src = r*t_local + chunk_sz + o`
- Launcher signatures gain `int64_t world_size` parameter.
- Added runtime guards in launchers (in this order): `assert(world_size > 0); assert(seq_len % (2*world_size) == 0)`.
- Updated `tensor/headtail_kernel.cuh` declarations to match.

### Step 3 — `tensor/dtensor.cpp` `HeadTail::loadbalance` / `unloadbalance`
- Pass `world_size` (inherited from `LoadBalancer` base) to launcher.
- Replaced the existing `seq_len % 2 != 0` precondition with `world_size > 0` + `seq_len % (2*world_size) != 0` checks, with clear error messages.

### Step 4 — Caller audit (verify only)
All four callers automatically receive chunk-level semantics; no caller code change required. API surface unchanged at the C++ level (only the internal CUDA launcher signature changed).

### Step 5 — Inline TODO documenting intentional duplication
Added `TODO(headtail-consolidation)` comment block above `perm_local` construction in `ContextParallel.h:91` warning future readers that the CPU-side perm_local and the CUDA `HeadTail::loadbalance` kernel implement identical chunk-level math and must stay in sync. Future cleanup may consolidate by calling `HeadTail::loadbalance` + `make_shards_inplace_axis` in place of the CPU loop.

## Sanity check (math)
- N=2, T=8: perm = [0,1, 6,7, 2,3, 4,5]; split-by-2 -> rank0=[0,1,6,7], rank1=[2,3,4,5]. Matches PyTorch.
- N=4, T=16: perm = [0,1, 14,15, 2,3, 12,13, 4,5, 10,11, 6,7, 8,9]. Matches PyTorch.
- N=1: identity permutation (LB is no-op for single rank, matches PyTorch).

## Build
- `make gpt2_cp_test` -> SUCCESS.

## Not yet done (follow-up)
- Self-inverse runtime unit check on real CUDA (defer to a small standalone harness if needed).
- Training-path regression run: since `external_balanced=true` short-circuits the internal kernel in the current training config (`cfg.load_balancing=true, cfg.cp_unshard=false`), this kernel rewrite has **no effect** on the active training path. Loss/grad must remain bitwise-identical.
- Phase 4 numerical validation harness (`cp_lb_causal_test.cpp`) and Phase 6 end-to-end loss-parity training run are still pending in the broader LB+causal plan.

## Files Modified
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/tensor/headtail_kernel.cu`
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/tensor/headtail_kernel.cuh`
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/tensor/dtensor.cpp`
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/gpt2_cp_test/context_parallel/ContextParallel.h` (TODO comment only)
