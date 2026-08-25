2026-05-16 - LB+causal gate untangled in ContextParallel forward/backward (Phase 1 of plan i-want-the-shard-transient-whisper) - Workspace: OldPush/TensorParallelismBeta/DTensor, Files: gpt2_cp_test/context_parallel/ContextParallel.h, ContextParallelBackward.h

## Summary (Phase 1 of multi-phase plan)
- `ContextParallel.h::forward_cp`: replaced single `lb_active = load_balance_ && !is_causal_` gate with two independent flags:
  - `external_balanced = pre_sharded && load_balance_` (skip CP-internal HeadTail kernel; pre-embedding shard already did it)
  - `sub_chunk_active = load_balance_` (drive round-robin Q/K/V sub-chunking + partial-merger + dkv_rotater regardless of is_causal_)
- Internal `loadbalancer_.loadbalance()` is now guarded on `load_balance_ && !external_balanced` (forward) and the post-allgather `unloadbalance` mirrors that gate.
- Causal-dispatch / sub-chunking / q_off-k_off branches inside the ring loop now read `sub_chunk_active` (was `lb_active`).
- `ContextParallelBackward` ctor takes two new args (`sub_chunk_active`, `external_balanced`); backward's sub-chunk dispatch reads `sub_chunk_active_` instead of `load_balance_`, and the final `unloadbalance` on all-gathered grads mirrors the forward's external_balanced gate. Added a runtime check that T_local is even when sub_chunk_active.
- Added PyTorch dispatch-table comment block at the top of forward_cp's Phase 1 section for future readers.
- Build verified: `make gpt2_cp_test` -> SUCCESS.

## Effect at runtime
- For training path with `cfg.load_balancing=true`, `cfg.cp_unshard=false`: pre-embedding HeadTail shard runs, CP sees `pre_sharded=true && load_balance_=true` -> `external_balanced=true` (skip internal HeadTail kernel), `sub_chunk_active=true` (do round-robin Q/K/V halving + partial-merger + dkv_rotater). Causal masking dispatched per the PyTorch table (i==0 IS_CAUSAL, others NOT_CAUSAL, never SKIP).
- For training path with `cfg.load_balancing=false`: behavior unchanged from previous PR (contiguous shard, no sub-chunking).
- For generation (`pre_sharded=false`) with `cfg.load_balancing=true`: CP applies its own internal HeadTail then sub-chunks. (Not currently exercised by the training script; included for completeness.)

## Not yet done (subsequent phases in plan)
- Phase 4: numerical validation harness — new `cp_lb_causal_test.cpp` comparing dQ/dK/dV against a single-GPU reference and a PyTorch `enable_load_balance=True` baseline.
- Phase 5: debug-and-fix any divergence found.
- Phase 6: enable in training script and capture loss-curve parity.
Awaiting user direction before proceeding (validation work is expensive and may surface real bugs in the existing sub-chunk machinery that the gate had been hiding).

## Files Modified
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/gpt2_cp_test/context_parallel/ContextParallel.h`
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/gpt2_cp_test/context_parallel/ContextParallelBackward.h`
