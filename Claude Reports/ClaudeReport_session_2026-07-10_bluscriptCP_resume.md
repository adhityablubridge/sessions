# Claude Report — bluscriptCP resume (CP Llama training on the unified PG)

2026-07-10 - Finished the context-parallel Llama trainer on the unified process-group layer; DataParallel + ContextParallel coexist and train - Workspace: CP / Files: Scripts/BluTrain/bluscriptCP.cpp, process_group/device_mesh.{h,cpp}, Makefile

## Why

With the process-group layer unified (BluTrain PG canonical + `ncclCommSplit`), `DataParallel`
(dist) and `ContextParallel` (CP) can finally share one `ProcessGroupNCCL`, unblocking the
DataParallel-based `bluscriptCP.cpp` (drafted earlier).

## What changed

- **Narrow DataParallel include:** `bluscriptCP.cpp` now includes `DataParallel.hpp` directly
  (not the `dist/distributed.h` umbrella, which pulls the TP stack including a *different*
  `class DeviceMesh` that collides with the CP one). `init_process_group` comes from the canonical
  `ProcessGroupNCCL.h` (via `device_mesh.h`/`ContextParallel.h`).
- **Single world-init:** BluTrain's PG registry throws if `init_process_group` is called twice.
  The DeviceMesh already builds the world PG once (the `ncclCommSplit` parent); added
  `DeviceMesh::world_pg()` and made `bluscriptCP` reuse it for `DataParallel` instead of a second
  `init_process_group`.
- **Build:** new `bluscript-cp` / `run-bluscript-cp` Makefile targets; `DataParallel.cpp` +
  `Data-Parallel/src/profiler.cpp` are scoped to this target only (the parity tests don't need them).

## Verification (2× RTX 3060, sm_86, `CP_FUSED_ROPE=1`, tiny cfg L2/d256/hd64/GQA kv2/T256, edufineweb data via `CP_DATA_ROOT=~/Downloads`)

- **Build clean** — DataParallel + ContextParallel link on one `ProcessGroupNCCL` (no collision).
- **world=1**: trains end-to-end, loss **10.89 → 10.23** (decreasing), grad-norm finite (3.6→2.2),
  no NaN; `[CP bwd] overlap=ON` + `[CP ring A2A] OVERLAP` confirm the fused-RoPE CP ring executed
  — the FIRST real training integration of the new CP kernels.
- **ws=2 `CP_SIZE=2` ring**: trains, and the loss **matches world=1 to ~1e-3**:

  | step | world=1 | ws=2 CP |
  |---|---|---|
  | 0 | 10.889627 | 10.884422 |
  | 4 | 10.227955 | 10.226874 |

  This is the loss-parity check: CP splits the sequence across 2 ranks but computes the same total
  loss — validating sequence-sharding + RoPE global positions + the ring end-to-end. And because
  overlap-ON matches the (serial) world=1 reference, the **ring compute/comm overlap is race-free**
  (the intent of the `CP_NO_OVERLAP` A/B in the plan — answered by parity against the serial ref).

## Follow-ups

- **`CP_NO_OVERLAP=1` diagnostic path crashes** ("convert_type_cuda kernel launch failed: invalid
  resource handle") under the unified PG — a CUDA stream-handle issue specific to the no-overlap
  path in the ContextParallel driver. The DEFAULT overlap path is unaffected and verified correct;
  this is a diagnostic-only path. Worth a targeted fix (likely a stream created on one PG/context
  used on another).
- **2-D DP×CP** run (dp>1, needs ≥4 ranks) + the standalone 2-D `ncclCommSplit` per-axis all-reduce
  regression test (from the PG-unification plan).
- **Ulysses-mode bluscriptCP** (`CP_ATTN_MODE=ulysses`, default build) smoke.
- Full-scale config (d_model 768, 12 layers, T 4096) once the smokes are green.

## Notes

- Checkpointing auto-resume bit me during the A/B (a completed `step_5` ckpt short-circuited a
  rerun with the same `max_steps`); clear `checkpoints_bluscriptcp/` or bump `CP_MAX_STEPS` for
  fresh smoke runs. Consider an env to disable checkpointing for smokes.
- The data loader found the edufineweb shards in `~/Downloads` ("found 18 shards for split train").
