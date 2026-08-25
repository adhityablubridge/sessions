# Claude Report — bluscriptCP follow-ups (#1 no-overlap, #3 Ulysses, #4 full-scale) + sub-group PG device bug

2026-07-10 - Root-caused and fixed a real device-binding bug in the ncclCommSplit sub-group PGs; verified CP_NO_OVERLAP, Ulysses, and a full-scale (114M/T4096) run - Workspace: CP / Files: BluTrain/dist/communication/src/processGroupNccl.cpp, process_group/device_mesh.cpp

## The bug (one root cause behind two crashes)

`CP_NO_OVERLAP=1` (ring) crashed with "convert_type / NCCL enqueue: invalid resource handle", and
Ulysses mode crashed with "Synchronization failed line 484 / Cuda 400 invalid resource handle".
**Same root cause**, introduced by the `ncclCommSplit` migration:

- BluTrain's `ProcessGroupNCCL` derives its CUDA device as `rank_ % gpus_per_node_`, and the
  DeviceMesh passed each sub-group PG the **sub-group rank** (`my_group_rank`).
- For a 2-D mesh with a **size-1 DP axis**, that axis's `my_group_rank = 0`, so its adopt-ctor did
  `cudaSetDevice(0)`. The DeviceMesh then created the **CP axis's** `comm_stream` while the device
  was still 0 — but the CP `comm_` (from `ncclCommSplit`) lives on the process's real device (1 for
  rank 1). **Stream/comm device mismatch → NCCL enqueue "invalid resource handle."**
- Blocking / non-stream collectives (no-overlap ring `alltoallv_async`, Ulysses `alltoall`) run on
  that mis-bound `communication_stream_` and fail. The **overlap ring escaped it** only because
  `cpRingStream()` is created lazily at ring-time on the (correct) current device — which is why the
  default path worked and masked the bug.

This would have broken ALL multi-axis / 2-D DP×CP configs and Ulysses, not just the diagnostic path.

## The fix

- **`processGroupNccl.cpp` (adopt-ctor):** `cudaGetDevice(&local_rank_)` — adopt the device the
  caller already set (the process's global physical device); do NOT derive it from the sub-group
  `rank`. (`adopted_comm` from `ncclCommSplit` lives on the caller's current device, so this keeps
  stream / comm / later-collective device consistent.)
- **`device_mesh.cpp`:** re-assert `cudaSetDevice(global_device)` at the **start of each axis
  iteration**, before `ncclCommSplit` + `cudaStreamCreate`, so a prior axis's PG ctor can't leave a
  wrong device current.

## Verification (2× RTX 3060, sm_86)

- **Regression:** `cp-rope-standin` ws=2 still **12/12 PASS** (1-D mesh — fix is transparent there).
- **#1 `CP_NO_OVERLAP=1` ring (ws=2):** now trains, and the overlap A/B matches — step0 **identical**
  (10.884422 vs 10.884422), step4 differ 2e-5 (bf16 nondeterminism). Confirms the ring compute/comm
  **overlap is race-free** (the plan's concern (d), now via the proper A/B rather than only the
  serial-reference parity).
- **#3 Ulysses (`CP_ATTN_MODE=ulysses`, ws=2):** now trains, loss 10.890→10.228; step0 (10.889622)
  parity-matches world=1 (10.889627) — Ulysses gathers the full sequence so positions are 0..T-1,
  exactly single-GPU.
- **#4 Full-scale (default d768/L12, q12/kv4, hd64, ffn2048, T=4096, 114M params, CP_SIZE=2,
  T_local=2048/rank, ring):** trains 3 steps, loss **11.02 → 10.21** (decreasing), val 11.00→10.15,
  no OOM, no NaN. CP sequence-sharding lets T=4096 fit across two 12 GB GPUs — the point of CP.

## Deferred

- **#2 2-D DP×CP** (dp>1) run + the standalone `ncclCommSplit` per-axis all-reduce regression test —
  needs ≥4 ranks (only 2 GPUs here). NOTE: the device fix is a prerequisite for 2-D to work at all
  (it was the exact bug the size-1 DP axis exposed), so 2-D should be validated first on a 4-GPU box.
- Longer training / larger batch via grad-accum; Ulysses at full scale.
- Consider an env to disable checkpointing for smoke runs (auto-resume of a completed ckpt
  short-circuits reruns at the same `max_steps`).
