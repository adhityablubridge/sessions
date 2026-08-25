# Claude Report — Make AllGather CP Rotator Non-Blocking (Overlapped)

- **Date/Time:** 2026-06-22 15:38
- **Work:** Convert the AllGather ring rotator from a CPU-blocking collective to an async, compute-overlapped one (like PyTorch / the P2P/AlltoAll rotators), to remove the erratic-throughput rendezvous jitter.
- **Files:** OldPush / `process_group/ProcessGroupNCCL.h`, `process_group/processGroupNCCL.cpp`, `gpt2_cp_test/context_parallel/RingRotator.h`, `gpt2_cp_test/context_parallel/ContextParallel.h`

---

## Problem

After the two AllGather correctness fixes (backward dkv -> AlltoAll; forward re-gather), the loss matched, but AllGather throughput oscillated wildly (~130k–200k tok/s) while PyTorch/P2P/AlltoAll were flat (~185k). Cause: `AllGatherRingRotator` did a **CPU-blocking** `all_gather(sync=true)` on the **shared** comm and its `next_buffer_streamordered` fell back to the CPU-blocking path. A blocking collective is a barrier; step time = compute + collective + inter-rank wait, and the wait jitters with per-step GPU skew. P2P/AlltoAll (and PyTorch's AllGather) issue the collective async on a dedicated stream and wait GPU-side, hiding it.

## Changes (4)

1. **`ProcessGroupNCCL.h`** — declared `all_gather_async_stream(send, recv, sendcount, dtype, stream)`.
2. **`processGroupNCCL.cpp`** — implemented it: `ncclAllGather` on the given `stream` using the dedicated **`cp_comm_`**, via `launch_work_collectives(stream, ..., /*sync=*/false, cp_comm_)`, returns a `Work` (no CPU sync). Direct mirror of `alltoallv_async_stream`.
3. **`RingRotator.h`**
   - Added `virtual void reset() {}` to `RingRotatorBase` (re-arm a rotator for a new ring; no-op for P2P/AlltoAll).
   - Rewrote `AllGatherRingRotator`: overlap path issues the gather async on `cpRingStream()` with the same machinery as AlltoAll — persistent pack-event (`cudaEventRecord` on pack_stream, `cudaStreamWaitEvent` on ring stream), `recordStream` on send + aggregated buffers, store the `Work`. `next_buffer_streamordered` does `work_->streamWait(compute_stream)` then an async chunk copy on the compute stream. `reset()` invalidates the cached gather + resets `idx_` while keeping the buffer storage and pack-event. Kept a legacy blocking path for `CP_NO_OVERLAP`.
4. **`ContextParallel.h`** — replaced the previous "recreate the AllGather rotator each forward" fix with `kv_rotator->reset()` on the persistent instance each forward. This keeps the rotator persistent (no per-call allocation churn; no event-destroy UB) while still re-gathering fresh KV each step.

## How the overlap works / caveat

The gather is issued at the first exchange (i=0) and overlaps step 0's **local** SDPA (which uses local K/V, no comm). Steps 1..N-1 GPU-wait on it. Because AllGather is a single collective, it can only overlap the gather with the first SDPA — P2P/AlltoAll hide a hop behind **every** step and so overlap more. But this removes the **CPU-side blocking barrier** that caused the oscillation, so throughput should be much smoother (if not perfectly flat).

## Verification / follow-ups (NOT done here)

- Not built/run this session. Rebuild and:
  - A/B `CP_NO_OVERLAP=1` vs default to confirm overlap engages and is correct.
  - Run the race diagnostics already in the tree (`CP_SELFCHECK=1`, `CP_SYNC_RING=1`, `CP_FWD_SYNC_RECV=1`) on the AllGather config to validate no recv->SDPA / buffer-reuse race was introduced.
  - Confirm loss still matches PyTorch (overlap must not change numerics) and that the throughput curve flattens.
- All AllGather fixes now in tree: backward dkv->AlltoAll (`ContextParallelBackward.h:156`), forward re-arm via reset() (`ContextParallel.h`), async overlap (`RingRotator.h` + PG async-stream all_gather).
