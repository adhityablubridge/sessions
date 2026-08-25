# Claude Report - CP Compute/Comm Overlap Divergence Fix

2026-06-11 - Root-caused and fixed the training divergence in the C++ Context-Parallel ring-attention compute/communication overlap - Workspace: TensorParallelismBeta/DTensor/gpt2_cp_test - Files: context_parallel/ContextParallel.h, context_parallel/ContextParallelBackward.h

## Problem

With compute/comm overlap enabled (dedicated non-blocking `cp_ring_stream_` + dedicated `cp_comm_` + persistent per-slot pack events + double-buffered rotators), CP training diverged: loss went erratic and grad-norm exploded mid-training, while the no-overlap path tracked PyTorch perfectly. Earlier fixes (dedicated cp_comm_ for PyTorch parity, dangling-view `.clone()`) reduced but did not eliminate it.

## Fast repro

Switched repro from 161M (~2 h) to 44M (~7 min): `CP_MODEL_44M=1 CP_WARMUP=40 CP_MAX_STEPS=480`, init `init_weights_named_44M.bin`, `mpirun -np 2`. Full overlap spiked deterministically at step ~417 (grad-norm 35.46). Note: mpirun under the agent harness requires the sandbox disabled to launch (IPC/network).

## Bisection (decisive)

Used the existing `CP_NO_OVERLAP_FWD` / `CP_NO_OVERLAP_BWD` gates:
- fwd-only overlap (`CP_NO_OVERLAP_BWD=1`): stable to step 479, max grad-norm 3.48
- bwd-only overlap (`CP_NO_OVERLAP_FWD=1`): stable to step 479, max grad-norm 3.41
- both overlap: diverges at step ~417, grad-norm 35

Conclusion: neither ring half is individually buggy. The corruption is an interaction that appears only when both forward and backward rings run async on the shared `cp_ring_stream_` / `cp_comm_`.

## Root cause

The forward ring loop (ContextParallel.h) and backward dKV ring loop (ContextParallelBackward.h) free their stack-local staging buffers (`send_buf[2]` / `dkv_send[2]`) and per-call rotators (`recv_[2]`, `pack_ev_[2]`) at function return, draining the last in-flight sends only with a GPU-side `streamWait` on the compute stream. The existing code comment documented the unsafe assumption: the GPU-side drain is sufficient only if the caching allocator reuses those freed blocks for writes issued on the compute stream.

Under full overlap that assumption breaks: the backward `dkv_rotater` issues NCCL writes from `cp_ring_stream_` (a different stream than the compute NULL stream). The caching allocator can hand a freed forward `send_buf`/`recv_` block to a backward buffer whose NCCL write comes from the ring stream, which the compute-stream ordering does not cover, racing the still-in-flight forward send that is reading it. This is a slow, rare buffer corruption -> the deterministic-onset, discrete grad-norm spike. With either half blocking, every reuse-write is issued from the compute stream, so the GPU-side drain holds -> only both-overlap fails.

## Fix

Changed both post-loop drains from GPU-side `->streamWait(compute_stream)` to CPU-blocking `->wait()`:
- ContextParallel.h post-loop drain (forward send_buf)
- ContextParallelBackward.h post-loop drain (dkv_send)

This guarantees the in-flight ring sends are complete before the stack-local staging buffers and per-call rotators free, regardless of which stream later reuses the memory. Also makes the per-call rotator `recv_`/`pack_ev_` teardown safe. Tail-overlap ceiling is negligible (only the final ring sends per pass cannot overlap downstream layers).

## Verification

Rebuilt (`rm gpt2_cp_test/gpt2_cp_test.o && make gpt2_cp_test`). Full-overlap 44M run (both rings async, confirmed via `[CP ring A2A] OVERLAP` + overlapped fwd/bwd timings) to step 479:
- max grad-norm (step>=50): 3.36 at step 155; no step>=50 exceeds 8
- step 417 (former divergence): grad-norm 0.38, loss 6.06 (was grad-norm 35)
- tail steps 476-479: loss ~6.0, grad-norm ~0.4 -- same regime as no-overlap / PyTorch baseline

Overlap is preserved (fwd ~270ms / bwd ~605ms with attn_cp overlapped) and divergence is eliminated.

## Follow-ups (not done)

- Confirm on 161M at real scale (definitive, ~2 h) and re-run the ws=2 step-0 grad parity gate (cosine ~1.0) to ensure numerics unchanged.
- Optional: if tail-overlap perf matters later, move staging buffers to a persistent/ref-counted pool so the pass can return with sends in flight (removes the CPU-wait ceiling).
