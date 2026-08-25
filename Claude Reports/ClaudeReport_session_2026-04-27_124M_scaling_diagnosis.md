# Claude Report — 2026-04-27 — 124M scaling diagnosis (C++ vs PyTorch CP)

**Workspace:** TensorParallelismBeta
**Files reviewed:** `DTensor/gpt2_cp_test/gpt2_cp_test.cpp`, `DTensor/Pytorch/gpt2_cp_headtail_fp32.py`
**Symptom:** 44M parity ~64k tok/s; 124M C++ = 12k vs PyTorch 30k.

## Top causes (high confidence, in tagged files)

1. **`optimizer.step()` runs twice per step.** Diagnostic block at `gpt2_cp_test.cpp:841-848` calls full Adam pre-forward (only at step>=1), the real call is at line 1019. At 124M, optimizer is a much larger fraction of step time → ~2× penalty here that 44M didn't show.
2. **`cudaDeviceSynchronize()` calls outside timers.** Lines 843, 848, 1017 drain the GPU queue → no overlap of bwd-tail with clip/optim dispatch. PyTorch syncs only inside its CudaTimer (same as our `CudaTimer`).
3. **MultiTensorKernels `scale` and `grad_norm` were unoptimized** (per-element binary search, no float4) — disproportionate cost as total numel grows. Replaced today with chunked + float4 design ported from the `_sm89` kernels.
4. **Optimizer launch count scales with total numel** (`MAX_BLOCKS_PER_LAUNCH=320`, ~12 launches/Adam-call at 124M vs ~4 at 44M). Combined with #1 → 24 launches/step in optim alone.
5. **Per-step debug prints** at lines 989-992 and 1021-1024 add host-side stalls — constant overhead, larger fraction at 124M.

## Suspect but need more files

- `ContextParallel::forward_cp` P2P overlap behavior at 2× KV-rotate bytes (n_embd 384→768).
- GEMM dispatcher selection for `[B*T_local, 768] @ [768, 50304]` lm_head — memory-bound shape.
- Whether grad all-reduce is overlapped with the backward tail.

## Recommended fixes

In order of expected gain:
1. Delete the diagnostic pre-forward `optimizer.step()` block (`gpt2_cp_test.cpp:836-848`).
2. Remove the bare `cudaDeviceSynchronize()` at line 1017 (and the syncs around the pre-optim block become moot once #1 is done).
3. Rebuild with today's MultiTensorKernels rewrite. Confirm `clip_*` and `optim_*` columns drop in the CSV.
4. Gate per-step debug prints behind `step % 100 == 0`.
5. Re-measure. If a gap remains, look at `ContextParallel.{h,cu}` and the GEMM path.

## Files modified

None this session for this diagnosis. (Earlier today: `MultiTensorKernels.cu` was rewritten with chunked + float4; backup at `MultiTensorKernels.cu.bak_2026-04-27`.)
