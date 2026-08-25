# Claude Report — AllGather CP Validation-Loss Root Cause + Fix

- **Date/Time:** 2026-06-22 13:02
- **Work:** Root-cause and fix the AllGather-only context-parallel validation-loss degradation.
- **Workspace / File:** OldPush / `TensorParallelismBeta/DTensor/gpt2_cp_test/context_parallel/ContextParallelBackward.h`

---

## Symptom

With the C++ context-parallel GPT-2, the **AllGather** ring rotator produced steadily worse validation loss than PyTorch, while **P2P** and **AlltoAll** rotators matched (slightly beat) PyTorch. Reference runs (44M, HeadTail LB, ws=2):

- `CP_Training_logs/CP_Training_log165.csv` — P2P
- `CP_Training_logs/CP_Training_log166.csv` — AllGather
- `CP_Training_logs/CP_Training_log159.csv` — AlltoAll
- `Pytorch/.../Pytorch_CP_AttnStyle_FP32_Training_log45.csv` — PyTorch reference

## Evidence (val_loss every 100 steps)

| step | P2P(165) | AlltoAll(159) | AllGather(166) | PyTorch(45) | AllGather−P2P |
|---:|---:|---:|---:|---:|---:|
| 0 | 10.8755 | 10.8755 | 10.8738 | 10.8756 | ~0 |
| 1000 | 5.4097 | 5.4095 | 5.4605 | 5.5225 | +0.051 |
| 2000 | 4.7536 | 4.7530 | 4.8725 | 4.9376 | +0.119 |
| 4000 | 4.2720 | 4.2714 | 4.4147 | 4.4616 | +0.143 |
| 6767 | 4.0903 | 4.0897 | 4.2534 | 4.2295 | +0.163 |

- Step 0 (untrained; validation = pure forward) is identical across all rotators -> **forward is correct** for AllGather.
- The AllGather gap **grows monotonically** with training steps -> signature of a **backward/gradient bug**, not a forward bug (a forward bug would be a constant offset from step 0).
- P2P and AlltoAll track each other to ~5 decimals -> both correct rings.

## Root Cause

Backward dK/dV uses a **travelling accumulator**: a buffer rotated around the ring with the local contribution added each step (`ContextParallelBackward.h` ~L130, accumulation loop L455-L518). The rotator was selected by `rotator_type_` via `create_rotator()` (L729-740), so AllGather backward used `AllGatherRingRotator`.

`AllGatherRingRotator` (`RingRotator.h:303-376`) communicates **only on the first** `exchange_buffers` call (`if (!aggregated_buffer_.is_valid())`), caches a one-shot all_gather snapshot, and every subsequent call is a **no-op** serving cached slices. This is correct for the **read-only forward KV** ring (each shard read once) but **cannot carry a write-accumulate gradient buffer**: per-step dK/dV updates after step 0 are never re-communicated, so cross-rank dK/dV contributions are silently dropped -> wrong gradients -> progressively worse weights -> growing val-loss gap.

P2P (`ncclSend/Recv`) and AlltoAll (`alltoallv`) issue a real transfer on **every** step, so the mutated accumulator physically moves and sums correctly -> immune.

This matches PyTorch exactly: `_templated_ring_attention_backward` FORCES its `dkv_rotater` to `_RotateMethod.ALL_TO_ALL` regardless of the configured forward rotate method (core torch 2.9 `tensor/experimental/_attention.py`). The C++ port did not replicate that override.

## Fix

In `ContextParallelBackward.h` (~L156), force the backward `dkv_rotater` to a true per-step ring when forward uses AllGather; leave forward and the read-only `kv_rotater` (recompute_k) untouched:

```cpp
dkv_rotater = (rotator_type_ == 2 /*AllGather*/)
    ? std::make_unique<AlltoAllRingRotator>(pg_)
    : create_rotator();
```

`AlltoAllRingRotator` was already in scope (used by `create_rotator()` case 1). Chose AlltoAll to match PyTorch's forced ALL_TO_ALL; P2P would also be correct.

## Verification / Follow-ups (NOT done)

- Not rebuilt/re-run (no build executed this session). Next: rebuild the C++ CP target and re-run the AllGather config; expect step-0 val unchanged and the curve to now track P2P/AlltoAll.
- Optional confirm: `GRAD_PARITY_DUMP=1` step-0 dK/dV with AllGather should now match P2P/PT (previously dQ matched but dK/dV did not).
- Non-LB path (batch accumulators + single sendrecv) does not use the travelling-accumulator rotator and is unaffected.
