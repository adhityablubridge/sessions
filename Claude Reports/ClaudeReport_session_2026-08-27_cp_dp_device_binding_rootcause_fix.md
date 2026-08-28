# CP dp>1 gradient corruption - root cause and fix

2026-08-27 - Root-caused and fixed the bluscriptCP multi-GPU gradient corruption; the
NCCL process group derived its CUDA device from the process-group rank instead of the
physical device. Workspace: BluTrain / dist/Context_Parallelism, dist/communication.

## Correction to the previous report

The earlier report of this bug attributed it to `dist/Data-Parallel`. That was wrong.
`DataParallel` is fully exonerated: the corruption reproduces with `DataParallel` never
constructed at all.

## Root cause

`dist/communication/src/processGroupNccl.cpp`

    local_rank_ = rank % gpus_per_node_;
    CUDACHECK(cudaSetDevice(local_rank_));

`rank` here is the **process-group** rank, not the physical device. `local_rank_` was then
used by `cudaSetDevice` in the constructor and in six collective entry points
(lines 214, 548, 591, 759, 773, 861).

`DeviceMesh` builds one process group per mesh axis. With `mesh_shape = {dp_size, cp_size}`
and `cp_size == 1`, the CP-axis group has **group-rank 0 on every process**. So on global
rank 1 the CP-axis PG bound the process to device 0 and every collective on that group
switched the calling process off its own GPU.

This is invisible whenever group-rank happens to equal the physical device - a 1-D world,
i.e. pure CP (`CP_SIZE == world_size`) or plain single-axis DDP - and corrupts everything
as soon as they differ, which is exactly `dp_size > 1`. That matches the observed pattern
precisely: pure CP healthy at 2 and 4 ranks, anything with `dp_size > 1` broken.

`device_mesh.cpp:127` already documents the intended contract - "the upstream base PG ctor
now ADOPTS the current device (CP device-binding fix), so it MUST be set correctly before
every PG construction" - and `initialize_process_groups` does set the physical device
immediately before constructing each PG. The contract simply was never honoured in the
constructor.

`init_process_group` had the same defect at line 387 (`cudaSetDevice(rank)` when creating
the comm stream).

## Fix

Both sites now adopt the device the caller has already made current:

    int adopted_device = 0;
    CUDACHECK(cudaGetDevice(&adopted_device));
    local_rank_ = adopted_device;

Backup of the original: `dist/communication/src/processGroupNccl.cpp.prefix.bak`.

## Evidence chain

Each step below was measured, not inferred.

1. ZeRO at dp=2 shows the same corruption, and ZeRO never constructs `DataParallel`
   (`if (world_size > 1 && !cfg.zero)`). The reported ZeRO norm is real - `grad_norm` is
   overwritten in the optimizer block with `sqrt(sum last_grad_norm^2)`, which is why the
   probe placed before that block printed 0.0000.
2. Added `CP_NO_DDP` to skip `DataParallel` construction entirely. At dp=2 with no reducer
   of any kind, no gradient sync, and `CP_SAME_DATA=1`:
   rank 0 norm 53.7795, rank 1 norm 9,217,563.
3. `CP_FWD_PROBE` confirmed the forward is **bitwise** identical on both ranks - loss bits
   `0x413291b2`, logits `sum=447879.49947222148`, `sumsq=126439429.08425011` to 17 digits.
   So the divergence is entirely in the backward, on bit-identical inputs.
4. `CP_GRAD_PROBE` showed every parameter blown up on rank 1 including `param[0]`
   (embedding), not one layer - consistent with a wrong-device execution, not bad math.
5. Rank-dependent, not device-dependent: physical GPU 1 alone is healthy (53.7807), and on
   GPUs 2,3 rank 1 is still garbage.
6. Not attention-mode specific: ring+split 9.2e6, ring+fused 1.5e7.
7. Deterministic (9.21e6 reproducible across runs), so not a race.

## Why it broke in some topologies and not others (predicted, then measured)

`DeviceMesh` lays out `mesh_shape = {dp_size, cp_size}` with CP as the fastest axis, so for
global rank `r`:

    cp_coord = r % cp_size          (the CP-axis group rank)
    dp_coord = r / cp_size          (the DP-axis group rank)
    correct device = r % gpus_per_node

The buggy line bound the device to the **group** rank, so a rank is corrupted exactly when
`cp_coord != r` for the CP-axis PG - the one that is used by the ring machinery every
microstep. Ranks where they coincide are accidentally correct.

The CP-axis PG is also constructed last in the axis loop, so its wrong `cudaSetDevice`
is what the process is left sitting on when training begins.

| topology | cp_coord per rank | correct device | predicted broken | measured (pre-fix, no reducer) |
|---|---|---|---|---|
| dp=1 cp=4 (pure CP) | 0,1,2,3 | 0,1,2,3 | none | 117.93 / 51.49 / 57.83 / 53.26 - all healthy |
| dp=2 cp=2 | 0,1,0,1 | 0,1,2,3 | ranks 2,3 | 86.48 / 41.26 / **1,991,361** / **1,243,923** |
| dp=4 cp=1 | 0,0,0,0 | 0,1,2,3 | ranks 1,2,3 | 53.78 / **9.2e6** / **9.2e6** / **9.2e6** |

All three match the prediction exactly. This is also why pure CP looked healthy throughout
the earlier investigation and why the bug tracked `dp_size > 1` rather than any particular
reducer: at `cp_size == world_size`, `cp_coord == r` identically.

Post-fix, `dp=2 cp=2` shows the two DP replicas agreeing per CP coordinate - ranks 0,2 =
86.4838/86.4851 and ranks 1,3 = 41.2616/41.2618 - which is the correct signature (the two
CP coordinates hold different sequence halves, so their local gradients differ; the two DP
replicas on the same data must match).

The corruption is silent rather than a crash because on an NVLink node with peer access
enabled a kernel launched on device 0 can dereference device-1 addresses without error. One
detail I did not pin down: which specific backward op inherits the ambient current device
while the whole forward stays correct (the forward is bitwise identical, so its ops evidently
set the device from their tensors). Establishing that was not needed to fix it, and it is the
natural follow-up if the pattern ever resurfaces.

## Verification after the fix

All ranks now agree exactly, in every topology.

| topology | before | after |
|---|---|---|
| np=1  dp=1 cp=1 | 53.78 | 53.78 |
| np=2  dp=2 cp=1, no reducer | 53.78 / 9,217,563 | 53.7798 / 53.7803 |
| np=2  dp=2 cp=1, DDP on | 53.78 / 19,822,810 | 34.7563 on both |
| np=2  dp=1 cp=2, pure CP | healthy | 44.3964 on both |
| np=4  dp=2 cp=2, DP x CP | ranks 2,3 = 1,991,361 / 1,243,923 | 34.6699 on all four |
| np=4  dp=4 cp=1 | ranks 1,2,3 = 9.2e6 | 53.7808 / 53.7805 / 53.7801 / 53.7802 |
| np=4  dp=1 cp=4, pure CP | healthy | 63.2832 on all four |
| ZeRO v1 bf16, dp=2 | 6,865,667 | 41.3390 -> 46.8457 |
| ZeRO v2 bf16, dp=2 | 6,865,693 | 41.3369 -> 46.8404 |
| Muon + ZeRO v2 bf16, dp=2 | broken | 41.3350 -> 46.4402 |

bf16 now works in these configurations; the `convert_cuda_bf16_to_fp32_sm90` "no kernel
image" error was a downstream symptom of the wrong-device launches, not a cause. Verified
separately that the converter kernel runs correctly standalone against `libtensor.so`, and
that both `sm_90` and `sm_90a` cubins load on this H100 - so TI's wrapper was reporting a
sticky error raised by an earlier failed launch.

## Second bug: Ulysses / hybrid dead on Hopper (fixed)

Independent of the device-binding bug above. `CP_ATTN_MODE=ulysses` failed in every
configuration including a single process at np=1, and so did `hybrid` (it uses the same
all-to-all leg).

Cause: `ContextParallel.h:1863`, in `forward_ulysses_fused`, calls libtensor's
`OwnTensor::cuda::gqa_fused_flash_attn_forward`. That entry point has exactly one definition,
`src/Kernels/cuda/attention/arch/GQA_fused_fwd_sm103.cu`, and it is built **only** for
`sm_100a` / `sm_103a` - Blackwell. On sm_90 the launch can only fail with "no kernel image is
available for execution on the device". TI then reports that sticky error at the *next* error
check, which is the `out_bf.as_type(Float32)` on line 1872 - so it surfaced as a bogus
`convert_cuda_bf16_to_fp32_sm90 kernel launch failed`. That misattribution is what made this
look like a dtype/converter problem for so long; the converter is fine and runs correctly
standalone.

Localised by `addr2line` on the exe frames in the abort backtrace:
`forward_ulysses_fused` (ContextParallel.h:1878) <- `forward_ulysses` (:1519) <-
`forward_cp` (:427) <- `CausalGQA::forward` (bluscriptCP.cpp:446).

The working kernel already existed: the `CP_ULYSSES_CPLOCAL` leg (`ContextParallel.h:1797`)
routes to `cp::cuda::gqa_fused_rope_cp_forward`, a WMMA fork of the same flash core that
compiles hd 64 and 128 and is arch-portable. Its own comment says it "is the only way Ulysses
runs on Hopper at all" - but it was gated behind an env var that defaulted OFF, so the
default path on Hopper was the one that cannot work.

Fix (`bluscriptCP.cpp:842`): decide by arch instead of by env. Query
`cudaDevAttrComputeCapabilityMajor` and default `ulysses_cplocal = (major < 10)`.
`CP_ULYSSES_CPLOCAL` still overrides in both directions, so `=0` forces the libtensor leg for
Blackwell parity testing. The run now prints which leg it took.

Verified with no special env, all ranks agreeing and loss descending:

| topology | step 0 | step 1 |
|---|---|---|
| ulysses np=1 dp=1 cp=1 | 53.7798 | 53.1436 |
| ulysses np=2 dp=1 cp=2 | 51.5780 both | 39.7878 both |
| ulysses np=4 dp=1 cp=4 | 49.1277 all four | 38.1417 all four |
| ulysses np=4 dp=2 cp=2 | 34.1118 all four | 41.4789 all four |
| hybrid np=4 ring=2 x ulysses=2 | 51.1102 all four | 60.7012 all four |
| ulysses + ZeRO v2 + bf16 + Muon, np=4 dp=2 cp=2 | 41.3390 all four | 46.4277 all four |

Note ulysses at np=1 gives 53.7798, bit-for-bit the ring np=1 baseline - the expected
cross-check, since at cp_size=1 both degenerate to the same computation.

## Still outstanding

1. TI error handling converts failed launches into misattributed errors. `ConversionKernels`
   reports a sticky error as its own; `Views/arch/ContiguousKernel_sm90.cu` prints on
   `cudaGetLastError() != cudaSuccess` and then returns `true`, turning a launch failure into
   silent corruption. Both cost significant debugging time here.
2. `bluscriptCP.cpp` binds `DeviceIndex device(Device::CUDA, rank)` from the **global** rank,
   ignoring `gpus_per_node`. Correct on one node, wrong on multi-node.
3. CP's own unlinked copy `process_group/processGroupNCCL.cpp` has the identical
   `local_rank_ = rank % gpus_per_node_` defect at lines 43-44 and in its collectives. Not
   compiled into `bluscriptCP_exec`, so it is latent, but it should be fixed in step with
   the shared one.
4. `make bluscript-cp` must be invoked as `make CP_FUSED_ROPE=1 CP_ATTN_SPLIT=1 bluscript-cp`.
   Building without those defines produces a binary that fails at np=1; the defines are
   compile-time gates while `CP_ATTN_FUSION=split` is runtime-only.
5. The four `dist/Data-Parallel` defects found by reading in the previous session are still
   genuine and still unfixed (`init_sync` mixed-dtype broadcast, dead `naive_grad_sync` with
   a local-vector write-back bug, `opts_.bucket_` never read). They are not this bug. The
   `cudaDeviceSynchronize` added to `finalize_backward` (its own comment mandates it) is
   still applied and can be reverted if unwanted.

## Debug instrumentation left in place (all env-gated, inert by default)

`bluscriptCP.cpp`: `CP_RANK_NORM` (per-rank norm/device/loss, plus a ZeRO-branch norm),
`CP_GRAD_PROBE` (per-parameter grad norms on every rank), `CP_FWD_PROBE` (raw loss bits +
logits checksum), `CP_NO_DDP` (skip `DataParallel` construction), `CP_NO_VAL` (skip
validation), `CP_SAME_DATA` (now applies to the val loader too, not just train),
`CP_DEV_OVERRIDE` (pin all ranks to one device), `CP_DDP_NO_REDUCE`, `CP_DDP_NO_BUCKET`,
`CP_DDP_NO_GRADVIEW`. `device_mesh.cpp`: `CP_DEV_OVERRIDE`.

Backups: `/tmp/bcp.bak`, `/tmp/bcp.pre_val.bak`,
`dist/communication/src/processGroupNccl.cpp.prefix.bak`.

## Note

`/root/claude-vault` is on the remote instance and has not been synced off. The previous
instance's report was lost this way.

## Operational note - disk

`CP_CKPT` defaults to ON, and a checkpoint is written at the final step regardless of
`CP_CKPT_FREQ`. Every one of the ~20 short diagnostic runs above therefore dumped a 7.6G
checkpoint, filling the 193G root filesystem to 100% mid-session and killing two runs with
ENOSPC. All 20 were `step_1`/`step_2` files from these diagnostics - no real training
checkpoint existed in `checkpoints_bluscriptcp` - and they were deleted, recovering 136G
(now 64G used / 110G free). Pass `CP_CKPT=0` on any short diagnostic run.
