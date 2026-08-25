# Claude Report — CP compatibility with upgraded BluTrain/TI, Stage 1 (process group)

2026-08-18 - Restored the CP context-parallel stack on the upgraded BluTrain/Tensor-Implementations by subclassing the frozen upstream ProcessGroupNCCL (1 pre-authorized upstream edit); verified via the full parity suite on 2x3060 - Workspace: CP

## Why
BluTrain + Tensor-Implementations were fast-forwarded to latest upstream. Three breakages surfaced:
1. **Process group (forced):** the upstream `ProcessGroupNCCL` no longer has the CP ring-overlap surface
   (`cpRingStream`, `*_async_stream`, `alltoallv_async`, adopt-ctor, `get_comm`), and its `comm_`/streams
   are private -> `device_mesh.cpp` + `RingRotator.h` would not build -> all CP modes blocked.
2. **Attention arch:** upstream moved to standalone QK-norm+RoPE + separate attention (old fused kernel
   still present, additive) -> the team's forward direction (Stages 2-4, pending).
3. **Checkpointer removed:** `checkpointing/Checkpointing.h` / `CheckpointManager` (bluscriptCP save/resume)
   is gone (new `Checkpoint.h` is activation checkpointing) -> blocks the full bluscriptCP build.

Plan (approved, hardened over 3 plancritique rounds): `~/.claude/plans/context-we-as-a-delegated-lerdorf.md`.

## Stage 1 — done + verified
- **1 upstream edit** (`BluTrain/dist/communication/src/processGroupNccl.cpp`): base ctor device
  `cudaGetDevice(&local_rank_)` instead of `rank % gpus_per_node_` (the log-150 fix) so CP sub-group PGs
  bind the right GPU for 2-D DP×CP. This was the single "unavoidable" edit the user pre-authorized.
- **`process_group/CPProcessGroupNCCL.{h,cpp}`** (new): `class CPProcessGroupNCCL : public ProcessGroupNCCL`.
  - Base built over the CP group via the existing public ctor -> inherited collectives operate on the group.
  - Ring surface on a dedicated `cp_comm_` + non-blocking `cp_ring_stream_`, via an **in-header**
    `launch_ring_work` (the base `launch_work_collectives` is a template *defined in the base .cpp* -> a
    subclass TU can't instantiate it, and it binds the Work to the private base `comm_`). Fatal failure
    path: `ncclCommAbort(cp_comm_)` + `MPI_Abort` (ring is intentionally un-watchdog'd).
  - Non-virtual base dtor -> documented `make_shared`/`shared_ptr<CPProcessGroupNCCL>` lifetime invariant.
- **`process_group/device_mesh.{h,cpp}`** rewritten: per axis `MPI_Comm_split` + two group-scoped
  `ncclUniqueId`s broadcast **within the sub-comm** (fixes the old 2-D wrong-id bug), named per-axis
  stream, deterministic two-comm init order. No `ncclCommSplit`/`get_comm`/adopt-ctor.
- **Type propagation:** `CPProcessGroupNCCL` through `RingRotator`/`ContextParallel`/`ContextParallelBackward`/
  `bluscriptCP` + `cp_rope_standin`/`cp_ulysses` tests; include path `process_group/CPProcessGroupNCCL.h`.
  Ulysses/DP stay on the base type (implicit upcast). No Makefile change (auto-discovered source).

## Verification (2x3060 sm86, libtensor rebuilt vs new TI, clean link)
- `cp-rope-standin` ws=2: **ALL PASS** cos=1.0000000 (fwd + dQ/dK/dV/dq_gamma/dk_gamma; contiguous + HeadTail).
- `cp-rope-fused` ws=2: **ALL PASS** (identity maxdiff=0.0; HeadTail seam rows exact).
- `cp-ulysses` ws=2: Ulysses path ran clean; MHA all PASS; FUSED-vs-plain-GQA all PASS (cos>0.9999);
  the only FAILs are the **pre-existing** GQA-backward-vs-fp32-ref bf16 precision artifacts (documented
  before the upgrade; unchanged by this work).

Net: the subclass + device_mesh rewrite restore CP ring AND Ulysses on the new upstream with zero new
failures and exactly one upstream line changed.

## Outstanding
- **Checkpointer:** decided to vendor the git-recovered pre-upgrade `checkpointing/Checkpointing.h`
  (+ `.cpp`) into the CP repo. Awaiting the user's `git show <old-ref>:...Checkpointing.h`. Until then
  the full bluscriptCP build (and the Stage-1 e2e ws=2 loss-parity smoke) is blocked; the PG fix itself is
  already verified via the parity tests.
- **Stages 2-4:** standalone QK-norm+RoPE CP kernel fork + portable separate causal attention (strip
  `norm_rope_tile` from the `_cp` fork) + Model-A wiring behind `CP_ATTN_FUSION`, additive with the fused
  path as fallback. Pending.
