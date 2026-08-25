# Claude Report — Process-group unification (BluTrain PG canonical + ncclCommSplit)

2026-07-10 - Unified the process-group layer so DataParallel and ContextParallel share one ProcessGroupNCCL - Workspace: CP / Files: BluTrain/dist/communication/{include/ProcessGroupNCCL.h, src/processGroupNccl.cpp}, process_group/device_mesh.{h,cpp}, context_parallel/*.h, Scripts/BluTrain/gpt2_cp_test.cpp, Makefile

## Why

`bluscriptCP.cpp` needs `DataParallel` (BluTrain dist) AND `ContextParallel` (CP) in one TU, but
the CP repo had forked its own `ProcessGroupNCCL` that diverged from dist's — two different
`class ProcessGroupNCCL` (same name/scope) + duplicate `NCCLCHECK` macro → redefinition. Goal:
one canonical PG for CP + DP + future parallelism. Chosen: **BluTrain's PG canonical**, sub-groups
via **`ncclCommSplit`** (NCCL-native; also fixes the 2-D-mesh wrong-NCCL-id bug where every axis PG
was bootstrapped over `MPI_COMM_WORLD`). Premise corrected during exploration: `sub_group.h` was
NOT NCCL-only (it `MPI_Bcast`es an id); both PGs were MPI-bootstrapped and the CP one was actually
a superset (it had the ring-overlap methods BluTrain lacked).

## What changed

- **A — extend BluTrain PG (additive):** ported the CP ring-overlap surface to run on the PG's own
  `comm_` (which, for a sub-group PG from `ncclCommSplit`, IS the CP-axis comm — so CP's separate
  `cp_comm_` workaround is dropped): `cpRingStream()` + `cp_ring_stream_`, `sendrecv_async_stream`,
  `all_gather_async_stream`, `alltoallv_async`, `alltoallv_async_stream`. Added a **comm-adopting
  ctor** `(world_size, rank, ncclComm_t adopted_comm, work, stream, root)` and `get_comm()`. All
  additive — DataParallel (already on this PG) unaffected.
- **B — `ncclCommSplit` sub-groups:** rewrote `process_group/device_mesh.cpp`
  `initialize_process_groups()` to build ONE world PG (`init_process_group`, one `MPI_Bcast`) then,
  per axis, `ncclCommSplit(world_comm, color=other-coords, key=axis-coord)` → wrap the split comm in
  a BluTrain PG via the adopt-ctor. Replaces `MPI_Comm_split` + per-axis WORLD id-bcast; correct
  for N-D meshes. `device_mesh.h` includes the canonical PG + holds `world_pg_`.
- **C — migrate CP includes:** repointed 8 files' `#include "process_group/ProcessGroupNCCL.h"` →
  `"ProcessGroupNCCL.h"` (canonical, via `-I`). **Collision fix (rename, not delete, per user):**
  `device_mesh.h`'s quote-include resolved CP's *same-directory* header first, pulling both PGs into
  one TU → renamed `process_group/ProcessGroupNCCL.h` → `ProcessGroupNCCL_cplegacy.h` (dormant on
  disk). `CpuSync_fixed.hpp` (CP's `Work`) is now included by nothing.
- **D — Makefile:** `INCLUDES += -IBluTrain/dist/communication/include -IBluTrain/dist/Data-Parallel/
  include`; `BLUTRAIN_PG_SRC = processGroupNccl.cpp + Data-Parallel/src/Error_logs.cpp` (the PG pulls
  the dist `Error`/`cond_check_fail` infra) added to `CPP_SOURCES`; `CPP_EXCLUDE +=
  process_group/processGroupNCCL.cpp`. Result: exactly one `ProcessGroupNCCL`/`Work`/`NCCLCHECK`.

## Verification (2× RTX 3060, sm_86)

- **Build clean** (exit 0): no redefinition, no duplicate symbols; link line shows
  `processGroupNccl.o` + `Error_logs.o` + `device_mesh.o`, CP's `processGroupNCCL.o` excluded.
- **cp-rope-standin (ring: fwd + dQ/dK/dV/dq_gamma/dk_gamma): 12/12 PASS, cos=1.0000000** —
  the ported ring-overlap methods (now on BluTrain's PG) + the `ncclCommSplit` 1-D world sub-group
  behave identically to the old CP PG.
- **cp-ulysses (all-to-all + Ulysses): 72 PASS**, only the 4 pre-existing `normal vs ref_sdpa
  dQ/dq_gamma` failures — the bf16-vs-fp32 test-reference precision artifact (already established as
  NOT a kernel bug), unchanged by the migration.

## Follow-ups (not done this session)

- **2-D `ncclCommSplit` regression test**: build `DeviceMesh({2,2},…)` on 4 ranks, `all_reduce` a
  vector over `get_process_group(0)` (DP axis) and `(1)` (CP axis), assert each reduces over exactly
  its axis members — the direct regression test for the old wrong-id bug + the new capability.
  (Needs 4 ranks or oversubscription.)
- **Resume `bluscriptCP.cpp`**: now that the PG is unified, wire `DataParallel.cpp` (+ its dist
  deps) into the build and finish the DataParallel-based bluscriptCP (draft already written); then
  its own verification plan (world=1 sanity, CP ring smoke + `CP_NO_OVERLAP` A/B, loss/grad-norm
  parity, gamma×DDP-hook checks a–d).
- The dormant `process_group/ProcessGroupNCCL_cplegacy.h` + `CpuSync_fixed.hpp` +
  `processGroupNCCL.cpp` remain on disk (build-excluded); can be removed later if desired.

## Notes

- MPI is still used ONCE for the world `ncclUniqueId` bcast (unavoidable without a socket/file
  bootstrap); sub-groups are now NCCL-native (`ncclCommSplit`), no per-sub-group MPI.
- `ncclCommSplit` requires NCCL ≥ 2.18 (have 2.30.4) and is collective over the parent (world) comm
  — all ranks call it per axis, which holds since every rank runs `initialize_process_groups`.
