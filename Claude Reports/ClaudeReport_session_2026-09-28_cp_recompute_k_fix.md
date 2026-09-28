# ClaudeReport_session_2026-09-28_cp_recompute_k_fix

2026-09-28 - 16:30 - Root-caused and fixed the CP recompute_k backward grad-norm explosion (stream-ordering bug, AlltoAll-only); server execution blocked - CP / context_parallel/ContextParallelBackward.h

## Task

"Fix the Recompute_K path locally for Scripts/gpt2_cp_test.cpp and try executing the
script on server 10.101.203 on GPUs 0,1 (after checking they are free)."

Path note: the tagged path `Scripts/gpt2_cp_test.cpp` does not exist; the file is
`Scripts/Blutrain/gpt2_cp_test.cpp`.

## Reproduction (local, 2x RTX 3060, ws=2)

No symptom was specified, so the defect was found empirically by A/B-ing the flag.

| config | step-0 loss | step-0 grad norm | step-1 grad norm |
|---|---|---|---|
| CP_RECOMPUTE_K=0 | 10.916338 | 3.7955 | 2.6540 |
| CP_RECOMPUTE_K=1 | 10.916338 | **4331.32** | **6.2e9** |

Loss and validation loss are byte-identical between modes, so the FORWARD is
correct and the defect is entirely in the backward K/V re-rotation.

## Bisection

1. **world_size=1** (ring collapses, no exchange): RK=0 and RK=1 both give norm
   3.7953 - identical. So the bug requires the ring.
2. **Overlap knobs**: `CP_NO_OVERLAP_BWD=1` -> 1.04e8; `CP_NO_OVERLAP=1` -> nan.
   Disabling overlap makes it WORSE, so it is not the usual compute/comm race.
   (These knobs never reach the faulty code - see the note below.)
3. **Rotator type** - decisive:

   | CP_ROTATOR | RK=1 step-0 norm | verdict |
   |---|---|---|
   | p2p | 3.7955 | correct (matches baseline exactly) |
   | **alltoall (default)** | **4433.15** | **BROKEN** |
   | allgather | 3.7955 | correct |

   AlltoAll-only, and AlltoAll is the default rotator.

## Root cause

`ContextParallelBackward.h` packed the ring staging buffer on **stream 0** and then
called the 1-arg `exchange_buffers`, which routes to
`CPProcessGroupNCCL::alltoallv_async`. That function issues its ncclSend/ncclRecv
on the **dedicated `cpRingStream()`** (CPProcessGroupNCCL.cpp:117), NOT on the
legacy null stream:

```cpp
cudaMemcpyAsync(kv_send.data<float>(), curr_k.data<float>(),
                k_bytes, cudaMemcpyDeviceToDevice, 0);   // stream 0
...
kv_send_work[s] = kv_rotater->exchange_buffers(kv_send); // ncclSend on cpRingStream
```

Nothing orders the pack against the collective. The A2A therefore transmitted
`s_kv_send` BEFORE the memcpys had filled it; the neighbour consumed stale bytes,
dK/dV came out wrong, and the grad norm exploded.

The comment in `RingRotator.h` asserting that "backward escapes it only because
its transfer rides the blocking comm stream (implicitly ordered with stream 0)"
is factually wrong - `cpRingStream()` is a dedicated stream, so no implicit
ordering with stream 0 exists. P2P and AllGather take different code paths and
happened to escape, which is exactly the AlltoAll-only signature measured above.

Why the overlap bisect misled: `CP_NO_OVERLAP*` gate CP's own `OVERLAP` variable,
but the recompute path calls the 1-arg `exchange_buffers`, which hardcodes
`overlap=false` regardless. The knob never touched the broken path, so its
negative result ruled nothing out.

## Fix applied

`context_parallel/ContextParallelBackward.h`, recompute_k send block - mirror the
forward ring, which has always been correct:

- pack memcpys: stream `0` -> `compute_stream`
- prior-send guard: `streamWait(0)` -> `streamWait(compute_stream)`
- `exchange_buffers(kv_send)` -> `exchange_buffers(kv_send, /*overlap=*/true, compute_stream)`

The 3-arg form records a per-slot pack event and makes the ring stream wait on it,
and performs the `recordStream()` allocator hand-off the ring stream requires.
An explanatory comment replaces the incorrect one.

Backup of the pre-edit header: scratchpad/ContextParallelBackward.h.bak

## Verification status

- **Compiles clean**: `mpic++ -std=c++2a -fsyntax-only` on the main TU (which pulls
  in the patched header), zero errors.
- **NOT verified at runtime.** Both routes are blocked:

  1. **Local full rebuild blocked** by an unrelated, pre-existing BluTrain breakage:
     `BluTrain/dist/zero/src/ShardedMuon.cpp` references `d_wire_scale_`,
     `d_clip_coef_`, `owned_grad_fp32_`, `unscaled_`, none of which its header
     declares. `ShardedMuon.h` was edited Sep 26 15:25 (37410 B) while the .cpp
     stayed at Sep 26 12:37 (77522 B) - a mid-refactor header. The server's pair is
     consistent (both Sep 8, header 30270 B). This is someone's work in progress in
     BluTrain, NOT CP, and was deliberately left untouched.
     (The A/B numbers above came from the existing Sep 8 binary, which predates
     that breakage.)
  2. **Server execution blocked**: MPI on 10.101.0.203 is wedged - see below.

  Note the local CUDA 13 toolkit is also partial (no libcublas.so.13 under
  /usr/local/cuda*); runs worked by prepending the repo venv's
  `BluTrain/envv/.../nvidia/cu13/lib` to LD_LIBRARY_PATH.

## Server state (10.101.0.203)

Reached via `blubridge@10.101.0.203` (key auth; the bare `10.101.203` in the request
resolves but defaults to the wrong username).

**GPUs 0 and 1 ARE free** (18 MiB / 0% util). GPUs 4,5 are busy at 98-99% with
~10.5 GB - someone else's job, untouched.

Repo `/mnt/volgrp03/3rd_floor/Adhitya/CP/Context_Parallelism` is current: BluTrain
updated (MemoryMode.h present), binary built today Sep 28 15:35, the Sep 8
CPProcessGroupNCCL fix present at :213/:395, data in Data_Loader/Data.

**MPI is wedged.** Every mpirun launch hangs before reaching the GPU:

- `mpirun -np 2 hostname` (PID 3430296) hung for **2d 23h 54m**
- `mpirun -n 2 ./mpi_runner` (604910) hung 1h 28m
- `mpirun -np 2 ./build/gpt2_cp_test_exec` (994613) hung 35m, started BEFORE this session
- stale session dir `/tmp/ompi.blubridge.0` owned by **root**, dated **Sep 16**

A trivial `timeout 25 mpirun -np 1 hostname` also hangs. Retrying with a fresh
session directory (`--mca orte_tmpdir_base` + TMPDIR) did NOT clear it.

Clearing this needs the hung mpiruns killed and the root-owned Sep 16 session dir
removed. Killing processes on that shared box was denied by the permission layer,
and the root-owned dir needs root, so this was not pursued.

**Orphans left behind:** my own launch attempts left PIDs 1142351, 1142352, 1142355.
They hold NO GPU (0% util) but are still resident and could not be cleaned up.

## Files changed

- `context_parallel/ContextParallelBackward.h` - recompute_k ring send block:
  stream-0 pack -> compute-stream pack + overlap-aware exchange_buffers

Nothing was changed on the server.
