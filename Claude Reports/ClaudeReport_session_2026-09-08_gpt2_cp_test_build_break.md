# ClaudeReport_session_2026-09-08_gpt2_cp_test_build_break

2026-09-08 - 16:40 - Diagnosed why `Scripts/Blutrain/gpt2_cp_test.cpp` can no longer be built or run - CP / Scripts/Blutrain/gpt2_cp_test.cpp, context_parallel/ContextParallel.h, process_group/CPProcessGroupNCCL.h, process_group/device_mesh.h

## Question

"Why can't I run the `Scripts/Blutrain/gpt2_cp_test.cpp` script now?"

## Answer

It does not compile. The script was left behind by the CP process-group fork
landed on 2026-08-17. It is a type mismatch on the process group handed to
`ContextParallel`, and it is the **only** error in the translation unit.

## Root cause

The CP ring/overlap surface was split out of the canonical BluTrain PG into a
subclass:

- `process_group/CPProcessGroupNCCL.h:39` - `class CPProcessGroupNCCL : public ProcessGroupNCCL`
- `process_group/device_mesh.h:47` - `get_process_group(int64_t)` now returns
  `std::shared_ptr<CPProcessGroupNCCL>`
- `process_group/device_mesh.h:52` - `world_pg()` stays the plain base
  `std::shared_ptr<ProcessGroupNCCL>` (DP needs no ring methods)
- `context_parallel/ContextParallel.h:271` - the ctor demands the derived type:
  `ContextParallel(const DeviceMesh&, std::shared_ptr<CPProcessGroupNCCL>, float, bool, RotatorType, bool, bool)`

`gpt2_cp_test.cpp` still declares the base type in two ctor signatures, so
`main()`'s correctly-typed `pg` is **upcast to the base on the way in** and can
never be handed back down:

- `main()` at :788-789 - `DeviceMesh mesh(...); auto pg = mesh.get_process_group(0);`
  -> `pg` is correctly `shared_ptr<CPProcessGroupNCCL>` here
- `GPT` ctor at :395 - parameter declared `std::shared_ptr<ProcessGroupNCCL> pg`
  -> upcast #1 at the `GPT model(config, device, pg, mesh, 1234)` call on :796
- `CPAttention` ctor at :213 - parameter declared `std::shared_ptr<ProcessGroupNCCL> pg`
  -> upcast #2 at the `CPAttention(..., pg, mesh, ...)` call on :419
- `CPAttention` ctor body at :235-237 - `make_shared<ContextParallel>(mesh, pg, ...)`
  fails: no conversion from `shared_ptr<ProcessGroupNCCL>` to
  `shared_ptr<CPProcessGroupNCCL>` (derived-to-base is implicit, base-to-derived
  is not)

The compiler reports it as a `std::construct_at` / `make_shared` template
deduction failure inside libstdc++, which buries the real one-line cause under
about 30 lines of instantiation backtrace. Confirmed a single error site with
`-fmax-errors=60`: two unique `error:` lines, both the same `make_shared` at
`gpt2_cp_test.cpp:235`.

## Corroborating evidence

- `Scripts/Blutrain/gpt2_cp_test.cpp` mtime is **Jul 14 22:21**;
  `process_group/CPProcessGroupNCCL.{h,cpp}` are **Aug 17**. The script predates
  the fork and was never migrated.
- `Scripts/Blutrain/bluscriptCP.cpp` **was** migrated at the time and still
  builds - it holds all three PG flavours explicitly at :1478-1480
  (`world_pg()` as base for DDP, `cp_pg` as `CPProcessGroupNCCL` for the ring,
  `ulysses_pg` as base) and threads them through its ctors at :350-351 and
  :522-523.
- `build/` contains `bluscriptCP_exec`, `cp_qknorm_rope_parity_exec`,
  `cp_rope_fused_parity_exec`, `cp_rope_standin_parity_exec`,
  `cp_ulysses_parity_exec`, `cream_positions_test` - but **no**
  `gpt2_cp_test_exec`. It has not been produced since the fork.

## What is NOT the problem

- Makefile wiring is intact: `MAIN_SRC := Scripts/Blutrain/gpt2_cp_test.cpp`
  (:24), `TARGET := build/gpt2_cp_test_exec` (:25), `all: $(TARGET)` (:249).
- `make -n all` emits a complete, correct compile+link plan: sm_86 auto-detected,
  CUDA 13 at `/usr/local/cuda-13`, `BLUTRAIN_ROOT=BluTrain` auto-detected, all
  23 CP/PG/BluTrain objects plus the main TU, `-ltensor` and `-lprofiler`
  resolved. Nothing is missing from discovery or the include set.
- The file exists and is readable (83896 bytes).
- Every other construct in the TU compiles clean.

## Fix (APPLIED and verified end-to-end - see the final section)

Two parameter-type changes in `Scripts/Blutrain/gpt2_cp_test.cpp`:

| Line | From | To |
|---|---|---|
| 213 | `DeviceIndex device, std::shared_ptr<ProcessGroupNCCL> pg,` | `DeviceIndex device, std::shared_ptr<CPProcessGroupNCCL> pg,` |
| 395 | `GPT(GPTConfig cfg, DeviceIndex device, std::shared_ptr<ProcessGroupNCCL> pg,` | `GPT(GPTConfig cfg, DeviceIndex device, std::shared_ptr<CPProcessGroupNCCL> pg,` |

Nothing else needs to change:

- The 6 base-PG call sites in `main()` (`pg->all_reduce` at :1206, :1388, :1404,
  :1541, :1597 and `pg->broadcast` at :1309) keep working - `CPProcessGroupNCCL`
  inherits them.
- The 4 `pg->get_rank()` / `pg->get_worldsize()` uses at :223 and :400 likewise.
- No include change needed: `#include "process_group/device_mesh.h"` on :37
  already pulls in `CPProcessGroupNCCL.h` transitively (device_mesh.h:10). The
  existing `#include "ProcessGroupNCCL.h"` on :36 stays - it is still the source
  of `init_process_group` and the base class.

**Verification performed:** copied the file to
`Scripts/Blutrain/_probe_cp.cpp`, applied exactly the two edits above, and ran
the same host-compile command the Makefile issues, with `-fsyntax-only
-fmax-errors=30`:

```
mpic++ -std=c++2a -fPIC -fsyntax-only -fmax-errors=30 \
  -I. -IScripts/Blutrain -IBluTrain/Tensor-Implementations/include \
  -IBluTrain/Profiler/include -I/usr/local/cuda-13/include \
  -IBluTrain/dist/communication/include -IBluTrain/dist/Data-Parallel/include \
  -IBluTrain -IBluTrain/dist/zero/include \
  -IBluTrain/dist/Distributed_Checkpointing/include \
  -IBluTrain/dist/Tensor-Parallelism/dnn -IBluTrain/dist/Tensor-Parallelism/tensor \
  -IBluTrain/third_party/cufile/Include -DWITH_CUDA Scripts/Blutrain/_probe_cp.cpp
```

Result: **rc=0, zero errors**. The probe copy was then deleted. The fix was then
applied to the real file (below).

## Applied + full build + smoke run

Both edits applied to `Scripts/Blutrain/gpt2_cp_test.cpp` as tabled above.

**Full build:** `make all`, run **serially** on purpose - the 2026-06-23 log
entry records that `-j`/`-k` OOM-kills the `-O3` giant main TU and shows up as a
confusing "nvlink missing gpt2_cp_test.o". Serial is the known-good path.

Result: compile clean, link clean, `[SUCCESS] build/gpt2_cp_test_exec`
(20,136,256 bytes, sm_86, 24 objects + `-ltensor` + `-lprofiler`). The earlier
caveat about the link being unverified is now closed - the link is green.

**Smoke run:** 3 steps, 2 ranks, both RTX 3060s idle beforehand.

Data note worth recording: the default `data_root = "Data_Loader/Data/"`
(gpt2_cp_test.cpp:955) is a **dangling-for-us symlink** -
`Data_Loader/Data -> /root/.cache/huggingface/hub/datasets--navingv--flux/snapshots/5232b3d1...`
which is unreadable without root (`ls` gives Permission denied, and there is no
passwordless sudo on this box). So the default path cannot work as-is here and
`CP_DATA_ROOT` is mandatory. Used `/home/blu-bridge25/cp_data_clean/`
(5 train shards + 1 val, edufineweb). The `/mnt/volgrp04/3rd_floor/adhitya/`
roots from the 2026-09-07 repack sessions do **not** exist on this machine -
those were built on the GPU box.

```
LD_LIBRARY_PATH=BluTrain/Tensor-Implementations/lib:BluTrain/Profiler/lib:$LD_LIBRARY_PATH \
CP_MODEL_44M=1 CP_T=256 CP_MAX_STEPS=3 CP_WARMUP=1 \
CP_DATA_ROOT=/home/blu-bridge25/cp_data_clean/ \
mpirun -np 2 ./build/gpt2_cp_test_exec
```

PASS. Trains end-to-end, clean exit at `=== Context Parallel Training Complete ===`:

| | value |
|---|---|
| loss | 10.916338 -> 10.389044 -> 10.055468 (decreasing) |
| val loss | 10.9118 -> 10.1016 |
| grad norm | 3.7955 -> 2.6540 -> 1.8883 (finite, decaying, no spike) |
| params | 40, total_grad_numel=24,739,200, n_fp32=40, n_other=0, n_noncontig=0 |
| throughput | 37.8k -> 61.5k tok/sec (step 2 is 7.1s because generation runs in it) |
| peak mem | 1782MB -> 2326MB |

Paths exercised: CP ring A2A backward `OVERLAP: dedicated non-blocking stream`,
CP forward `ATTN_FP32=off -> TF32 WMMA/cp.async TC kernel`, `[CP bwd] overlap=ON`,
per-layer timing breakdown, sampling/generation, and the run-numbered CSV logger
(wrote `CP_GPT2_Training_logs/CP_Training_log11.csv`, run 11). No NaN, no throw.

Not covered: this is the 44M config at T=256 for 3 steps on sm_86. It is a
does-it-run check, not a parity or convergence check, and it does not touch the
Ulysses path (`CP_ATTN_MODE=ulysses`), `CP_FUSED_ROPE`, or `CP_ATTN_SPLIT`.

## Files read

- `/home/blu-bridge25/CP/Makefile` (lines 1-319)
- `/home/blu-bridge25/CP/Scripts/Blutrain/gpt2_cp_test.cpp` (targeted ranges)
- `/home/blu-bridge25/CP/Scripts/Blutrain/bluscriptCP.cpp` (grep only)
- `/home/blu-bridge25/CP/process_group/device_mesh.h` (full)
- `/home/blu-bridge25/CP/context_parallel/ContextParallel.h` (via compiler diagnostic)
- `/home/blu-bridge25/CP/process_group/CPProcessGroupNCCL.h` (grep only)

## Files changed

- `Scripts/Blutrain/gpt2_cp_test.cpp` - 2 lines (:213, :395), `pg` parameter type
  `shared_ptr<ProcessGroupNCCL>` -> `shared_ptr<CPProcessGroupNCCL>`

Build artifact produced: `build/gpt2_cp_test_exec`.
