2026-07-16 - Memory-scaling test folder for bluscriptCP (production CP model) vs DeepSpeed-Ulysses Qwen3 - Workspace CP / Scripts/Blutrain/bluscriptCP.cpp, Tests/bluscriptcp/mem_scaling_sweep.sh, Tests/bluscriptcp/mem_scaling_table.py

# Session Report

## Objective
Create a new memory-scaling test folder (mirroring `CP/Tests/`) that sweeps
sequence length T and compares GPU memory occupancy of:
- **bluscriptCP.cpp** (our production context-parallel Llama-style GQA model), and
- **DeepSpeed-Ulysses Qwen3** reference (`LlamaFactory/ds_ulysses/qwen3_ds_ulysses.py`),

runnable on both the 2x RTX 3060 box and the Nx RTX 6000 Ada box.

## What was delivered

### 1. `CP_MEM_PROBE` mode ported into `Scripts/Blutrain/bluscriptCP.cpp`
bluscriptCP had no probe mode (unlike `gpt2_cp_test.cpp`). Added:
- Env: `CP_MEM_PROBE=1`, `CP_MEM_PROBE_STEPS` (default 2, >=2 so optimizer state
  is counted), `CP_MODEL_LABEL` (snapshot label).
- `rotator_label` string set during CP_ROTATOR parse; forced to `"ulysses"` in
  ulysses mode so the snapshot filename matches the sweep.
- In probe mode: `global_batch = B*T*dp_size` (grad_accum=1), `max_steps` =
  probe steps, warmup=1, checkpointing disabled; validation guarded by
  `!mem_probe`.
- After the final probe step: `cudaDeviceSynchronize` + `cudaMemGetInfo` +
  caching-allocator stats (reserved/active/requested/frag + running-max peaks) +
  filtered `nvidia-smi` per-GPU used, written as
  `CPP_<label>_ulysses_T<T>_ws<ws>.txt` in the SAME format `gpt2_cp_test.cpp`
  uses, then `MPI_Barrier` + break.

### 2. `Tests/bluscriptcp/mem_scaling_sweep.sh`
- Two impls: `CPP` (build once via `make bluscript-cp` -> `build/bluscriptCP_exec`,
  driven by CP_* env + mpirun) and `DS` (real 2-step `deepspeed --num_gpus N`
  run per T on `qwen3_ds_ulysses.py`, grad_accum=1 via `global_batch_tokens=B*T`).
- DS memory captured two ways, no python edits: `torch.cuda.max_memory_reserved`
  (the script's `mem_gpu_mb` CSV column -> `torch.peak_reserved_mb`) AND a live
  `nvidia-smi` peak sampled in the background while it trains (-> `SMI_USED_MB_PER_GPU`).
- Coarse T-doubling until OOM (optional fine binary-search, default off since DS
  runs are slow). Stale-output guard (CLEAN=1/FORCE=1). Portable via
  `CUDA_VISIBLE_DEVICES`. Ulysses head-split feasibility guard: skips configs
  where world_size does not divide both q_heads and kv_heads.

### 3. `Tests/bluscriptcp/mem_scaling_table.py`
Adapted aggregator: parses `CPP_*`/`DS_*` snapshots + OOM rows from
`mem_scaling_results.csv`; canonical impl names `bluscriptCP` / `DS-Ulysses`;
collapses each impl's native peak (bluscriptCP=cudaMemGetInfo, DS=torch reserved)
into one `peak_mb` column; `smi_max_used_mb` is the apples-to-apples cross-impl
metric. Emits table CSV/MD + max-T-before-OOM limits CSV/MD.

## Verification (NOT run end-to-end)
- C++: `mpic++ -std=c++20 -fsyntax-only -DCP_FUSED_ROPE=1 ...` -> rc=0 (only
  harmless OpenCL pragma notes). Allocator `get_stats`/`MemoryStats` fields
  confirmed against `device/CachingCudaAllocator.h`.
- `bash -n mem_scaling_sweep.sh` -> clean.
- `python3 -m py_compile mem_scaling_table.py` -> clean.
- Confirmed present: 16 edufineweb shards in `Data_Loader/Data`, deepspeed in the
  LF venv, both `ds_ulysses/*.py` scripts, build target + exec path.
- NOT executed: full `make bluscript-cp` build and the actual sweep runs (heavy;
  box may be in use). Running `./mem_scaling_sweep.sh` builds C++ once then sweeps.

## Key facts / caveats
- Current box = 2x RTX 3060 (12 GB). Ada box elsewhere -> use CUDA_VISIBLE_DEVICES.
- Ulysses splits attention heads: world_size must divide q_heads(6) AND kv_heads(2)
  -> world_size in {1,2} for the 48M config. Wider CP needs more kv_heads (edit
  CONFIGS for C++, pass `--kv_heads N` via `DS_EXTRA_ARGS` for DS).
- Arch nuance: bluscriptCP head_dim=64 (d_model/q_heads, fused-kernel needs 64/128)
  vs the DS Qwen3 reference's decoupled head_dim=32; both ~47-48M params.
- bluscriptCP ulysses is the DEFAULT build (no CP_FUSED_ROPE flag needed).

## How to run
```
cd CP/Tests/bluscriptcp
./mem_scaling_sweep.sh                 # edit CUDA_VISIBLE_DEVICES first
python3 mem_scaling_table.py mem_scaling_runs_bluscriptcp
```
