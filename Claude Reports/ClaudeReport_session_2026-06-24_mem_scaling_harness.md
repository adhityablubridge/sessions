2026-06-24 - Memory-occupancy scaling sweep harness for CP GPT-2 (C++ BluTrain + PyTorch) - Workspace: OldPush, Files: TensorParallelismBeta/DTensor/{gpt2_cp_test/gpt2_cp_test.cpp, Pytorch/gpt2_cp_attnstyle_fp32.py, mem_scaling_sweep.sh, mem_scaling_table.py}

## Goal
Measure GPU memory occupancy vs sequence length T (1024 -> doubling -> OOM) for
several param configs (25M wt-on, 44M wt-off, 124M wt-on, 163M wt-off) on BOTH
the C++ and PyTorch CP implementations, for all rotators, capturing an nvidia-smi
snapshot after 2 steps per run and tabulating.

## Config matrix (vocab=50304)
Param targets map onto the two existing dim-sets, toggling weight tying:
- 25M  = 384/3/6  tied      | 44M  = 384/3/6  untied
- 124M = 768/12/12 tied     | 163M = 768/12/12 untied
Weight tying drops the lm_head (vocab*n_embd: ~19.3M small / ~38.6M large).
Rotators: PyTorch = {alltoall, allgather}; C++ = {p2p, alltoall, allgather}.

## Changes
Both scripts made env-driven (defaults preserve original behavior):
- PyTorch: T, N_EMBD/N_LAYER/N_HEAD, WEIGHT_TYING, MODEL_LABEL, ROTATE_METHOD,
  MEM_PROBE(+MEM_PROBE_STEPS), MEM_SNAPSHOT_DIR.
- C++: CP_T, CP_N_EMBD/CP_N_LAYER/CP_N_HEAD, CP_WEIGHT_TYING, CP_MODEL_LABEL,
  CP_ROTATOR, CP_MEM_PROBE(+CP_MEM_PROBE_STEPS). Threaded RotatorType through
  GPTConfig -> CPAttention ctor.
- Probe mode: run exactly N steps with grad_accum=1 (total_batch=B*T), skip
  validation/token-gen, then snapshot + exit. B fixed at 4.
- Snapshot taken IN-PROCESS after the last probe step (live, before teardown):
  full nvidia-smi text + parseable header (params, peak). PyTorch records
  torch.cuda.max_memory_reserved; C++ records cudaMemGetInfo used.
- KEY FIX for shared servers: the parseable SMI_USED_MB_PER_GPU line is filtered
  to only the GPUs in CUDA_VISIBLE_DEVICES (nvidia-smi otherwise lists all
  physical GPUs, so a co-tenant like vLLM would pollute the number). Full
  nvidia-smi dump stays complete.

## Deliverables
- mem_scaling_sweep.sh: editable CONFIG block (GPUs, impls, T cap, configs,
  rotators); builds C++ ONCE then loops impl x rotator x config x T, auto-doubles
  T until OOM, writes labeled snapshots + mem_scaling_results.csv. Snapshot
  filename: <impl>_<label>_<rotator>_T<seq>_ws<N>.txt.
- mem_scaling_table.py: parses snapshots + OOM rows -> mem_scaling_table.{csv,md};
  single peak_mb column (peak_src says torch_reserved vs cudaMemGetInfo) +
  smi_max_used_mb for cross-impl comparison.

## Local run (2x RTX 3060 12GB, ws=2) - completed
25M/44M reach T=4096 OK, OOM at 8192; 124M/163M reach T=2048 OK, OOM at 4096.
PyTorch peak < C++ peak at same config/T (e.g. 124M T2048: PT ~7796MB vs C++
~10020MB reserved). Rotator choice has minor memory effect.

## Notes / caveats
- Build is ONCE per source change, NOT per config/T (values are runtime env).
- Server caveat: data_root is still HARDCODED in both scripts to the local path
  (/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Data_Loader/Data). On the
  server this throws at the data loader and the harness mislabels it "OOM"
  (is_fail treats any rc!=0 as OOM). TODO if needed: add DATA_ROOT/CP_DATA_ROOT
  env. Also ensure CUDA_VISIBLE_DEVICES picks IDLE GPUs (server 0-3 run vLLM).
- "OOM" status column = any failed run (no snapshot), not strictly CUDA OOM.
