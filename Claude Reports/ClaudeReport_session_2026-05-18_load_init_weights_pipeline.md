2026-05-18 - Re-introduced LOAD_INIT_WEIGHTS pipeline (PT writer + C++ reader) for true bit-parity init between PT and C++ CP training - Workspace: OldPush/TensorParallelismBeta/DTensor, Files: Pytorch/gpt2_cp_headtail_fp32.py, gpt2_cp_test/gpt2_cp_test.cpp

## Problem
After running the gpt2_cp_test and gpt2_cp_headtail_fp32 comparison, the dumped `debug_rank_lb{0,1}.md` files showed completely different `c_attn.weight` and `c_fc.weight` values at Step 0:
- PT c_attn.weight[0]: 0.01655072532594204
- C++ c_attn.weight[0]: 0.0261631

Even though both used `seed=1234`, the underlying RNG implementations (PyTorch's `torch.manual_seed` vs the C++ `Tensor::randn`) produce different sequences. The earlier session note stating "deterministic seeding is sufficient" was incorrect.

## Solution
Resurrected the `LOAD_INIT_WEIGHTS` mechanism that existed in `/home/blu-bridge25/TP/TensorParallelismBeta/` (older snapshot). The pipeline:

1. PyTorch initializes the model and (on rank 0) writes all parameters to `init_weights.bin` in C++ parameter order, with a companion `init_weights_manifest.txt`.
2. C++ reads `init_weights.bin` (when env var `LOAD_INIT_WEIGHTS=init_weights.bin` is set) and overwrites every parameter's storage in registration order.

## C++ vs PT parameter order
- PT `named_parameters()`: `[wte, wpe, ln_f.w, ln_f.b, *blocks(12N), lm_head]`
- C++ `model.parameters()`: `[*blocks(12N), lm_head, wte, wpe, ln_f.w, ln_f.b]` (per registration order in `GPT` ctor lines 366-403)
- Per-block order matches: `attn.ln.{w,b}, attn.c_attn.{w,b}, attn.c_proj.{w,b}, mlp.ln.{w,b}, mlp.c_fc.{w,b}, mlp.c_proj.{w,b}` (12 params)
- Linear weights transposed PT [out, in] -> C++ [in, out]; embeddings unchanged.

## Changes

### Pytorch/gpt2_cp_headtail_fp32.py (after `model.to(device)` ~line 448)
- Added rank-0 block that writes `init_weights.bin` + `init_weights_manifest.txt` in C++ order with the linear-weight transpose rule and `torch.distributed.barrier()` so other ranks wait for the file.

### gpt2_cp_test/gpt2_cp_test.cpp (after `auto params = model.parameters();` ~line 583)
- Added env-var gated loader: if `LOAD_INIT_WEIGHTS` is set, open the file, read each parameter's bytes into a host fp32 tensor, push it to device, and `copy_` into the parameter in-place.
- Reads in `params` registration order, which must match the manifest order written by PT.

## Build
`make gpt2_cp_test` -> SUCCESS.

## How to run for comparison
```bash
# 1) Run PT first (writes init_weights.bin in cwd)
cd /home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/Pytorch
torchrun --nproc_per_node=2 gpt2_cp_headtail_fp32.py

# 2) Copy or symlink init_weights.bin into the C++ working dir
cp init_weights.bin ..

# 3) Run C++ with LOAD_INIT_WEIGHTS pointing at the file
cd /home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor
LOAD_INIT_WEIGHTS=init_weights.bin mpirun -np 2 ./gpt2_cp_test_exec

# 4) Diff the dumps
diff -u debug_rank_lb0.md Pytorch/debug_rank_lb0.md
diff -u debug_rank_lb1.md Pytorch/debug_rank_lb1.md
```

## Not yet done (follow-up)
- Actually run the parity comparison end-to-end and check whether the HeadTail chunk-level kernel rewrite (2026-05-16) makes the dumps match PyTorch.
- The PT script currently has `n_layer=12, n_head=12` (line 444); C++ side reads its own config. Both must use identical layer/head/embed/vocab/T values for the manifest sizes to line up at load time, otherwise the C++ reader will throw on `short read`.

## Files Modified
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/Pytorch/gpt2_cp_headtail_fp32.py`
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/gpt2_cp_test/gpt2_cp_test.cpp`
