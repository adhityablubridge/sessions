# Gradient Checksum Logging for C++ vs PT Loss Gap Investigation

**Date:** 2026-05-08
**Project:** TensorParallelismBeta
**Branch:** `_adhi_`
**Files modified:**
- `DTensor/gpt2_cp_test/gpt2_cp_test.cpp`
- `DTensor/Pytorch/gpt2_cp_headtail_fp32.py`
**Files created:**
- `DTensor/diff_grad_checksums.py`

## Background

Persistent ~0.5 loss gap between C++ CP training and PyTorch CP reference. All prior fixes (forward kernel HD=64, dK/dV ring accumulation) made no visible difference to the training curve. Activation checksums showed:
- Embeddings: <0.03% abs_sum diff (essentially identical)
- Ring step 0 output: 1.1% diff
- Ring step 1 output: **31.6% diff** — large softmax amplification from ~1% Q/K/V input difference
- LSE at both steps: <1e-4 diff (normalization identical)

The correctness test (`cp_lb_causal_correctness_test`) is **forward-only** — backward was never verified. The persistent gap suggests a systematic gradient error.

## What was done

### 1. Gradient checksum infrastructure (both scripts)

Added `DUMP_GRADS=1` environment gate (independent of `DUMP_ACTS`) that writes per-rank gradient stats after step 0 backward, before gradient clipping.

Output files: `grad_checksum_cpp_rank{0,1}.txt` and `grad_checksum_pt_rank{0,1}.txt`

Format (same as activation checksums):
```
transformer.wte.weight rank=0 numel=38400000 sum=... abs_sum=... sumsq=...
```

### 2. C++ naming convention matches PyTorch `named_parameters()` output

Parameter name mapping used in C++:
- `transformer.wte.weight` → `model.wte.weight`
- `transformer.wpe.weight` → `model.wpe.weight`
- `transformer.h.{i}.attn.ln.weight` → `model.attn_blocks[i]->ln.weight`
- `transformer.h.{i}.attn.c_attn.weight` → `model.attn_blocks[i]->c_attn.weight`
- `transformer.h.{i}.attn.c_proj.weight` → `model.attn_blocks[i]->c_proj.weight`
- `transformer.h.{i}.mlp.ln.weight` → `model.mlp_blocks[i]->ln.weight`
- `transformer.h.{i}.mlp.c_fc.weight` → `model.mlp_blocks[i]->fc_up.weight`
- `transformer.h.{i}.mlp.c_proj.weight` → `model.mlp_blocks[i]->fc_down.weight`
- `transformer.ln_f.weight/bias` → `model.ln_f.weight/bias`
- `lm_head.weight` → `model.lm_head->weight`

### 3. diff_grad_checksums.py created

Copy of `diff_act_checksums.py` with glob patterns changed to `grad_checksum_pt_rank*.txt` and `grad_checksum_cpp_rank*.txt`.

## How to run

```bash
# C++ (step 0 only)
cd DTensor && DUMP_GRADS=1 mpirun -np 2 ./gpt2_cp_test_exec

# PyTorch (step 0 only)
cd DTensor && DUMP_GRADS=1 mpirun -np 2 python3 Pytorch/gpt2_cp_headtail_fp32.py

# Diff
cd DTensor && python3 diff_grad_checksums.py
```

First divergence in gradient checksums will pinpoint which component (embedding, attention QKV, attention proj, MLP fc, MLP proj, lm_head) has wrong gradients.

## Status

- Gradient checksum logging: **implemented, builds clean**
- Grad checksum data: **not yet collected** (need to run both scripts with DUMP_GRADS=1)
- Loss gap root cause: **unknown** — next step is to run and diff
