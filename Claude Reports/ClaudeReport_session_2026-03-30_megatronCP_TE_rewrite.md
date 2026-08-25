# ClaudeReport_session_2026-03-30_megatronCP_TE_rewrite.md

2026-03-30 - Rewrote megatronCP.py to use Megatron-LM's TEDotProductAttention for CP ring-attention - TensorParallelismBeta / megatronCP.py

## What was done

**Goal:** Rewrite `megatronCP.py` to use Megatron-LM's built-in Context Parallel implementation (not custom functions) so the user can compare performance against their C++ DTensor CP implementation.

**Key finding:** Megatron-LM's CP attention is exclusively implemented in `TEDotProductAttention` (`megatron/core/extensions/transformer_engine.py`). The regular `DotProductAttention` explicitly asserts `context_parallel_size == 1`. TransformerEngine is required.

## TransformerEngine Installation

- System: Python 3.10, PyTorch 2.9.0, CUDA 12.8 runtime, CUDA 13.0 toolkit at `/usr/local/cuda`
- No prebuilt `transformer_engine_torch` binary available; compiled from source
- Command used: `CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH MAX_JOBS=1 pip install transformer_engine_torch --extra-index-url https://pypi.nvidia.com --no-build-isolation`
- Final installed versions: `transformer_engine==2.13.0`, `transformer_engine_cu12==2.13.0`, `transformer_engine_torch==2.13.0`
- NVIDIA CUDA libs are in `/usr/local/lib/python3.10/dist-packages/nvidia/*/lib/` and must be preloaded via ctypes before TE import (not all libs — exclude nccl/cudnn which conflict with PyTorch's bundled copies)

## Architecture of new megatronCP.py

- **Process groups:** `parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp_world_size)`
- **TEDotProductAttention:** initialized with `TransformerConfig(context_parallel_size=cp_world_size, ...)`, `attn_mask_type=AttnMaskType.causal`, `cp_comm_type="p2p"` (ring attention)
- **Sequence handling:** Full `[B, T]` loaded on each rank → embedded to `[B, T, C]` → scattered along seq dim → each rank holds `[B, T//cp, C]` for all transformer layers
- **Input to TEDotProductAttention:** sbhd format `[T_local, B, n_head, head_dim]`; TE handles ring K/V communication internally
- **Loss:** computed locally per rank on `targets_local`, then all-reduced (SUM / cp_world_size) across CP group

## Config matches C++ gpt2_cp_test.cpp

- B=4, T=1024, vocab_size=50304, n_embd=384, n_layers=3, n_heads=6
- weight_tying=False, max_lr=6e-4, max_steps=6768, warmup_steps=676
- grad_accum_steps=16, global_batch=65536
- cp_comm_type="p2p" (C++ uses AlltoAll with load_balancing=true — slight difference in comm pattern)

## Files modified

- `/home/blu-bridge25/TP/TensorParallelismBeta/Megatron-LM/megatronCP.py` — full rewrite