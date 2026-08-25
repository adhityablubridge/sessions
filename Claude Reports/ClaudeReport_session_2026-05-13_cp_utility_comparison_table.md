2026-05-13 - CP utility-based feature comparison table (Ours vs Megatron-LM vs PyTorch native CP) - TensorParallelismBeta / misc/comparison_table_prompt.md

# Summary

Built a utility-level (not kernel-level) Context Parallelism capability comparison across three implementations.

## Sources cross-checked
- Ours: DTensor/gpt2_cp_test/gpt2_cp_test.cpp, DTensor/gpt2_cp_test/context_parallel/ContextParallel.h
- PyTorch: DTensor/Pytorch/gpt2_cp_headtail_fp32.py + torch.distributed.tensor.experimental.context_parallel, _cp_options, set_rotate_method
- Megatron: Megatron-LM/megatronCP.py + TEDotProductAttention (cp_comm_type p2p / a2a / a2a+p2p / all_gather)

## Corrections vs previous draft
- PyTorch CP supports: HeadTail LB (enable_load_balance), round-robin sub-chunking (3-mask), AllToAll & AllGather rotators, FP32 merger (convert_to_f32), cuDNN/Flash/Efficient/Cutlass backends.
- Megatron CP: zigzag LB by default; multiple cp_comm_type options; TE FusedAttention requires bf16/fp16 (no fp32 CP).
- Ours: HeadTail (load_balance flag, zigzag pairing), AllToAll rotator, custom fused fwd + mem-eff bwd attention, FP32 throughout, output AllGather unshard, sparse CE, fused GeLU, custom embedding kernels — but no round-robin sub-chunking (uses global-position mask) and no padding-mask under CP.

## Categories in final table
1. Sequence sharding & load balancing
2. CP communication / rotator
3. Attention backend
4. Precision & numerics
5. Masking & causality
6. Surrounding utilities (LN, GeLU, CE, embeddings, optimizer)
7. Composability (TP / PP / FSDP, auto-shard buffers)

Final HTML table delivered inline in chat.  
