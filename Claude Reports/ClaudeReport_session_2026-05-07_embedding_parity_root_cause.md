# Claude Report: Embedding Parity Root Cause (Step 2)
**Date:** 2026-05-07  
**Workspace:** TensorParallelismBeta  
**Task:** Debug forward activation divergence between C++ and PT

---

## TL;DR

**Root cause of "embedding divergence" was a TEST SETUP BUG** — running C++ without `LOAD_INIT_WEIGHTS=init_weights.bin` env var, so it used its own random init instead of PT's saved weights.

After fixing: **embeddings match bit-exact**. Real divergence starts at attention (block_0_out).

---

## Systematic Debugging Process

### Phase 1: Evidence Gathering
Added diagnostic dumps at component boundaries:
- `idx_sharded` (post-shard tokens, both ranks)
- `tok_emb`, `pos_emb` (rank 0 only for C++ pre-shard)
- `wte_weight`, `wpe_weight` (direct table dumps)
- `pos_idx` (verify lookup indices)

### Phase 2: The Tell
- Init checksum (after load): `wpe sum = 28.22, abs_sum = 6278`
- Forward-time `wpe_weight` dump: `sum = 0.21, abs_sum = 6280`
- abs_sum essentially identical, but signed sum near-zero in current C++ run

This contradiction revealed: the file `init_checksum_cpp_after_load.txt` was from a PREVIOUS run; current run wasn't loading weights.

### Phase 3: Root Cause Confirmation
Re-ran with `LOAD_INIT_WEIGHTS=init_weights.bin`:

```
tok_emb:  PT=-52.5  C++=-52.5  rel_diff=1.4e-16  (bit-exact)
pos_emb:  PT=+28.2  C++=+28.2  rel_diff=1.1e-11  (bit-exact)
emb:      PT=+60.4  C++=+60.4  rel_diff=0.0     (bit-exact)
Loss:     PT=10.8671  C++=10.8757  (0.08% diff)
```

---

## Real Divergence (Next to Debug)

`block_0_out` shows the actual issue:
- sum: PT=-418.5 vs C++=-250.4 (40% rel_diff)
- abs_sum: 94313 vs 94131 (0.2% diff — magnitudes match)
- std: matches within 0.2%

**Pattern:** Magnitudes are essentially equal, but signed sum diverges. Suggests:
1. FP32 reduction order difference in attention
2. Or actual attention computation bug

---

## Key Lesson

When debugging multi-component systems, **verify the full test setup BEFORE deep-diving into code logic**:
- Env vars (LOAD_INIT_WEIGHTS in this case)
- File freshness (after_load.txt was stale)
- Command flags

The diagnostic dumps WERE the right move — they exposed an inconsistency between dumped state and reference file, leading to discovery of the setup bug.

---

## Files Modified
- `DTensor/gpt2_cp_test/gpt2_cp_test.cpp`
  - Added wpe_weight, wte_weight, pos_idx, idx_sharded, tok_emb, pos_emb diagnostic dumps
  - Added cudaDeviceSynchronize after embedding ops
  
- `DTensor/Pytorch/gpt2_cp_headtail_fp32.py`
  - Added matching idx_sharded, pos_sharded, tok_emb, pos_emb dumps
  - pos_emb dumps `pos_emb[0:1]` to remove B-replication for parity with C++

---

## Run Commands

```bash
# C++ (must include LOAD_INIT_WEIGHTS!)
LOAD_INIT_WEIGHTS=init_weights.bin DUMP_ACTS=1 mpirun -np 2 ./gpt2_cp_test_exec

# PT
DUMP_ACTS=1 torchrun --nproc_per_node=2 Pytorch/gpt2_cp_headtail_fp32.py

# Diff
python3 diff_act_checksums.py
```

---

## Next Phase: Attention Block Divergence

Need to instrument inside the attention block (between layers within block_0):
- attention output (raw QK^T softmax V)
- LayerNorm 1 input/output
- MLP input/output
- Block input + attention output (residual)

This will localize whether divergence is in:
- Attention compute (FlashAttn vs ring attention)
- Ring rotation aggregation
- LayerNorm precision
- MLP (matmul + GeLU)
