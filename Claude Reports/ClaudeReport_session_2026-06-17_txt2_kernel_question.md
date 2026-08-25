# Claude Report — 2026-06-17 — TxT/2 specialized attention kernel question

2026-06-17 - Investigated whether CP round-robin sub-chunking needs a dedicated TxT/2 attention kernel - OldPush / TensorParallelismBeta - torch _attention.py, AttentionForward.cu

## Question
1. Do we need a separate TxT/2 specialized attention kernel to handle round-robin sub-chunking efficiently (without thread underutilization / low MFU)?
2. Does PyTorch (2.9.0+cu128) have a separate case/kernel for the TxT/2 case?

## Findings

### What PyTorch actually does
File: `torch/distributed/tensor/experimental/_attention.py` (v2.9.0), function
`_templated_ring_attention` (line 307), ring loop lines 420-473.

Zigzag/round-robin load balancing gives each rank two sub-chunks `(a,b)`, each
length `c = T_local/2`. Three shapes occur during the ring
(lines 442-462):

| Ring step  | q              | k/v            | Shape    | is_causal |
|------------|----------------|----------------|----------|-----------|
| i == 0     | both (2c)      | both (2c)      | 2c x 2c  | True (triangular) |
| i <= rank  | both (2c)      | first half (c) | 2c x c   | False (dense)     |
| i >  rank  | second half (c)| both (2c)      | c x 2c   | False (dense)     |

- Rectangular halves come from plain slicing: `key.chunk(2,dim=2)[0]` (L453),
  `query.chunk(2,dim=2)[1]` (L462).
- ALL three branches call the SAME op at L466 (`op(q,k,v,is_causal=...)`),
  which is the standard aten flash / mem-efficient SDPA. No TxT/2-specific
  kernel, no sm86-specific kernel. The "case" split is control flow in the
  Python ring driver, not a separate kernel.
- `_partial_update` / `_SDPAMerger` (L123, L151) handle merging the partial
  (second-half) logsumexp/out for the `i > rank` branch.

### Answer to Q1: No separate kernel needed
- Need ONE kernel that handles arbitrary rectangular (Tq != Tk) + is_causal flag.
- MFU: the rectangular branches run is_causal=False -> fully DENSE, no
  triangular 50% waste. Only the i==0 diagonal block is triangular.
- Flash/efficient attention tiles over Q and KV; a rectangular shape changes
  tile COUNT not per-tile occupancy. Smallest dim T_local/2 = T/(2*cp); for
  T=1024, cp=2 -> 256, still 4x a 64-row tile. No underutilization.

### Answer to Q2: separate CASE yes, separate KERNEL no
PyTorch reuses the identical SDPA backend for square and rectangular shapes;
rectangular support is free because the kernels already accept seqlen_q !=
seqlen_kv. Load-balance logic is entirely in the Python ring driver.

### Cross-refs (prior logs)
- 2026-05-08: cp_attn_kernel_bug_hd64_TqNeTk -> already had to support Tq!=Tk.
- 2026-06-05 (log 66): on RTX 3060 sm86, TxT/2 numerically equivalent to
  latest TI attn; no port needed; attention ruled out as divergence source.

## Recommendation for C++ port
Keep a single rectangular-capable `AttentionForward.cu` with an is_causal flag.
Branch in the ring driver (mirroring PyTorch's 3-case split), not in the kernel.
A second TxT/2 kernel buys no MFU here.
