# Claude Report — CP-compatibility analysis of the new standalone QK-norm+RoPE kernels

2026-07-31 - Analyzed what changes make the team's NEW standalone fused QK-norm+RoPE kernels CP-compatible (position 4-delta scheme), no code changed - Workspace: CP / Deliverable: .mdfiles/standalone_qknorm_rope_CP_analysis.md

## Task
The team stopped using the fused-in-attention `GQA_fused_*_sm103_cp.cu` (Attention+RoPE+QKnorm) and
made NEW kernels fusing **RoPE + QK-norm ALONE** (`fused_qknorm_rope_{fwd,bwd}_kernel.cu`, fp32, no
attention). Requested: analyze the changes needed to make them CP/HeadTail compatible (esp. the
sin/cos cache offsets) and write it to a `.md`. Analysis only — no implementation.

## Inputs read
Both new kernels; both CP reference kernels (`GQA_fused_{fwd,bwd}_sm103_cp.cu`); both original TI
kernels; `RopeDeltas.h` (delta source of truth); prior logs (Model B history, CREAM CP_SIZE=1, the
non-causal plain-GQA-kernel note from 2026-07-06).

## Core findings
- **Only functional CP change = the cos/sin index:** `pos_offset` scalar → head/tail 2-delta
  `pos = local + (local >= T/2 ? d1 : d0)` (RopeDeltas.h). Norm, `rstd` save/load, `gamma`/dgamma,
  NeoX/interleaved pairing, ITER-3 caching, tree-reduce — all per-token, position-independent.
- **De-fusing implies Model A** (rope the local shard ONCE before the ring at the rank's own deltas
  `compute_deltas(r,i=0,Full)` where q==k; plain attention + ring rotates already-roped K). This
  **removes** all the hard fused-kernel CP machinery: no causal/tile-skip, no local-monotonic
  sub-chunk (Full/KHeadHalf/QTailHalf), no per-step re-derivation, no `T_q/T_k` split. A single
  `(d0,d1)` pair suffices; `len == T`.
- **Non-position issues flagged:** (1) dtype — kernels are fp32, CP path is bf16 → need a bf16 I/O
  variant or explicit cast pass; (2) add `cache_seq_len` + OOR guard (kernels index the cache
  unchecked) + optional CP_ROPE_DEBUG counter; global cache must be resident at YaRN-extended length.
- **Gating risk:** Model A moves causal masking to the paired plain GQA attention kernel, which per
  the 2026-07-06 parity work was **non-causal** → adoption is blocked on a causal plain-attention
  kernel. Called out explicitly as the integration risk (outside the two RoPE files).
- **CREAM/RandPos/PoSE:** works at CP_SIZE=1 unchanged (deltas=0 + gathered `cream_full_cache_`);
  CP>1 deferred (needs an optional per-token `pos_map` in the kernel) — recommended baking that hook
  into the signature now if CP>1 CREAM is on the roadmap.

## Deliverable
`.mdfiles/standalone_qknorm_rope_CP_analysis.md` — 10 sections incl. exact fwd/bwd signature+body
diffs (with line refs), proposed CP signatures, ContextParallel.h integration points, and a
verification plan (deltas=0 bit-identical vs the fused CP kernel, HeadTail seam, OOR=0, e2e
bluscriptCP CP_SIZE=2 loss parity).

## Open questions left for the team (in the .md)
Model A vs Model B; bf16 variant vs cast; CP>1 CREAM (pos_map) roadmap; keep saving rstd vs recompute.
</content>
