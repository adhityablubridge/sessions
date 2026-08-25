# Claude Report — 2026-07-15 19:59 — 3D Hybrid Context Parallelism {DDP, CP-Ring, CP-Ulysses}

Workspace: **CP** (`/home/blu-bridge25/CP`)

## Goal
Add a 3-D hybrid context-parallel path to bluscriptCP: **Ulysses (all-to-all) on the
inner axis, Ring (P2P rotation) on the outer axis** (Unified Sequence Parallelism —
USP / LoongTrain). Recommended placement: Ulysses intra-node (NVLink), Ring
inter-node. Deliverables: implementation + pseudocode plan, test scripts (not
runnable on this 1-2 GPU box), README update.

## Design (approved plan)
- **Nesting is forced**: Ulysses inner (all-to-all reassembly needs the U ranks of a
  group to hold the U sub-pieces of ONE ring block), Ring outer (causal HeadTail
  zigzag lives on the ring axis, spanning the full T).
- **Sequence sharding** composes HeadTail-over-ring (R) then contiguous-over-ulysses
  (U); rank holds `T/(R*U)` tokens. After the inner `ulysses_combine` reassembles
  the U pieces in rank order, the ring sees the full `T/R` block in `[head|tail]`
  layout — exactly what `compute_deltas(N=R)` expects, so **no kernel change**.
- **Placement is by rank index** (DeviceMesh groups arithmetically); correctness is
  placement-agnostic. Physical mapping (which axis on which interconnect) is the
  launcher's job via node-major rank placement. An **advisory** fast-domain guard
  (NO_GPUS_PER_NODE % U) warns but does not abort — enabling ring-across-NUMA.
- **Backward** composes: `partition` adjoint → ring `ContextParallelBackward` →
  `combine` adjoint. DDP's global weight-grad all-reduce is unchanged/uniform.

## Files changed
- `context_parallel/HybridUlyssesBackward.h` (NEW): `UlyssesCombineBackward` /
  `UlyssesPartitionBackward` — 1-in-1-out autograd nodes whose adjoints are the
  inverse all-to-all, so the ring's own backward chains between them.
- `context_parallel/ContextParallel.h`: `enable_hybrid()`, `forward_hybrid()`,
  `combine_ag`/`partition_ag` (autograd-wrapped all-to-alls), `finalize_hybrid_output`
  (unshard = inner Ulysses gather THEN outer Ring gather + de-zigzag),
  `hybrid_selfcheck` (CP_SELFCHECK round-trip), `shard_sequence_pre_embed_hybrid`,
  hybrid members + `in_hybrid_ring_` re-entrancy guard, forward_cp dispatch.
- `Scripts/Blutrain/bluscriptCP.cpp`: `CP_ATTN_MODE=hybrid` + `CP_ULYSSES_SIZE`;
  3-D mesh `{dp, ring, ulysses}`; ring_pg/ulysses_pg + ranks; CausalGQA/GPT plumbing;
  hybrid target sharding in fwd/val/train; divisibility asserts + advisory
  fast-domain warning; config prints/dump.
- `Makefile`: `cp-hybrid` / `run-cp-hybrid` targets.
- `Tests/cp_hybrid_parity.cpp` (NEW): parity vs a **pure-ring baseline** (same bf16
  fused kernel over the world PG) — bf16-consistent oracle. `CP_RING_SIZE` selects
  the factorization per launch.
- `Tests/hybrid_cp_smoke.sh` (NEW): multi-node integration driver (node-major launch,
  finite-loss + placement checks). Cluster-only.
- `README.md`: third attention mode, hybrid topology + fast-domain/NUMA notes, env
  table rows, build+run example.

## Verification (on this 2×RTX3060 box)
- Build clean: `make CP_FUSED_ROPE=1 bluscript-cp` and `make CP_FUSED_ROPE=1 cp-hybrid`
  (both exit 0).
- **Hybrid parity (MHA), NP=2, ring=2/ulysses=1**: forward, dQ, dK, dV all
  **cos=1.0, maxdiff=0** vs pure-ring baseline — validates dispatch, combine/partition
  autograd wrappers, composed sharder, and ring end-to-end (at U=1 the all-to-all is
  identity, so this is exact equality with pure-ring).
- **bluscriptCP hybrid end-to-end**, NP=2 (`ring=2 x ulysses=1`), 3 steps: loss
  10.90 → 10.50 → 10.23, no NaN/Inf, exit 0 — validates 3-D mesh + hybrid sharder +
  enable_hybrid + fwd/bwd/optimizer integration.

## Known limitations / follow-ups
- **Real U>1 needs ≥4 real GPUs.** Oversubscribing NP=4 on 2 GPUs fails with
  "invalid device ordinal" — the framework's PG/NCCL init assumes `rank == global
  device ordinal`. Run `run-cp-hybrid NP=4/8` and `hybrid_cp_smoke.sh` on a cluster.
- **GQA in the ring-RoPE path**: at U=1 (where hybrid must equal pure-ring), GQA
  (nkv<nq) forward diverges (cos≈0.04) in the **baseline pure-ring** run itself —
  i.e. it is a property of the ring-RoPE fused kernel (`gqa_fused_rope_cp`), NOT of
  the hybrid layer. GQA cases are gated behind `CP_HYBRID_GQA=1` in the parity test
  and flagged; hybrid GQA correctness tracks ring-RoPE GQA support (needs separate
  investigation). MHA is bit-exact. bluscriptCP still asserts `kv_heads % U == 0`
  (topological requirement for the head split).

## How to run on a cluster
```
make CP_FUSED_ROPE=1 cp-hybrid
CP_RING_SIZE=2 make CP_FUSED_ROPE=1 run-cp-hybrid NP=4      # ring=2, ulysses=2
CP_RING_SIZE=2 make CP_FUSED_ROPE=1 run-cp-hybrid NP=8      # ring=2, ulysses=4
CP_RING_SIZE=4 make CP_FUSED_ROPE=1 run-cp-hybrid NP=8      # ring=4, ulysses=2
GPUS_PER_NODE=4 NNODES=2 bash Tests/hybrid_cp_smoke.sh
```
Plan file: `/home/blu-bridge25/.claude/plans/make-a-detailed-plan-tidy-cerf.md`
