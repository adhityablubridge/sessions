# Claude Report — 2026-06-30 — CP Ulysses (DeepSpeed) Sequence-Parallel Attention

**One line:** Added DeepSpeed-style Ulysses attention as an additive, opt-in mode alongside the existing ring CP path, with a passing ws=2 parity test and a verified-non-destructive ring path.
**Workspace:** CP
**Files:** context_parallel/UlyssesAttention.h, context_parallel/UlyssesAttentionBackward.h, context_parallel/ContextParallel.h, Tests/cp_ulysses_parity.cpp, Makefile
**Plan:** /home/blu-bridge25/.claude/plans/i-am-planning-to-wiggly-dusk.md

---

## Goal
Implement Ulysses sequence parallelism (like DeepSpeed `ulysses_sp.py`) in the CP framework
(`context_parallel/`) using the existing OwnTensor tensor library and the existing CP attention
kernels. Strictly additive / non-destructive — the ring path must stay byte-identical when Ulysses
is off.

## What Ulysses is (vs ring)
Ring rotates K/V around a ring and merges partial outputs via LSE. Ulysses instead does a single
layout swap: an all-to-all turns the sequence-sharded layout `[B,H,Tl,D]` (all heads, my seq slice)
into a head-sharded layout `[B,Hl,T,D]` (my head group, full sequence); each rank runs ONE ordinary
full-sequence causal SDPA over `Hl=H/P` heads; a second all-to-all swaps back. No ring, no LSE merge,
no zig-zag load balancing. After the gather every rank holds the full contiguous sequence `0..T-1`,
so positions are trivial (the GPT-2/wpe path is unchanged).

## Design (additive, mirrors the proven enable_rope() pattern)
- **UlyssesAttention.h** — raw (non-autograd) layout helpers:
  - `ulysses_combine`  `[B,H,Tl,D] -> [B,Hl,T,D]` (reshape split H=P*Hl, permute P outermost,
    uniform `ncclAllToAll`, permute+reshape to full contiguous T).
  - `ulysses_partition` — exact structural inverse.
  - `ulysses_gather_seq` — plain rank-order seq gather `[B,H,Tl,D] -> [B,H,T,D]`, **without** the
    ring path's `unloadbalance` de-zigzag (Ulysses shards contiguously).
- **UlyssesAttentionBackward.h** — `Node(3)`. apply(): branch on the **persisted `unshard_`** flag
  (narrow full grad to local slice when unsharded), `combine` (= partition-backward), ONE
  `sdpa_fused_backward` (existing CP kernel, square, no offsets), `partition` (= combine-backward),
  plain gather to full `[B,H,T,D]`. No de-zigzag anywhere.
- **ContextParallel.h** — added `use_ulysses_` member, `enable_ulysses()` setter, a one-line gate at
  the TOP of `forward_cp` (`if (use_ulysses_) return forward_ulysses(...)`), and the private
  `forward_ulysses` method. Default off ⇒ ring path untouched.

## Attention kernel reuse (per user direction — NO new kernel)
The local full-sequence SDPA calls the SAME `sdpa_fused_forward` / `sdpa_fused_backward`
(FusedSDPAOp.h) the ring uses, which wrap `OwnTensor::cp::cuda::mem_efficient_attn_forward_tc_strided`
/ `mem_efficient_attn_backward_strided` in the existing AttentionForward.cu / AttentionBackward.cu
(and sm89 variants). Ulysses' regime is square causal (`T_q==T_k`, `q_off=k_off=0`); on Ada that
routes to TI's own `fused_attn_forward_tc_sm89_cuda`, on sm86 (the 3060s here) it uses the CP-port
WMMA TF32 kernel. Single block ⇒ no SDPAMerger / LSE merge.

## Critique rounds folded into the design (4 review passes before coding)
1. `pre_sharded=true` is **rejected** (throws) for Ulysses v1 — ring callers hand in zig-zag-sharded
   tensors that would silently mis-order the gathered sequence.
2. Forward unshard uses a **plain** rank-order gather and **skips** `unloadbalance` (verified the ring
   forward at ContextParallel.h:907-942 and backward at ContextParallelBackward.h:765-785 bundle a
   conditional de-zigzag into the gather).
3. Backward gather has the same carve-out (do NOT "mirror exactly"; unconditionally skip de-zigzag).
4. The `unshard_` flag is **persisted on the node** and the backward branches on it (not on inferred
   shape).

## Verification (2x RTX 3060, sm_86)
- **Static:** the Ulysses TU (pulls in all three new headers + the gate) syntax-checks clean.
- **Parity (`Tests/cp_ulysses_parity.cpp`, ws=2):** vs single-GPU causal SDPA reference, forward +
  dQ/dK/dV all `cos=1.0000000`, maxdiff 1e-4..1e-5 (TF32 noise), for **both** `unshard=true` and
  `unshard=false`. `make run-cp-ulysses NP=2` → ALL gates PASSED.
- **Regression:** the ring `cp_rope_standin` parity, **recompiled against the edited header**, still
  passes every gate in both load-balance modes → additive changes are non-destructive.

### Test-bug found & fixed during bring-up
First `unshard=false` backward run mismatched. Root cause was the **test reference**, not the code:
the backward `combine` all-to-all is collective, so every rank's local-slice upstream grad is gathered
and the effective loss is the full-sequence output sum (identical for both unshard modes). The
reference must use the full ones-grad in both cases (the initial slice-masked reference was wrong).
Fixed → all green.

## Out of scope (documented future work)
- **ws=4 parity** is the mandated axis-coverage gate (a transposed P-axis bug can hide at P=2). NOT
  runnable here (only 2 physical GPUs); the test's device mapping was made oversubscription-safe
  (`rank % deviceCount`) so it can run on a 4-GPU box or via GPU-sharing later.
- Fuse the three Q/K/V all-to-alls into one; overlapped all-to-all (`alltoallv_async_stream` on
  `cpRingStream()`); GQA/MQA (`kv_replication_factor`); `CP_ATTN_MODE=ulysses` wiring into
  gpt2_cp_test.cpp; re-enabling `pre_sharded` (needs a contiguous-vs-zig-zag shard tag).

## Memory note
Ulysses' local SDPA runs the full `T×T` causal matrix per call (vs ring's incrementally-sized
blocks), so peak activation per rank rises (full `[B,Hl,T,D]` resident) — relevant at large P with
few heads per rank.
