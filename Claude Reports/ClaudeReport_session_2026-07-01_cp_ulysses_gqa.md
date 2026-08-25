# Claude Report — 2026-07-01 — CP Ulysses GQA/MQA (DeepSpeed Strategy B)

**One line:** Added grouped/multi-query (GQA/MQA) support to the Ulysses CP attention path as a strictly-additive parallel path, matching DeepSpeed's shard-first strategy, verified at ws=2.
**Workspace:** CP
**Files:** context_parallel/UlyssesGQAAttentionBackward.h (new), context_parallel/UlyssesAttention.h, context_parallel/ContextParallel.h, Tests/cp_ulysses_parity.cpp
**Plan:** /home/blu-bridge25/.claude/plans/i-am-planning-to-wiggly-dusk.md (GQA Extension v2 section)

---

## Goal
Support GQA/MQA (`nkv < nq`) in Ulysses the way DeepSpeed's `ulysses_sp.py` does — **Strategy B
(shard-first)**: the KV all-to-all carries only `nkv` heads (comm ×1, never the ×g of a full
`nkv→nq` expansion); the grouped broadcast happens locally after the gather; when `nkv < P`, KV is
**partially** replicated by `rep = P/nkv` (to exactly P heads, one per rank), never up to `nq`.

## Strictly-additive structure (per user requirement)
The proven MHA v1 code is byte-unchanged:
- **New** helpers in `UlyssesAttention.h`: `head_repeat_interleave` (narrow+cat) and
  `head_group_reduce` (narrow + raw `Tensor operator+`). `combine`/`partition`/`gather` untouched.
- **New** node `UlyssesGQAAttentionBackward.h`. The MHA `UlyssesAttentionBackward` is untouched.
- **New** method `ContextParallel::forward_ulysses_gqa`, reached by a **single** added early-return
  guard at the top of `forward_ulysses` (`if (k heads != q heads) return forward_ulysses_gqa(...)`) —
  the same additive-gate idiom as the `use_ulysses_` gate. MHA (`nkv==nq`) never enters the new code.
- **New** GQA/MQA cases appended to the parity test; MHA cases unchanged.

## Mechanism (local broadcast, since the CP SDPA is MHA-only)
Our CP SDPA (`sdpa_fused_forward/backward`) is MHA-only and OwnTensor has no `repeat_interleave`, so
DeepSpeed's `enable_gqa` local broadcast is done explicitly:
- Forward: (optional) replicate KV by `rep` when `nkv<P`; `combine` Q by `nq`, KV by `eff_kv`
  (separate all-to-alls); locally expand `kv_local → nq_local` (`head_repeat_interleave` by `g_local`)
  and run one MHA SDPA; `partition` the Q-shaped output back.
- Backward: reverse order of adjoints — SDPA backward → group-reduce over `g_local` (adjoint of the
  local broadcast) → `partition` (adjoint of `combine`) → group-reduce over `rep` (adjoint of the
  replication). Head-alignment is automatic from contiguous scatter; `g_local` always equals the
  queries-per-local-KV-head. Both saved flags (`unshard_`, `pre_sharded_`, `is_causal_`) are persisted
  and branched on explicitly, never inferred from shape.

## Bug found & fixed during bring-up
First GQA run: forward + dQ passed (cos=1.0) but dK/dV failed badly (cos 0.04–0.7). Since dQ has no
group-reduce and everything through `head_group_reduce` failed, the culprit was
`OwnTensor::reduce_sum` over a **middle axis of a 5-D** tensor `[B,groups,r,T,D]` — it produced wrong
results. Reimplemented `head_group_reduce` as an explicit sum of the `r` sub-slices via `narrow` +
the raw `Tensor operator+` (the same primitive the ring backward uses for gamma accumulation). All
dK/dV then matched cos=1.0. (A separate test-side note: reference KV must be built without grad, then
expanded and marked as leaves, so `grad_view()` is valid.)

## Verification (2× RTX 3060, sm_86, ws=2)
- `cp_ulysses_parity`: forward + dQ/dK/dV all **cos=1.0000000** (maxdiff ~1e-4, TF32 noise) for:
  MHA (`unshard` true+false), **GQA `nkv=4` (g=2)**, **GQA `nkv=2` (g=4)**, **MQA `nkv=1` (rep=2)**.
- Ring `cp_rope_standin` regression: still **ALL PASS** (additive changes don't touch it).
- `gpt2_cp_test.cpp` TU syntax-checks clean against the edited header (GQA unreachable there; GPT-2
  is MHA).

## Out of scope / pending
- **ws=4 GQA axis-coverage** (`nq=16,nkv=4,P=4` and MQA `nq=16,nkv=1,P=4`) — mandated but needs a
  4-GPU box (only 2 here). The test's device mapping is oversubscription-safe so it runs on 2 GPUs via
  GPU-sharing if desired; the loop already covers `nkv∈{4,2,1}` and self-skips on divisibility.
- A **GQA GPT-2 model** in `gpt2_cp_test.cpp` (needs a separate nkv-head KV projection — a model
  change beyond CP). GQA is a CP-layer capability here; `CP_ATTN_MODE=ulysses` training stays MHA.
- GQA-aware fused local kernel (would remove the explicit expand/reduce) — only if measured worthwhile.
