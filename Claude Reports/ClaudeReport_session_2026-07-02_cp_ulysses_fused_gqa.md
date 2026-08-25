# Claude Report — 2026-07-02 — CP Ulysses FUSED (RoPE+QK-norm+GQA) path + kernel-bug finding

**One line:** Wired the team's fused GQA kernel (RoPE+QK-norm) into Ulysses as an additive Llama-style path; found and isolated a dQ/dq_gamma bug in the fused GQA *backward* kernel for true GQA.
**Workspace:** CP
**Files:** context_parallel/UlyssesFusedGQAAttentionBackward.h (new), context_parallel/ContextParallel.h, Tests/cp_ulysses_parity.cpp
**Plan:** /home/blu-bridge25/.claude/plans/i-am-planning-to-wiggly-dusk.md (GQA Extension v3)

---

## What was built (strictly additive)
- **`enable_ulysses_fused(cos_sin_cache, q_gamma, k_gamma, eps, interleaved)`** + **`forward_ulysses_fused()`**
  in `ContextParallel.h` (one guard line atop `forward_ulysses`; v1 MHA and v2 plain-GQA paths untouched).
- **`UlyssesFusedGQAAttentionBackward.h`** — new node calling `OwnTensor::cuda::gqa_fused_flash_attn_backward`.
- Calls the team's bf16 fused kernel `gqa_fused_flash_attn_forward/backward` (RoPE + QK-norm + causal GQA).
  The kernel is GQA-native (`hkv = hq/G`), so the v2 local head-broadcast is **gone**; only the `nkv<P`
  replication (for the all-to-all distribution) remains, with a group-sum replication adjoint in backward.
- Mirrors the production `sdpa_gqa_fused` recipe: pack `[Q|K|V]` via `as_type(Bfloat16)` + `cat(flatten)`,
  split grads via `narrow_view + reshape + as_type(Float32)`.
- **Ulysses makes RoPE trivial:** after the combine all-to-all each rank holds the full contiguous
  sequence, so `pos_offset = 0`, positions `0..T-1` (no ring 4-delta bookkeeping).
- **Gamma grads** (shared `[hd]` over a head-partitioned axis) are **SUM all-reduced** over the CP group
  (`op_t::sum`); correct even with replication (RMSNorm `dgamma` is linear in the upstream grad and the
  replicated copies share an identical forward). Gammas are all-or-nothing (asserted).

## KERNEL BUG FOUND (not CP): fused GQA backward dQ / dq_gamma
The fused GQA **backward** kernel (`GQA_fused_bwd_sm103.cu`, `gqa_fused_flash_attn_backward`) returns
**incorrect `dQ` and `dq_gamma` for true GQA (`1 < Nkv < Nq`)**. Forward, `dK`, `dV`, and `dk_gamma`
are all correct.
- Repro (`Tests/cp_ulysses_parity.cpp`, `run_fused_mode`, hd=64, T=64): `nq=8, nkv=2, G=4` →
  `dQ cos≈0.118`, `dq_gamma cos≈0.270`; forward/`dK`/`dV`/`dk_gamma` `cos>0.999`.
- **Isolation:** reproduced at **ws=1** (no all-to-all, no CP orchestration — the kernel is called
  directly on `[B, Nq=8, T, D]` / `[B, Nkv=2, T, D]`). So it is the kernel, not the CP wiring.
- **Cross-checks:** MHA (`Nkv=Nq=8`, G=1) and MQA (`Nkv=1`, G=8) fused paths pass ALL grads
  (`cos>0.9999`) at ws=1 and ws=2. The v2 plain-SDPA GQA parity passes `nkv=2` fully (so CP's GQA
  combine/partition/gather is correct). An independent fp32 reference (rms_norm+rope+MHA) agrees with the
  kernel on every output except `dQ`/`dq_gamma` and only for `1<Nkv<Nq`.
- **Likely locus:** the dQ path's group indexing (`hkv = hq/G`) — the trivial G=1 (MHA) and Nkv=1 (MQA)
  cases work; intermediate GQA does not. Hand this to the kernel team.

## Test handling
The intermediate-GQA fused `dQ`/`dq_gamma` comparisons are marked **XFAIL** (printed, not counted) with a
"KNOWN fused-GQA-backward kernel bug" note. Everything else is a real gate.

## Verification (2× RTX 3060, ws=2 unless noted)
- `cp_ulysses_parity`: **ALL gates PASS**. v1 MHA (unshard t/f), v2 plain GQA `nkv=4,2` + MQA `nkv=1`,
  fused MHA `nkv=8`, fused MQA `nkv=1` — all forward+dQ/dK/dV(+gammas) `cos≈1.0` / `>0.9999` (bf16 for
  fused). Fused GQA `nkv=2`: forward+dK/dV/dk_gamma PASS; dQ/dq_gamma XFAIL (kernel bug).
- Also reproduced the kernel bug at **ws=1**.
- **Regression:** ring `cp_rope_standin` parity still ALL PASS (rebuilt against the edited header).
- `gpt2_cp_test.cpp` TU syntax-clean.

## Out of scope / follow-ups
- **Fix the fused GQA backward dQ/dq_gamma kernel bug** (kernel team) — then flip the XFAILs to real gates.
- ws=4 fused GQA axis coverage (needs a 4-GPU box; device mapping is oversubscription-safe).
- Wiring the fused path into a real Llama GQA model in `gpt2_cp_test.cpp` (model build beyond CP).
- Reconcile the in-node gamma SUM all-reduce with an external param all-reduce for real training.
