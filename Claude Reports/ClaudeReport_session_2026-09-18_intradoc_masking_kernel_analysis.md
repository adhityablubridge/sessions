# ClaudeReport_session_2026-09-18_intradoc_masking_kernel_analysis

**Date - Time:** 2026-09-18 - 10:22
**One line:** Analysis of what intra-document masking would require in the H200 attention kernel used by bluscriptCP.cpp, and the change map to bring the TI kernel to CP parity plus intradoc masking.
**Workspace:** CP
**Files:** `.mdfiles/intradoc_masking_TI_vs_CP_kernel_analysis.md` (new, 615 lines)

---

## What was asked

Whether intradoc masking can be implemented for the CP long-context path; what changes it needs in the
attention kernel bluscriptCP.cpp uses on H200; whether that kernel is the same as the one in TI; and a
detailed md report on the requirements and changes needed in the TI kernel to match the CP one and also
have intradoc masking.

## What was done

Read-only analysis. No code changed. Files read: bluscriptCP.cpp, ContextParallel.h,
SplitAttnSDPA_cp.h, GQACausalCP_sm90.h, GQA_causal_cp_{fwd,bwd}_sm90.cu, SDPAMerger.h, LoadBalancer.h,
TI's GQA_{fwd,bwd}_sm90.cu + GQA_fwd_generic.cu, TI's DataLoader.h, mix_buckets.py, and the existing
.mdfiles/hopper_v62_v44_CP_analysis.md. Prior logs grepped; entries 71/72/210/211 gave the TI-upgrade
and Hopper-pivot context.

## Findings

1. The H200 kernel is NOT TI's. All 25/32 run scripts export CP_ATTN_MODE=ring + CP_ATTN_FUSION=split,
   which routes to the CP fork `GQA_causal_cp_{fwd,bwd}_sm90.cu`. It shares TI's whole
   TMA/wgmma/warp-specialised machinery and diverges on exactly 3 axes: separate `T_q`/`T_k`, runtime
   `is_causal`, and a generalized clamped schedule bound (`my_nkv_of`) replacing TI's `shared_nkv =
   pair+1` square-causal algebraic identity.

2. Neither kernel has any mask surface. A TI-wide grep for attn_mask/cu_seqlens/doc_id/varlen/block_mask
   returns only unrelated CPU hits. Masking today is one `is_causal` bool plus the `gcol > grow` compare.

3. Intradoc masking is feasible, and the key enabling property is that a doc-ID *equality* mask is
   permutation-invariant, whereas local-index causal masking is not. That is what makes it work under
   HeadTail sharding and ring rotation where non-diagonal steps have no global position for the peer's K.

4. No data format change needed - shards are EOT=50256 delimited, so doc IDs are an
   exclusive-cumsum(input == 50256) on GPU. Recommended over adding a loader channel.

5. No RoPE change needed - RoPE is exactly relative, so global-vs-doc-local positions give identical
   within-document scores. STRING and Ms-PoE are the exceptions and both are already on other paths.

6. Three silent-NaN hazards that neither kernel can hit today and both would hit with intradoc masking:
   - fully-masked row in a live tile gives `(-inf) - (-inf) = NaN` in the online-softmax rescale;
     needs a dead-row guard. A large finite sentinel instead of -INFINITY does NOT work - it inverts
     the mask and fails quietly.
   - `LSE = -inf` partials NaN out SDPAMerger's sigmoid form (`log(sigmoid(-inf))`); needs a
     log-add-exp rewrite, which is independently more stable.
   - backward `sLSE = -inf` gives `+inf` in the exponent.

7. The structurally hardest kernel change is that the forward consumer's active KV range stops being a
   prefix `[0, my_nkv)` and becomes an interval `[n_start, n_end)`, so the consumer loop needs an
   inactive-head phase mirroring the existing inactive-tail baton loop, and the QK prologue moves.

8. Performance: for short-doc-heavy mixes (b1: ~10 docs per 64k window) intradoc masking is a net
   speedup, roughly 1/d of causal FLOPs for d documents. For pure-b5 (one doc per window) it degenerates
   to plain causal, so that arm stays directly comparable to existing results - which also makes it a
   good correctness canary.

## Recommendation

Three phases: (1) upstream the CP parity generalization into TI, gated on bit-for-bit reproduction of
the base kernel at T_q==T_k; (2) plumb doc IDs end to end with masking OFF, gated on bit-identical loss
curves; (3) enable masking, with adversarial tests that are asserted to actually fire each hazard path.
~2-3 weeks, schedule risk concentrated in the consumer-loop restructure.

## Open items

- Decision needed on whether to also mask boundary labels out of the loss (separate flag, kernel-independent).
- SDPAMerger log-add-exp rewrite is worth doing regardless of intradoc masking.
