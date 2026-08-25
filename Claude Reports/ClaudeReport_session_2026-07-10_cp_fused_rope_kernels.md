# Claude Report — CP fused-RoPE 4-delta kernels (Phase 3)

2026-07-10 - Implemented the offset-parameterized (4-delta) fused GQA RoPE attention kernels for context-parallel load balancing - Workspace: CP / Files: context_parallel/GQA_fused_{fwd,bwd}_sm103_cp.cu, FusedRoPESDPA.h, ContextParallel.h, Makefile, Tests/cp_rope_fused_parity.cpp

## Goal

Phase 3 of the CP fused-RoPE plan: make the two CP-copied fused GQA kernels consume the frozen
4-delta position interface (RopeDeltas.h) instead of the single scalar `pos_offset`, so a rank's
non-contiguous `[head | tail]` shard (HeadTail zig-zag load balancing) rotates at its TRUE global
positions, and the K/V that travels the ring rotates at ITS source-rank positions.

## Decisions (confirmed with user)

- **Option B core, renamed.** Keep the existing bf16 WMMA kernel; rename the library symbols to
  `OwnTensor::cp::cuda::gqa_fused_rope_cp_{forward,backward}` (avoids the libtensor duplicate-symbol
  collision). Scalar `pos_offset` -> `q_d0,q_d1,k_d0,k_d1`.
- **bf16 WMMA** core reused verbatim (no fp32 rewrite).
- **Causal unchanged.** Zig-zag `[head|tail]` packing is globally monotonic and the CP driver
  slices each ring step into `Full/KHeadHalf/QTailHalf` sub-chunks with the right `is_causal`, so
  local-index causal + tile-skip stay correct. The 4 deltas index RoPE ONLY, never masking.

## The only new math

Per side, replacing `pos = local_idx + pos_offset`:
```
pos(local_idx) = local_idx + (local_idx >= len/2 ? d1 : d0)   // len = T_q (Q) / T_k (K)
```
byte-identical to the host source of truth `RopeDeltas.h::rope_global_pos`. `cos_sin_cache` stays
the full global cache on every rank; deltas just re-index it (cheaper than per-step cache
rearrangement, and works for the ring-traveling K).

## Changes

- **GQA_fused_fwd_sm103_cp.cu / GQA_fused_bwd_sm103_cp.cu**: whole TU gated `#if CP_FUSED_ROPE`
  (default build => empty TU => no libtensor collision), namespace `cp::cuda`. `norm_rope_tile`
  takes `(tile_row0,d0,d1,len)`; kernels carry `T_q/T_k` + 4 deltas; bwd reconstruct writes rstd
  to smem (`sQrstd`/`sKrstd`) so the signature drops the saved `q_rstd/k_rstd`; finalize pos uses
  the 4-delta formula. Renamed bf16 wrappers (separate Q/K/V, since T_q may != T_k). Debug-only
  out-of-range-`pos` atomic counter (`CP_ROPE_DEBUG`) — silent OOR would mean a bad delta.
- **FusedRoPESDPA.h**: extern decls + wrapper bodies reconciled to the bf16 signature; wrappers
  `.contiguous().as_type(Bfloat16)` the fp32 Q/K/V, run the kernel, cast outputs back to fp32.
  Tensor-level wrapper signature unchanged => ContextParallel.h call sites untouched.
- **ContextParallel.h**: loud monotonicity-contract comment at the fused call site (line ~646):
  changing sub-chunk slicing requires re-confirming local order == relative global order.
- **Makefile**: `CP_FUSED_ROPE=1` / `CP_ROPE_DEBUG=1` knobs (append `-D...` to flags); new
  `cp-rope-fused` / `run-cp-rope-fused` targets.
- **Tests/cp_rope_fused_parity.cpp**: single-GPU Phase-3 parity — feeds RAW Q/K/V so the kernel
  does norm+RoPE+attention.

## Precision correction (folded in)

The earlier "true-GQA backward dQ/dq_gamma kernel bug" (Logs 2026-07-02/07-07) was **retracted**:
it was a bf16-vs-fp32 artifact in the parity TEST reference (the kernel stores rotated/normed Qr/Kr
and probabilities as bf16 before the backward GEMMs; an fp32 autograd reference keeps them fp32,
and dQ/dgamma are precision-sensitive). In this session the fp32-autograd backward reference
reproduced the SAME low cosines even for MHA (G=1) — confirming the artifact — and they VANISH
against the bf16 production reference. Stale note in the YaRN report was corrected.

## Verification (2x RTX 3060, sm_86)

- **Default build**: `make cp-rope-standin cp-ulysses` clean; standin parity 12/12 PASS
  (cos=1.0) — the `_cp.cu` compile empty, no libtensor collision, existing de-fused CP path
  unaffected by the shared-file edits.
- **CP_FUSED_ROPE=1 build**: kernels + wrappers compile and link (symbols resolve).
- **cp_rope_fused_parity** (`CP_FUSED_ROPE=1 CP_ROPE_DEBUG=1`):
  - Case A (deltas=0 identity, T=64): forward + dQ/dK/dV **BIT-IDENTICAL to the production
    non-CP kernel** (cos=1.0000000, maxdiff=0); dq_gamma/dk_gamma cos=1.0, maxdiff~3e-7 (atomic
    order). This proves the whole port (separate ptrs, T_q/T_k, rstd-recompute, causal, gammas)
    is exact.
  - Case B (HeadTail seam, T_local=32, deltas q=k={0,32}, ref = gathered cache {0..15,48..63}):
    forward cos=0.99999, seam rows (T/2-1, T/2) exact — the new head/tail split is correct.
  - Out-of-range `pos` counter = 0.

## Remaining / follow-ups

- **ws=2 end-to-end ring** test with the fused kernel driven through `ContextParallel`
  (`enable_rope`, raw inputs, contiguous + HeadTail) is the natural next verification. The kernel
  itself (fwd+bwd) is proven bit-identical to production at deltas=0, the delta math is unit-
  tested (cp_rope_deltas_test) + seam-tested here, and the CP plumbing (compute_deltas + call
  site) is wired — so this is integration coverage, not new-logic risk.
- **Build robustness (no `make clean` needed)**: the Makefile was hardened so stale objects can't
  happen: (1) a flag-stamp (`$(OBJDIR)/.cp_rope_flags`, rewritten only when the flag string
  changes, a prerequisite of every object rule) makes `CP_FUSED_ROPE`/`CP_ROPE_DEBUG` a real build
  dependency — flipping auto-rebuilds; (2) `-MMD -MP` + `-include *.d` add automatic header-
  dependency tracking (previously absent — header edits silently didn't rebuild). Verified:
  same-flags => 0 rebuilds, flag-flip => 13 TUs rebuilt, touching `FusedRoPESDPA.h` => only its
  1 includer rebuilt. **Gotcha**: pass the flags to the RUN target too
  (`make CP_FUSED_ROPE=1 CP_ROPE_DEBUG=1 run-cp-rope-fused`); a bare `make run-cp-rope-fused`
  re-evaluates with the flag off and rebuilds to the empty default kernel (which throws).
- GQA head_dim limited to 64/128 (compiled specializations), Br=16/Bc=32 (inherited).
