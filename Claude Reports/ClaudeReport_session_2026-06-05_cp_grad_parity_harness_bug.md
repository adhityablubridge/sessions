# Claude Report — 2026-06-05 — CP step-0 gradient parity: weight-scramble harness bug

**Workspace:** TensorParallelismBeta / DTensor
**Primary files:** gpt2_cp_test/gpt2_cp_test.cpp, Pytorch/gpt2_cp_attnstyle_fp32.py, Pytorch/compare_step0_grads.py, Pytorch/compare_fwd_parity.py

---

## One-line
Step-0 PT↔C++ gradient parity showed orthogonal gradients; root cause was the
`LOAD_INIT_WEIGHTS` harness loading PyTorch weights into C++ in the wrong
parameter order (scrambling the model). Added a name+transpose-aware loader
(`LOAD_INIT_NAMED`); forward parity then confirmed (emb cosine 1.0). No real
base-model or CP bug — residual divergence is numerical (TF32/GELU), compounding
over layers.

## Context / questions carried in
- Earlier sessions resolved Q1 (attnstyle loss saturation) and Q2 (kernels not
  slower in CP) and fixed the attnstyle double-shard (1024→512→256). Trace now
  512/256 matching legacy.
- Open thread: at 161M, C++ training grad-norm rose to ~2 vs PT ~0.8 (note: that
  plot was 44M, not directly comparable). Set out to do a step-0 gradient parity
  check (same init weights, same batch) between PT attnstyle and C++.

## Investigation arc
1. Built a step-0 gradient dump on both sides; compared per-param L2 BY NAME
   (149 params matched). C++ grads were ~150x smaller than PT and per-param
   ratios were wildly non-uniform (weights ~0.003, biases ~0.06, ln_f ~1.2).
2. Added per-element cosine on `ln_f.weight` and `h.0.c_attn.weight`:
   cosine ≈ 0 (orthogonal), even after correctly all-reducing param grads to the
   full batch. Sorted-vector cosine 0.83 (similar distributions, scrambled order).
3. Ruled out: shard mismatch (all-reduced full-batch grads still orthogonal),
   token-batch mismatch (first-8 token ids identical), layout/transpose (ln_f is
   1D, still orthogonal).
4. Isolation at world_size=1 (no ring/LB/sharding): STILL orthogonal → not a CP
   bug, a base-model-level discrepancy (or harness bug).
5. Forward-parity dump (emb / blk0 / lnf, ws=1): **emb itself orthogonal**
   (cosine 0.0017) despite identical idx/pos and identical weight file.
6. Decomposition vs init weights: PT emb = exactly `wte[idx]+wpe[pos]`
   (cos 1.0); C++ emb uncorrelated with wte[idx], wpe[pos], or their sum.
   C++ token-row norms varied (0.57–0.80) vs PT uniform ~0.78.

## ROOT CAUSE
`LOAD_INIT_WEIGHTS` (positional loader) reads `init_weights.bin` — written in
PyTorch `named_parameters()` order (wte, wpe, ln_f, h.0..h.11, lm_head) —
sequentially into C++ `model.parameters()` order, which is DIFFERENT
(blocks first, then lm_head, then wte/wpe/ln_f last; see GPT ctor registration).
Total numel matches (163,109,376) so no short-read error fires. Result: every
weight loaded into the wrong tensor → fully scrambled model. The entire step-0
parity comparison was invalid; real training (C++'s own init, never
LOAD_INIT_WEIGHTS) was always fine, which is why the training curves matched PT.

Also note: even with correct order, C++ Linear weights are `[in,out]` (transpose
of PT `[out,in]`), so a correct loader must transpose 2D Linear weights.

## FIX
- PT: `SAVE_INIT_NAMED=1` writes `init_weights_named.bin` as records
  `<int32 name_len><name><int32 ndim><int64 dims...><float32 data>` in PT layout.
- C++: `LOAD_INIT_NAMED=<file>` reads records, matches each parameter BY NAME
  (walking wte/wpe/ln_f/h.{i}.*/lm_head), and transposes when PT dims are the
  reverse of the C++ tensor dims. Old positional `LOAD_INIT_WEIGHTS` left in
  place but documented as the scrambler.

## VERIFICATION (ws=1, named loader)
- emb  (transformer input): cosine **1.000000**, ratio 1.0  ✔ scramble fixed
- blk0 (after block 0):      cosine 0.9475
- lnf  (final hidden):       cosine 0.5743
- per-token: position 0 anomalous (cos 0.17 at blk0, 0.095 at lnf) vs ~0.96 else.

## Remaining (separate, smaller) question
Residual forward divergence compounds 0.95→0.57 over 12 layers, larger than pure
TF32 rounding. Candidates: TF32 fused attention kernels vs PT fp32 SDPA, GELU
variant (PT `nn.GELU(approximate="tanh")` vs C++ `autograd::gelu` — verify exact
vs tanh), and a position-0 causal-degenerate edge case. Benign for training.
NOT investigated this session.

## Instrumentation added (all env-gated; no effect on normal runs)
- PT: SAVE_INIT_WEIGHTS, SAVE_INIT_NAMED, DUMP_STEP0_GRADS, DUMP_FWD,
  CP_PRINT_SDPA_SHAPES, CP_GRAD_ALLREDUCE, GRAD_PARITY_DUMP.
- C++: LOAD_INIT_WEIGHTS (positional, broken), LOAD_INIT_NAMED (correct),
  DUMP_STEP0_GRADS, DUMP_FWD, CP_DEBUG_SHAPES.
- Scripts: compare_step0_grads.py (by-name L2 + cosine), compare_fwd_parity.py
  (emb/blk0/lnf cosine + per-token).

## Gradient parity re-run with LOAD_INIT_NAMED (DONE)
ws=1, same batch (token ids 2982,11,2026,...), same named weights:
- Gradient MAGNITUDES now match PT across all 149 params:
  TOTAL L2 PT 17.157 vs C++ 17.192 (ratio 1.002); per-layer ratios 0.95–1.14.
  The scramble signature (0.003 ratios, flat biases, layer-6 jump) is GONE.
- Gradient DIRECTIONS partially aligned: ln_f.weight cos 0.42,
  h.0.c_attn.weight cos 0.033 (was -0.04 / -0.0006 when scrambled).
  Output-side grads more aligned than deep ones -> consistent with the residual
  forward divergence compounding backward + step-0 random-init noise.
- Conclusion: magnitude catastrophe fully resolved by the loader fix; remaining
  gap is the directional forward-numerics divergence below.

## Residual divergence root-caused: biased TF32 truncation in CP attention (FIX APPLIED)
- GELU ruled out: both PT (nn.GELU approximate="tanh") and C++ (autograd::gelu,
  GELUKernels.cu) use the SAME tanh approximation. Not the cause.
- Both attention kernel versions use TF32 WMMA + fp32 accumulators + fp32
  online-softmax. The ONLY math difference: the LATEST TI kernel
  (Tensor-Implementations/.../attention/Attention{Forward,Backward}.cu) applies
  `round_tf32_wmma_frag()` (+0x1000u = half a TF32 ULP) to each WMMA operand
  fragment before mma_sync; the OLD TxT/2 copies
  (gpt2_cp_test/context_parallel/Attention{Forward,Backward}.cu) did NOT.
- Significance: default tf32 load TRUNCATES the fp32 mantissa toward zero — a
  BIASED (non-zero-mean) error that accumulates coherently across 12 layers and
  every training step, exactly matching the symptom (C++ grad-norm climbs to ~2
  vs PT ~0.8; forward blk0 cos 0.95 -> lnf 0.57). The +0x1000u makes it
  round-to-nearest (unbiased), as the latest kernel does.
- Config note: both PT and C++ confirmed 161M (163,109,376 params). The earlier
  "44M" caveat was stale — it referred to the old C++ `fourtyfour=true` config,
  since set to false. The grad-norm comparison IS apples-to-apples.

## FIX APPLIED (surgical backport, NOT full kernel replacement)
Added `round_tf32_wmma_frag()` helper + calls at every WMMA site in the two
TxT/2 CP kernels (TxT/2 round-robin subchunking left untouched):
- AttentionForward.cu: 2 sites (QK^T, PV).
- AttentionBackward.cu: 7 sites (dS, dQ x2, dK x2, dV x2).
Verified: round-call count == mma_sync count (FWD 2/2, BWD 7/7).

## Next steps (verify the fix)
1. Rebuild gpt2_cp_test and re-run forward parity (compare_fwd_parity.py):
   blk0 cos should climb 0.95 -> ~0.999, lnf 0.57 -> high.
2. Re-run step-0 grad parity (compare_step0_grads.py): directional cosine should
   rise (ln_f 0.42, c_attn 0.033 -> much higher).
3. Real 161M training with LOAD_INIT_NAMED, log grad-norm per step ~50 steps:
   confirm C++ now tracks PT ~0.8 instead of drifting to ~2.
4. If still a gap, inspect the position-0 causal-degenerate case (per-token cos
   0.17 at pos 0).
