# Claude Report — CREAM context extension in bluscriptCP

- **Date - time:** 2026-07-20 19:43
- **One line:** Implemented CREAM position-relabeling context extension (single-GPU / CP_SIZE=1) in the CP Llama trainer, fully additive, with a no-CUDA-change cache-gather trick.
- **Workspace / files:** CP / context_parallel/CreamPositions.h, context_parallel/ContextParallel.h, Scripts/Blutrain/bluscriptCP.cpp, Tests/cream_positions_test.cpp, Tests/cream_cache_gather_parity.cpp, Makefile, README.md

---

## Goal

Assess whether `Scripts/Blutrain/bluscriptCP.cpp` can implement the CREAM method
(bigai-nlco/CREAM, NeurIPS 2024 — *Continuity-Relativity indexing with Gaussian Middle*)
and, if so, implement it. CREAM extends the usable context window by fine-tuning at the
physical sequence length `T` while feeding manipulated RoPE position **labels** drawn from a
larger `scaled_max` range (head block + truncated-Gaussian middle + tail block), keeping the
attention matrix `T x T`.

User constraints: single-GPU (`CP_SIZE=1`) first, CP later; serve both training and the
`CP_EVAL_PPL` eval path; **every change must be additive** (no existing implementation erased;
`CP_CREAM_MODE=off` byte-identical to before).

## Key finding (why it is feasible with no CUDA change)

RoPE positions in this stack are never stored as data. The fused kernel computes
`pos = local_idx + delta` and indexes `cos_sin_cache[pos]`
(`GQA_fused_fwd_sm103_cp.cu:94-97`, mirrored by `RopeDeltas.h::rope_global_pos`). At
`CP_SIZE=1` all deltas collapse to 0, so `pos == local_idx`, and the wrapper derives
`cache_seq_len` from the cache tensor's row count (`FusedRoPESDPA.h:114`). Therefore feeding a
cache whose row `j` holds the cos/sin for CREAM label `L[j]` makes the UNMODIFIED kernel rotate
each token at its label. Implementation = gather cache rows per step; no kernel edits.

## What was built (all additive, gated behind `CP_CREAM_MODE`)

1. **`context_parallel/CreamPositions.h`** (new, header-only, std-only, unit-testable). Three
   layers isolating cross-language port hazards: pure `cream_labels_from_draws`, the
   `rand_factor_from_u` / `end_id_range` range-CDF layer, and an RNG sampling wrapper. Faithful
   port of `src/train.py` including the four subtle details: `.astype(int)` **truncation** (not
   round), **true** `rand_factor/2` division, inclusive `randint`
   (`std::uniform_int_distribution`), and `np.interp(u, cdf, x)` argument order. Plus PoSE and
   RandPos.
2. **`context_parallel/ContextParallel.h`** — added `set_rope_cache()` (grad-free assert); only
   called under CREAM, so the default path is untouched.
3. **`Scripts/Blutrain/bluscriptCP.cpp`** — `CP_CREAM_MODE/SIGMA/SEED/LOG` env; `factor =
   YARN_SCALE` and `scaled_max = factor*T` **derived** (no independent knob → cannot drift);
   startup asserts (`YARN_SCALE>1`, `factor>=2`, `CP_SIZE=1`, `scaled_max % T == 0`); a separate
   `cream_full_cache_` built at `scaled_max` with YaRN active (base cache left verbatim); the
   `T != context_length` guard preserved verbatim on the off-path and gated for CREAM; per-step
   install that gathers once and fans out to all layers (training resamples per step; eval uses a
   fixed seed); optional middle-block-start distribution log.
4. **Tests** — `Tests/cream_positions_test.cpp` (golden values captured from the real Python
   math; includes the odd-`rand_factor` true-division guard) and
   `Tests/cream_cache_gather_parity.cpp` (gather exactness, `kernel(cache=Gd, delta=0)` vs
   de-fused reference, YaRN-active direction check). Both wired into the Makefile.
5. **README.md** — CREAM env-var section with the batch-shared-labeling caveat and single-GPU
   scope.

## Verification

- `make cream-positions` — 34/34 checks PASS (golden math, range/CDF, structural, guards, randint,
  PoSE/RandPos).
- `make CP_FUSED_ROPE=1 cream-cache-parity` — PASS: gather rows exact (maxdiff 0), kernel vs ref
  cos=0.9996, YaRN direction identical at pos 0 and diverges (cos 0.36) at high pos.
- `make CP_FUSED_ROPE=1 bluscript-cp` — clean build.
- Smoke (tiny cfg, `CP_SIZE=1`, `T=256`): off-mode loss 10.87→10.27 (baseline unchanged);
  `CP_CREAM_MODE=cream YARN_SCALE=8` (scaled_max=2048) loss 10.87→10.29, no NaN, finite grad-norm;
  distribution log varies across the middle range.

## Notes / deferred

- v1 shares one labeling across the batch (resampled per step) and relabels the natural
  contiguous token chunk — sufficient for extension; documented.
- The pre-existing YaRN warn when the base cache (length `context_length`) is built with
  `YARN_SCALE>1` is harmless under CREAM (that base cache is overridden by the gather); the
  `cream_full_cache_` at `scaled_max` matches `factor*T` cleanly.
- **Deferred (CP_SIZE>1):** needs an explicit per-token position array threaded into
  `gqa_fused_rope_cp_forward/backward` (device `pos = q_pos[idx]`), fed from the sharder's
  already-computed-but-unused `ShardedInputs::pos_local`. This also unlocks true per-row labels.

## Plan file

`/home/blu-bridge25/.claude/plans/polished-cuddling-graham.md` (approved after three review rounds:
source-diff fidelity, cross-language port hazards, and the additive/non-destructive requirement).
