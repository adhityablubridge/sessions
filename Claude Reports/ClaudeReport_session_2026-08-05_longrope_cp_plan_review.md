# Claude Report — Review of the CP-for-LongRoPE plan (pitfalls + logical fallacies)

- **Date - time:** 2026-08-05 15:21
- **One line:** Reviewed `.mdfiles/polished-cuddling-graham.md` (CP_SIZE>1 support for LongRoPE eval+search) against the actual code; found 5 blockers, 5 methodology fallacies, 5 operational risks. No code changed.
- **Workspace / files:** CP / .mdfiles/polished-cuddling-graham.md (reviewed against Scripts/Blutrain/bluscriptCP.cpp, Tests/bluscriptcp/{run_longrope_pipeline.sh,longrope_search.py}, context_parallel/{ContextParallel.h,UlyssesAttention.h})

---

## Blockers (plan as written will not run / will run wrong)

1. **Default `CP_ATTN_MODE` is `ulysses`, not ring** (`bluscriptCP.cpp:133`). The definitive
   commands set `CP_SIZE=4 CP_N_KVHEAD=2` and no attn mode. The pure-Ulysses branch has NO
   head-divisibility startup guard (the kv_heads check at `:831` is inside the `hybrid` branch);
   it throws late from `UlyssesAttention.h:47` ("H must be divisible by world_size"). Must set
   `CP_ATTN_MODE=ring` -> which also means the HeadTail map, not the contiguous one, is the
   live path.
2. **`Tpos` is overloaded global-vs-local in the CSV path.** In `bluscriptCP.cpp:1445` `Tpos`
   is used for (a) window construction from `CP_EVAL_TOKENS_BIN` (`:1469`, `:1506-1521`) and the
   `input` tensor shape — GLOBAL, forward re-shards it; and (b) logits indexing `bi*Tpos + t`
   (`:1555`) and the CSV loop — LOCAL / GLOBAL respectively. Setting `Tpos = cfg.T/cp_size`
   silently feeds a T_local-length sequence into `model.forward`. Needs two distinct names
   (`T_glob` / `T_loc`) with every use classified.
3. **`s=64` fine-tune dies at startup.** `run_longrope_pipeline.sh` passes
   `CP_GLOBAL_BATCH=$GBATCH` (default 131072) with `CP_B=1 CP_T=262144`;
   `bluscriptCP.cpp:872-874` computes `tokens_per_micro = B*T*dp_size = 262144` and dies on
   `global_batch % tokens_per_micro != 0`. Needs `GBATCH >= 262144` — which breaks the
   token-budget parity the script comments claim.
4. **Fitness cost is unbudgeted and infeasible at these lengths.** The resident evaluator's NLL
   loop (`:1409-1424`) is serial CPU logsumexp over the full vocab and, unlike the CSV path
   (`:1552`), has no `#pragma omp`. At `CALIB=16`, `T_local=32768`, `V=50304`: ~2.6e10 `exp()`
   per candidate per rank. Plus the fp32 host copy of logits is 6.6 GB/rank at s=32 and
   13.2 GB/rank at s=64 (x4 ranks). `SEARCH_BUDGET_SEC` default 30600 (8.5 h, not overridden in
   the plan's commands) will terminate the GA after a handful of candidates.
5. **Collective divergence in the resident CP loop.** Plan broadcasts the candidate path but not
   the parse result: a rank that throws and `continue`s (`:1382-1385`) skips the collectives the
   others enter -> NCCL hang. Same for `die()` inside the loop. Also `mpirun -np 4` merges four
   ranks' stdout into the pipe `longrope_search.py` line-reads.

## Methodology / logical fallacies

6. **Search calibrates on the same windows the final eval scores.** Search sets
   `CP_EVAL_WINDOWS=calib_windows` (`longrope_search.py:139`) and the evaluator does
   `val_loader.reset()` then takes the first N windows; step 4 evals the SAME val split from
   reset with `EVAL_WINDOWS=32`. Calibration windows are a strict subset of eval windows ->
   LongRoPE is fit to the reported test set, YaRN is not. Any win is unfalsifiable.
7. **`MSCALE=0` vs YaRN's baked-in `m`.** `build_rope_cache` multiplies `m=0.1 ln(s)+1` into
   cos/sin unconditionally; the plan's "paper-faithful no-m" LongRoPE arm therefore differs from
   the YaRN arm in TWO variables. Needs the 2x2 (both arms with and without m).
8. **"Formula-breakage is model-size-robust" is the load-bearing premise and is asserted, not
   established.** The paper's 99.64-vs-1.87 gap assumes an in-distribution PPL ~2; at PPL ~150
   the measurable range is compressed and both arms may saturate. No pre-registered criterion,
   and no saturation control that distinguishes "LongRoPE held" from "both broke".
9. **Unequal fine-tune between arms.** The LongRoPE arm gets `FT_STEPS=300` of re-warmup
   fine-tune; the plan's YaRN baseline is "run the existing YaRN-LCE eval path" (no fine-tune).
10. **Section 4's "empirical parity is stronger than a unit test".** This introduces a 4th copy
    of the position map, and the parity runs are only at `CP_SIZE=2` while the definitive run is
    `CP_SIZE=4`. Also `Section 3`'s "all new work under `if (cp_size>1)`" contradicts sections
    1-2, which change `Tpos` unconditionally.

## Operational risks

11. `longrope_best.txt` is a hardcoded untagged filename; the two `nohup ... &` commands collide
    on it (and on the 4 GPUs).
12. `wait_gpu_free` checks only the first GPU with a 4 GB threshold, while the job now needs 4
    GPUs and far more than 4 GB.
13. Candidate files live in a local `tempfile.mkdtemp` — fine on one node, breaks multi-node.
14. `DataLoaderLite` is constructed with `dp_rank/dp_size` (`:1009-1012`), so identical input per
    CP rank holds ONLY because `world_size == cp_size => dp_size == 1`. Worth asserting, since it
    is the invariant the whole design rests on.
15. Recommended order: fix 1-5, run CP-vs-1GPU parity at CP_SIZE=4 in ring mode, then s=32 with
    disjoint calibration windows, and gate the s=64 spend on the s=32 result.

## Status

Review only — no files modified.

---

# Round 2 — review of the revised plan (2026-08-05 15:40)

All 15 round-1 findings are addressed in the revision. Re-review of the revised text:

## Clean fixes (no further action)

`CP_ATTN_MODE=ring` + new early ulysses guard (1); `T_global`/`T_local` disambiguation (2);
`GBATCH` for s=64 (3); broadcast-the-parsed-genome instead of the path (5); `CP_EVAL_SKIP_WINDOWS`
held-out eval (6); `YARN_NO_MSCALE` 2x2 (7); shared `local_to_global_pos` helper (10);
`dp_size==1` assert (14); CP_SIZE=4 parity + malformed-file no-hang test (10, 5); artifact tagging
and multi-GPU `wait_gpu_free` (11, 12). Confirmed `longrope_search.py:132` already splats `--arch`
tokens into the child env, so `CP_ATTN_MODE=ring` in ARCH does reach the evaluator.

## Fixes that create a new problem

- **B1. GPU-CE fixed the SEARCH, not the deliverable.** Change #2 leaves the per-position CSV path
  doing `logits.as_type(Float32).to_cpu()` + full-vocab logsumexp: 13.2 GB device + 13.2 GB host per
  rank at s=64 (6.6 GB at s=32), x4 ranks. Pipeline step 4 runs exactly this at T=262144. Needs
  chunking over `t` or a GPU per-position NLL.
- **B1b. Two definitions of overall PPL.** Fitness becomes `Allreduce(mean)/world_size`; the CSV
  reports `sum(nll)/sum(cnt)`. Equal only if token counts match per rank and no target is skipped —
  `sparse_cross_entropy_loss` (`LossOps.h:37`) has no `ignore_index`, while the CPU loop skipped
  out-of-range targets. Add a one-time cross-check at CP_SIZE=1.
- **B2. GBATCH=262144 breaks token-budget parity.** Same `FT_STEPS=300` now means 2x the tokens for
  the s=64 arm vs s=32 and vs base. Halve FT_STEPS at s=64 or match tokens explicitly; the matched
  YaRN fine-tune must use identical GBATCH/FT_STEPS.
- **B3. `YARN_NO_MSCALE` edits `build_rope_cache`** — the function the original plan promised
  untouched. Re-run the off-mode byte-identity regression.
- **B4. Contiguous becomes the untested branch.** Verification #2's "CP_ATTN_MODE unset at
  CP_SIZE=1" proves nothing (both maps are identity at CP=1). Add a contiguous parity at CP_SIZE=2.

## Still open

- **C1. CP shards T but not V — the logits ceiling is not solved for the fine-tune.** COMPOSE=alone
  at s=64/CP=4: logits [1,65536,50304] fp32 = 13.2 GB + grad 13.2 GB + CE workspace, on top of
  model/optimizer/activations. s=32 (6.6 GB) is plausible; s=64 is marginal-to-OOM. Run the existing
  mem-probe mode before committing 8 h of search.
- **C2. No absolute runtime number.** Make it a precondition: time candidate #1, extrapolate, abort
  if projected scored-candidate count is below a stated floor.
- **C3. "(margin)" is an unfilled blank** — a criterion with a placeholder is not pre-registered.
  Also the unigram floor is the far ceiling; the informative reference is the model's own
  sliding-4k-window PPL. Fill both numbers before launching.
- **C4. Held-out window supply.** s=64 needs (8+32)x262144 ~ 10.5 M val tokens; if the val split is
  shorter DataLoaderLite wraps and eval silently re-scores the calibration windows — reopening the
  leak the fix closed. Assert `(skip+windows)*T <= val tokens`.
- **C5. The 2x2 protocol has no runnable commands** — the block still produces one cell
  (LongRoPE/no-m); the matched YaRN fine-tune is prose only, and the +m cell needs its own search.
- **C6. First-ever 4-rank ring run** (prior CP verification was 2 GPUs; 4-rank was explicitly
  deferred). Parity #1/#2 at CP_SIZE=4 is the right gate — run it before anything else.
- **C7. Internal inconsistency:** step 5 still says the runs "collide on longrope_best.txt" after
  Change 4 tags it by S, and both commands still end in `nohup ... &` despite "run ONE at a time".
- **C8.** The new ulysses guard should cover `q_heads % cp_size` too, not just kv_heads.

---

# Round 3 — review of the revised plan (2026-08-05 16:05)

All 12 round-2 items folded in as a "Round 2 revisions" changelog. Verdict: the plan is executable;
one genuine correctness bug remains, plus two must-fix practical items and several
consistency/precision points.

## Confirmed against code (round-2 concerns that resolve)

- **Held-out supply (R2-9) is not a risk.** `Data_Loader/Data/edufineweb_val_00000.bin` is 200 MB
  uint16 = 100 M tokens (matching the loader's `max_tokens_=1e8` cap). s=64 needs
  `(8+32)*262144 ~ 10.5 M` — 10x headroom. Keep the assert as cheap insurance, but record the number.
- `DataLoaderLite::skip_batches()` already exists, so `CP_EVAL_SKIP_WINDOWS` is a 2-line change on
  the loader branch.
- `reset()` sets `pos_ = B*T*rank_` where `rank_ = dp_rank = 0`, so all CP ranks genuinely start at
  the same token — the dp=1 invariant holds concretely, not just by argument.

## New issues created by the round-2 fixes

- **N1 (correctness). `CP_EVAL_SKIP_WINDOWS` covers only one of the two input branches.** The CSV
  eval also has the `CP_EVAL_TOKENS_BIN` bypass, where windows come from `base = w*T` into the token
  file (`:1506`), not from `val_loader` — advancing the loader does nothing there. Needs
  `base = (skip + w) * T_global` and the `max_w` clamp at `:1469` must subtract skip. Otherwise the
  bypass silently re-scores calibration windows: the exact leak, in the HF-comparable path.
- **N2. `FT_STEPS=150` at s=64 collides with the hardcoded `CP_REWARMUP=100`.** 100/300 = 33 % warmup
  at s=32 vs 100/150 = 67 % at s=64 — the token-matching fix creates an LR-schedule mismatch between
  factors. Scale `CP_REWARMUP` to 50 at s=64 and state which invariant is held (tokens, not steps).
- **N3. "Optionally seq-chunk the host copy" should be mandatory.** The 13.2 GB/rank at s=64 is HOST
  RAM (4 ranks concurrently ~ 52.8 GB); `CP_MEM_PROBE` measures GPU only, so item 6 does not cover
  it. Chunking `to_cpu()` over `t` is a few lines and removes the risk.
- **N4. The item-6 gate is too blunt.** "s=64 proceeds only if the probe fits" kills the arm on a
  TRAINING OOM, but search and eval are forward-only. Documented fallback should be `COMPOSE=none`
  at s=64 (search + eval the base ckpt), still a valid searched-vs-formula test provided YaRN is
  also evaluated un-tuned at that factor.
- **N5. The abort floor (~3x seed count = 9 candidates) is too permissive** to support the plan's own
  "seeds-only cannot demonstrate anything". A defensible floor is one full generation past the seeds
  (~ POP + 2*(N1+N2) ~ 112). If that does not fit, cut CALIB or T instead of proceeding.
- **N6. The sliding-4k reference is underspecified** — mean over [0,4096) vs steady-state tail differ
  substantially. The comparison targets the long run's TAIL, so the reference must be the 4k run's
  tail (e.g. mean over [2048,4096)), produced by CP_EVAL_PPL at T=4096, YARN_SCALE=1, same held-out
  windows.
- **N7. Phase 1's framing inverts the claim.** "LongRoPE-no-m vs YaRN-no-m" is the clean MECHANISM
  test (searched lambda vs formula lambda); it is NOT a comparison of the published methods, since m
  is part of YaRN as published and as implemented. Either make the as-published pair the headline
  with no-m/no-m as control, or state plainly that the headline is a mechanism claim.

## Document consistency

- **N8. The revisions are a changelog, not applied edits.** Verification section 5's command block
  still trails `&` on both runs, still says they "collide on longrope_best.txt" (retracted by item
  12), and still omits `FT_STEPS=150` for s=64 (item 3). Executing the block verbatim gives
  pre-review behavior. Fold items 3 and 12 in, or mark the block superseded.
- **N9. Context contradicts protocol.** Line 8 asserts "formula-breakage is model-size-robust" as
  fact while line 117 calls it "a hypothesis, not a given"; line 10 says the fine-tune "does not fit
  one 48 GB GPU -> needs CP across 4" while item 6 concedes it may not fit on 4 either.

Priority: N1 (correctness) > N2, N3 (before any run) > N8, N9 (doc) > N4-N7 (judgment).

---

# Round 4 — review of the revised plan (2026-08-05 16:30)

All 9 round-3 items applied, and the command block / Context section were folded in this time (the
round-3 "changelog not edits" complaint is resolved). The plan is ready to implement. Three items
left, two of them concrete.

## Must fix (verified against the script)

- **`CP_REWARMUP` is a hardcoded literal in the pipeline, so `CP_REWARMUP=50` is inert.**
  `run_longrope_pipeline.sh:127` and `:134` pass `CP_REWARMUP=100` inside the `run_bin` argument
  list, and `run_bin` is `env "$@" ... mpirun`, so the explicit literal wins over the inherited env.
  The s=64 warmup-halving fix does nothing until the script uses `CP_REWARMUP=${CP_REWARMUP:-100}`.
  Add that to Change 4 (`FT_STEPS`, `CALIB`, `EVAL_WINDOWS` are already overridable at `:32`, `:51`,
  `:34`; `GPUS` and `CP_REWARMUP` are the two that are not).
- **The YaRN arm does not exist in the pipeline and is under-specified.** The script has no YaRN
  branch at all — Change 4 says only "add a matched YaRN fine-tune step". Phase 1 now needs TWO YaRN
  arms (+m and no-m) at the same `S`/`COMPOSE`, and the output name
  `ppl_longrope_x${S}_${COMPOSE}${MTAG}.csv` has no arm field, so they would overwrite each other.
  Needs an explicit `ARM={longrope|yarn}` switch that (a) sets `YARN_SCALE=$S
  YARN_ORIG_MAXPOS=$ORIG_MAXPOS` and omits `CP_LONGROPE_FACTORS` on the YaRN arm, (b) skips the
  search phase for YaRN, (c) adds the arm + no-m tag to `OUT` and to `FT_RUN` numbering.

## Should add (cheap pre-flight)

- **Dump the YaRN cache at s=32 and s=64 before the science runs.** `CP_DUMP_ROPE` already exists
  (added 2026-07-23). `find_correction_range` throws on a degenerate range ("R5 bad-range"), and the
  ramp indices shift with `s`; a 30-second dump confirms no throw and a sane `m` at both factors
  before an 8 h search depends on it.
- **Disk budget.** Each arm copies the whole base run (`cp "$CKPT_DIR"/..._run${BASE_RUN}_*`) and
  then writes its own checkpoints. Phase 1 is 3 arms x 2 factors = 6 branches; worth a line so the
  box does not fill mid-run.

## Accounting / consistency

- Round-2 item 10 still says Phase 1 = "1 search + 2 matched FT + 2 evals", but the round-3 matrix
  makes Phase 1 three cells (LongRoPE-no-m, YaRN+m, YaRN-no-m) => 1 search + 3 FT + 3 evals.
- Verification section 2 still lists only the CP=1 off-mode repeat; the contiguous parity at
  `CP_SIZE=2` (round-2 item 5) was never folded into the body.

Verdict: no correctness defects remain in the design. The two must-fix items are wiring gaps in the
pipeline script, not flaws in the plan's logic.


