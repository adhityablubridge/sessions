# ClaudeReport_session_2026-08-10_longrope_search_ce_bug

2026-08-10 - LongRoPE search: candidate-parallel runs, unattended runner, and a size-dependent
fault in the search fitness - CP / Scripts/Blutrain/bluscriptCP.cpp,
Tests/bluscriptcp/longrope_search.py, Tests/bluscriptcp/longrope_progressive.sh,
Tests/bluscriptcp/longrope_autorun.sh, .gitignore

## Headline

The LongRoPE evolutionary search DOES beat the formula seeds. Earlier runs that converged on
their own NTK seed were not a property of the method - two separate harness problems masked it,
one of which was a genuine correctness fault in the search fitness.

48M (run200) @ 64k / 16x, calibration windows [36,40), 813 evals / 30 generations:

| arm | calibration | held-out [40,48) |
|-----|-------------|------------------|
| LongRoPE searched | 178.51 | **165.23** |
| NTK | 188.39 | 173.69 (+5.1%) |
| YaRN | 208.26 | 191.49 (+15.9%) |
| PI | 312.31 | - |

93% of the calibration gain transferred to held-out data. This result is VALID: it ran at
T=65536, below the fault boundary described below.

## The fault: whole-tensor CE in the search fitness

The search fitness called `sparse_cross_entropy_loss` once over the entire
`[B, T_local, V]` logits tensor. That is correct while the element count stays under 2^32 and
silently wrong above it. Boundary = 2^32 / 50304 ~= 85,400 tokens.

Isolated by a both-paths reconcile sweep (search evaluator vs CP_EVAL_PPL CSV) at CP=1 with
identical windows, cache, and checkpoint:

| T | elements | CSV path | search path | |
|---|----------|----------|-------------|---|
| 16,384 | 8.24e8 | 54.9950 | 54.994995 | agree |
| 32,768 | 1.65e9 | 51.0084 | 51.008415 | agree |
| 65,536 | 3.30e9 | 55.2463 | 55.246292 | agree |
| 131,072 | 6.59e9 | 59.8720 | **7.55** | **8x divergence** |

The CSV path escapes it because it copies 4096 positions at a time with `size_t` pointer math;
the largest single index it forms is ~2.1e8.

### Fix

Chunk the CE over the sequence (`CP_SEARCH_CE_CHUNK`, default 16384 -> 8.2e8 elements/call) and
token-weight the per-chunk means. `T_local <= chunk` still takes the original single call.

Validated on the server after rebuild:
- T=65536: 55.246292 - **bit-identical** to the pre-fix single call, though it now goes through
  4 chunks. Proves the token-weighted mean is exact.
- T=131072: 7.55 -> **59.871937**, matching the CSV path's 59.8720 to 6 significant figures.

### Scope of the damage

- Affected: the two 114M 128k searches only. Both re-run.
- NOT affected: the 48M 64k search (below the boundary), every CSV/eval curve including
  `ppl_prog_s32.csv` and the 32x extension result, and all fine-tunes (they never use the
  search fitness).

Why it was never seen before: training at 128k runs CP=8, so the local shard is 16k. Only a
forward-only search at CP=1 puts 6.6e9 elements through the loss in a single call.

## Hypotheses ruled out on the way (recorded so they are not re-litigated)

1. **Easy calibration slice** - token statistics of the calibration region are normal
   (~16k unique, top token ~4%, rep@1k ~0.008), indistinguishable from the eval region.
2. **Jensen / arithmetic vs geometric mean** - the notebook reports `ppl.mean()` (59.61) while
   the search reports `exp(mean nll)`. But the count-weighted geometric aggregate of the same
   CSV is 50.79, not ~7.3, so aggregation did not explain the gap.
3. **CP topology / the known multi-rank ring deviation** - the divergence reproduces at CP=1 on
   both sides.
4. **CP_EVAL_SKIP_WINDOWS handling** - both paths agree at 16k WITH skip=36 (54.995), so the
   loader advances identically.
5. **Stale binary / env not propagating** - workers print `skip=36`; `strings` confirms the
   symbol; `/proc/<pid>/environ` confirms the controller.
6. **int32 (2^31) overflow** - T=65536 is already above 2^31 elements and agrees exactly. The
   boundary is 2^32, not 2^31.

## Other work this session

- **Candidate-parallel search**: `EvaluatorPool` + `--pool N`, result-identical to serial
  (bit-identical gate + mock orchestration test). GPU count can change between resumes:
  `pool x CP_SIZE = len(devices)`, non-contiguous IDs fine, memo/state make it seamless.
- **Incremental memo writes**: results now persist as each candidate finishes rather than after
  the batch barrier. A kill previously discarded an entire generation - gen 0 is a single
  POP-sized batch that can take 2 h. Determinism preserved (memo is keyed by genome, so jsonl
  line order never reaches the GA).
- **`TOPK` / `PMUT` exposed** in `longrope_progressive.sh`. The arg defaults (32 of POP=48 =
  67% breed, p_mut=0.3 resampling ~10 of 32 dims) cannot make small moves near a smooth
  optimum; 24 / 0.1 restores the paper's 50% selection ratio and refines locally. This is what
  let the 48M search beat its NTK seed.
- **`longrope_autorun.sh`** (new): polls until every requested GPU has >= MIN_FREE_MB free for
  STABLE_CHECKS consecutive polls (never kills other jobs), then hands off to the progressive
  chain.
- **`.gitignore`**: search artifacts (`longrope_best*.txt`, `longrope_prog_state/`, generated
  `ntk_s*.txt` / `yarn_s*.txt`) were tracked because the pattern still named the old fixed
  `longrope_best.txt`. The state dir is written live during a search, so a stray
  `git checkout`/`git clean` would have reverted or destroyed a running search's memo.

## Follow-ups

1. Re-run the 114M 128k search against the corrected fitness (in flight).
2. Add a **sequence-length** reconcile check (both paths at 16k/64k/128k) to the test suite.
   `cp_parity.sh` covers CP topology but nothing covered T - this fault would have been caught
   on day one.
3. Pre-existing multi-rank ring deviation (CP=4 HeadTail) is still open and unrelated; only
   needed if pure ring at >= 4 ranks becomes load-bearing.
