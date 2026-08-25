# Claude Report — bluscriptCP re-warmup anchor fix + long-context PPL eval

2026-07-16 - LR re-warmup resume fix and PPL-vs-position long-context eval mode — Workspace: CP / Files: Scripts/Blutrain/bluscriptCP.cpp, graph_bluscript_cp.ipynb

## Context

Continued the YaRN long-context work. Two needs surfaced while running a 114M
context-extension (pretrain T=4096 -> extend to T=16384, YARN_SCALE=4): (1) resuming
an interrupted extension run restarted the LR warmup; (2) no way to check whether the
extension actually worked. Both addressed in bluscriptCP (server-run; verified locally
by compile only).

## 1. Re-warmup anchor fix

Problem: the 2-stage re-warmup (`CP_REWARMUP`) anchored the fresh warmup ramp at
`start_step` = the step resumed from. Correct on the first extension resume, but every
subsequent resume re-anchored a new ramp at the current checkpoint step -> warmup
restarted each time.

Fix: added `CP_REWARMUP_ANCHOR=<step>` to pin the ramp to a fixed absolute step
(the step the extension warmup first began at = the base checkpoint step). Default -1
falls back to the old `start_step` behavior. `get_lr` computes `local = step - anchor`,
so resuming mid-ramp continues the ramp and resuming past `anchor+rewarmup` is already
in cosine-decay. The `[re-warmup]` line now prints anchor, resume step, and phase
(`mid-ramp` / `past ramp, in cosine-decay` / `BEFORE anchor`).

Usage (continue an extension that ran 1087->1175, target 1413, base stopped at 1087):
```
CP_T=16384 YARN_SCALE=4 YARN_ORIG_MAXPOS=4096 CP_CKPT_RESUME=<ext_run> \
CP_MAX_STEPS=1413 CP_REWARMUP=36 CP_REWARMUP_PEAK=0.3 CP_REWARMUP_ANCHOR=1087 \
CP_CKPT_FREQ=25 make CP_FUSED_ROPE=1 CP_SIZE=2 run-bluscript-cp NP=4
```
`CP_MAX_STEPS`/`CP_REWARMUP`/`CP_REWARMUP_PEAK` must stay identical across resumes.

## 2. Long-context PPL-vs-position eval mode (`CP_EVAL_PPL`)

Rationale: the standard "did extension work" check. PPL binned by token position over
long windows must stay flat past the original context (extension worked) rather than
blow up (naive extrapolation). Frontier labs use RULER / needle-in-a-haystack for
"effective context length"; a 114M from-scratch model is capability-limited, so
PPL-vs-position + the naive-extrapolation contrast is the trustworthy signal at this
scale (passkey deliberately skipped).

Implementation (additive, forward-only, exits before training):
- Gated by `CP_EVAL_PPL=1`; requires `NP=1 / CP_SIZE=1` so `model.forward` returns full
  `[B,T,vocab]` and position `t` maps directly to the global index (no cross-rank NLL
  gather). Enforced with a runtime check.
- Loads the checkpoint via the existing `CP_CKPT_RESUME=<run>` + `CheckpointManager`
  (loads that run's latest step). Checkpoint stores only T-independent params (no `wpe`;
  RoPE has none), so a BASE checkpoint loads fine at T=16384.
- Teacher-forces `CP_EVAL_WINDOWS` (default 10) contiguous windows of length T from the
  held-out `val` split (unseen data). Per position: `NLL = logsumexp(logits) -
  logits[target]` (stable, over full padded vocab 50304). Accumulates per-position
  sum/count -> writes `pos,count,mean_nll,ppl` to `CP_EVAL_OUT` (default ppl_vs_pos.csv).
- Skips training-log CSV and config writes (redirects both ofstreams to /dev/null in
  eval mode) so it doesn't pollute the run's training logs.
- `#pragma omp parallel for` over positions (each `t` written by one thread; harmless
  no-op without -fopenmp, just serial). Serial cost ~minutes/window at V=50304.

The YaRN cache is set by env at model construction (`build_rope_cache` reads YARN_*), so
the base-vs-extended comparison is produced by running the same mode per config:
- BASE ckpt + `YARN_SCALE=1` @ T=16384  -> naive-extrapolation baseline (expect blow-up)
- EXT  ckpt + `YARN_SCALE=4 YARN_ORIG_MAXPOS=4096` @ T=16384 -> YaRN (expect flat)

## 3. Notebook cell (graph_bluscript_cp.ipynb)

Appended a cell that reads the eval CSVs (scp'd from server), plots smoothed PPL vs
position on a log-y axis with a vertical marker at the original context length, saves
ppl_vs_position.png, and prints a verdict table (mean PPL before vs after orig ctx +
ratio; ratio ~1 = flat = worked, >>1 = blow-up). Existing plotting cells untouched.

## Verification

- `mpic++ -std=c++2a -fsyntax-only` on bluscriptCP.cpp: clean.
- `mpic++ -c` full compile-to-object (codegen/template instantiation): clean, exit 0,
  16MB .o. Local *link* not attempted (the `CP_FUSED_ROPE` fused-kernel symbol is built
  on the server, not in this checkout).
- Notebook JSON valid (7 cells).
- Runtime (server, 114M @ 16384) pending — cannot run here (checkpoints are on the
  server; 114M @ 16384 exceeds the local 2x3060).

## Files touched

- `Scripts/Blutrain/bluscriptCP.cpp` — `CP_REWARMUP_ANCHOR` in the LR-schedule block;
  `eval_ppl_mode` flag; eval branch before the training loop; /dev/null guards on the
  config + log ofstreams in eval mode.
- `graph_bluscript_cp.ipynb` — appended PPL-vs-position plotting + verdict cell.

## Follow-ups

- Run the two eval configs on the server, scp CSVs, run the notebook cell.
- If PPL eval is too slow serially, build bluscriptCP host objects with -fopenmp (the
  pragma is already in place) — Makefile CXXFLAGS change.
- Optional: persist the re-warmup anchor into the checkpoint so `CP_REWARMUP_ANCHOR`
  need not be passed manually on each resume.
