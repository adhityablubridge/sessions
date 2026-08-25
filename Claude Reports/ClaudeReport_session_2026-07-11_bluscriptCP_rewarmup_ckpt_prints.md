# Claude Report — bluscriptCP 2-stage re-warmup + checkpoint-visibility prints

2026-07-11 - Added a 2-stage (context-extension) re-warmup LR path and checkpoint-visibility prints to bluscriptCP; IMPLEMENTED but NOT built/verified (a run was occupying the box) - Workspace: CP / File: Scripts/Blutrain/bluscriptCP.cpp

## Context

Discussion around YaRN context extension (CP_T 1024 -> 4096, YARN_SCALE=4). Established:
- Extending context causes a transient train/val perplexity spike (longer seqs + rescaled RoPE +
  YaRN attention temperature m). Meeting it at full/peak LR risks a loss spike/divergence.
- Recommended recipe: stop the base-context run DECAYED (not at peak), then run the long-context
  phase as its own schedule with a fresh short re-warmup to a REDUCED peak (~0.1-0.3x max_lr).
- Correction to an earlier claim: the existing `get_lr` warmup is ABSOLUTE from step 0, so it
  CANNOT re-ramp at a resume step. A resume-time re-warmup genuinely needed a code change.
- Separately, the user's "not checkpointing" was just the default `ckpt_freq=250` (first save at
  step 250); and the boxed Configuration block / startup did not surface checkpoint state.

## Changes (Scripts/Blutrain/bluscriptCP.cpp)

1. `get_lr` gained an optional `warmup_start` anchor:
   - ramp over `[warmup_start, warmup_start+warmup)`, then cosine-decay to `min_lr` by `max_steps`.
   - `warmup_start=0` reproduces the previous schedule byte-for-byte (fresh runs unaffected).
   - Guards: peak < min_lr clamped to constant min_lr; warmup window past max_steps avoids /0.
2. Two env knobs (2-stage re-warmup), read near the other CP_* env:
   - `CP_REWARMUP=<steps>` (default 0 = off). Only active when resuming (`start_step > 0`).
   - `CP_REWARMUP_PEAK=<fraction of max_lr>` (default 1.0) for a reduced long-context peak.
3. Effective-schedule vars computed after checkpoint load (start_step known), before the loop:
   `rewarmup_active = (rewarmup_steps>0 && start_step>0)`; `lr_warmup_start / lr_warmup / lr_peak`
   selected accordingly; the loop's `get_lr` call now passes these. Prints
   `[re-warmup] anchored at step S for W steps -> peak <lr> (<frac> x max_lr), then cosine-decay ...`.
4. Checkpoint-visibility prints:
   - boxed Configuration block: `checkpointing : true/false (ckpt_freq=..., dir=...)`.
   - run start (rank 0): `Checkpointing: ON -> <dir>/<prefix>_step_<N>.ckpt (every F steps, keep K)`
     or `Checkpointing: OFF`.

## Intended usage

Stage 1 (base, let it decay): normal run at CP_T=1024 to a decayed LR, checkpointing on.
Stage 2 (extend + re-warmup):
```
CP_T=4096 YARN_SCALE=4 YARN_ORIG_MAXPOS=1024 \
CP_CKPT_RESUME=<run> CP_MAX_STEPS=<new_target> \
CP_REWARMUP=<W> CP_REWARMUP_PEAK=0.3 \
  make CP_FUSED_ROPE=1 run-bluscript-cp NP=<n>
```

## Status / next

- NOT built or run (user had a run in progress; explicitly asked not to run anything).
- Pending: `make CP_FUSED_ROPE=1 bluscript-cp`, then a world=1 resume smoke (checkpoint a few
  steps, resume with CP_REWARMUP=5 CP_REWARMUP_PEAK=0.3) to confirm the `[re-warmup]` line + the
  ramp-from-resume-step behavior, and that fresh-run LR is unchanged (warmup_start=0 path).
- Static consistency checked (declarations precede loop/use; call site updated; both prints present).

## README (added same session)

Documented in README.md (bluscriptCP section):
- New **Checkpointing / resume / LR re-warmup** env table: `CP_CKPT`, `CP_CKPT_FREQ` (note: first
  save at step N — default 250), `CP_CKPT_NEW_RUN`, `CP_CKPT_RESUME`, `CP_REWARMUP`,
  `CP_REWARMUP_PEAK`; plus run-numbered checkpoint/log naming.
- New **2-stage context extension (YaRN + re-warmup)** subsection: base-stage-decay -> long-context
  resume with reduced-peak re-warmup, with the resume command; notes that params are
  seq-len/world-size-independent (RoPE computed, not learned) so the extended CP_T / different world
  size loads the same checkpoint, and the CP_MAX_STEPS/CP_CKPT_RESUME resumability gate.
- YaRN note: `scale=1.000` at init means YaRN off; keep `CP_T == YARN_SCALE*YARN_ORIG_MAXPOS`.
