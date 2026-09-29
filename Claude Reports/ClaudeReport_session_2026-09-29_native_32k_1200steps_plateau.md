2026-09-29 - 14:32 - 1.59B native-position 32k extension carried to 1,200 steps: the curve plateaus at 37-40 against YaRN's 84.8 - BluTrain / stage_native/run_native_seg3.sh, rerun_seg3_eval.sh

# Session report

## Headline

Three 400-step segments of NATIVE-POSITION training (YARN_SCALE=1, true angles,
theta 500,000 unchanged, constant lr 1.2e-4) carried the 1.59B from step 8600 to
9800. The reach curve at 32,744 PLATEAUS.

| segment | steps | mean acc @32,744 | delta |
|---|---|---|---|
| 1 | +0 .. +400 | 17.6 | - |
| 2 | +500 .. +800 | 36.1 | +18.6 |
| 3 | +900 .. +1200 | 38.4 | +2.2 |

Segment 3 bought +2.2 points. n=100 points carry about +-9 at these rates, so
that is indistinguishable from zero. Native extension SATURATES near 37-40.

After segment 2 I read the slope as "+11 points per 400 steps, maybe four more
segments to reach 84.8". That was an extrapolation from one segment and it bent
immediately -- the same error shape as the 2026-09-26 two-point extrapolation,
one scale larger.

![[fig1_plateau.png]]

## Final cells at step 9800, native positions

| length | seg 1 (+400) | seg 2 (+800) | seg 3 (+1200) | YaRN x8 |
|---|---|---|---|---|
| 4,096 | 99.0 | 97.0 | 98.0 | 100.0 |
| 8,192 | 88.0 | 97.0 | 98.0 | 100.0 |
| 16,360 | 50.0 | 67.0 | 71.0 | 100.0 |
| 32,744 | 27.8 | 38.5 | 40.5 | 84.8 |

32,744 at n=400: Wilson 95% [35.8, 45.4] against the x8 rung's
84.8 [80.9, 87.9]. Non-overlapping.

![[fig2_cliff.png]]

### depth at 32,744

| depth | hits | pct |
|---|---|---|
| 0.0 | 0/80 | 0.0 |
| 0.25 | 0/80 | 0.0 |
| 0.5 | 5/80 | 6.2 |
| 0.75 | 77/80 | 96.2 |
| 1.0 | 80/80 | 100.0 |

## Full curve at 32,744

| steps into the native run | accuracy |
|---|---|
| +0 | 0.0 |
| +100 | 20.0 |
| +200 | 20.0 |
| +300 | 20.0 |
| +400 | 27.8 |
| +500 | 36.0 |
| +600 | 39.0 |
| +700 | 31.0 |
| +800 | 38.5 |
| +900 | 38.0 |
| +1000 | 40.0 |
| +1100 | 35.0 |
| +1200 | 40.5 |

![[fig3_native_vs_yarn.png]]

## What is settled

THE RULER DID EXTEND. 32,768 against 4,096 for every YaRN rung -- x4, x8 and the
680M's x32 all keep 4,096 BY CONSTRUCTION. Dims completing a full rotation went
32/64 -> 42/64. Grad norm met the untrained positions at 14.8508 and decayed 97%
within 19 steps to 0.05-0.08, below the x8 rung's 0.0967 starting value. That
retires the 2026-09-25 decision to adopt YaRN on a first-step grad norm of
1.3248: it was a transient, eleven times smaller than what this run absorbed in
twenty steps, and four rungs were built on it.

IT DOES NOT BUY RETRIEVAL AT THIS BUDGET. 1.26B tokens on the new region against
9.02B on the original 4,096 yields less than half YaRN's score, and the curve is
flat over the last 400 steps. The constraint is not step count at this scale.

## Data position

The corpus is 419.7M tokens = exactly 400 steps per epoch. Three segments = 3.00
epochs, inside the ~4-epoch band where Muennighoff et al. (arXiv:2305.16264) find
repeated data roughly equivalent to fresh. A fourth segment would sit exactly at
the ceiling, and the curve says it would not move.

Only 178 steps of genuinely fresh >=32,768 material remain in the HF repos,
commoncrawl-bound (measured yields over 183 shards: cc 2.471%, edu 1.817%,
code 21.643%, math 12.086%).

## Errors made this session

- Derived run_native_seg3.sh from run_native_cont.sh by sed and did not rename
  the log path, so segment 3 appended to run_native_cont.log while post_seg3.sh
  waited on run_native_seg3.log. Left alone the chain would have polled five
  hours and died with FATAL. Repaired with a symlink.
- Gated the post chain on `[ -d blumodel_step_9800 ]`, which goes true when the
  directory is CREATED, not when the checkpoint is complete. dcp2ckpt read a
  partial checkpoint, printed "convert FAILED 9800", and all four eval cells
  skipped -- while run_longeval.sh printed "DONE -> summary.csv" over empty
  files. Cost ~40 minutes; no data lost.
- The `c2 -> c3` global rename in post_seg3.sh pointed the +800 curve line at
  segment 3's file. Caught before launch.
- Told the user after segment 2 that the curve was climbing at ~+11 points per
  400 steps. Segment 3 measured +2.2.

## Next

The question is no longer "more steps". Either native extension needs a budget
of a different order -- Llama 3.1 staged 8K->128K over far more than 1.26B
tokens -- or something else in the setup binds. Two candidates never tested:
ORIG_MAXPOS raised from 4,096 with a correspondingly lower factor (a middle
ground between native and x8), and a lower pretrain theta, which is a
pretraining-time decision and would need the 680M to test affordably.
