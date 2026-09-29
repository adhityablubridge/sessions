2026-09-28 - 19:10 - 1.59B native-position 32k extension: the first RoPE ruler longer than 4,096, and why it still loses to YaRN at this budget - BluTrain / stage_native/run_native32k.sh, post.sh, ship_now.py

# Session report

## What was run

4,096 -> 32,768 with NATIVE POSITIONS. YARN_SCALE=1, so yarn_rope_cache returns
autograd::build_rope_cache unchanged and every position receives its TRUE angle
p*theta_i. theta 500,000 unchanged, constant lr 1.2e-4, 400 steps, and the SAME
>=32,768 60/40 corpus the x8 rung trained on -- so YARN_SCALE is the only variable
that differs from the run that scored 84.8.

Parent: run_2026-09-25/dcp_control_step_8600, a 1.59B at context_length 4,096,
theta 500,000, never YaRN'd. Positions 0..4,095 therefore kept exactly the angles
they were trained on and 4,096..32,767 were new: a pure ruler extension with no
remapping. (step_8000 was never shipped to HF; this was the only un-compressed
1.59B checkpoint in existence.)

## The headline: the ruler moved, and it was not enough

| checkpoint | ruler | dims completing a full turn |
|---|---|---|
| base / control 8600 (4k) | 4,096 | 32/64 |
| ext8190 (16k, YaRN x4) | 4,096 | 32/64 |
| ext8590 (32k, YaRN x8) | 4,096 | 32/64 |
| ext8990 (32k, YaRN x8, 100% long) | 4,096 | 32/64 |
| **nat9000 (32k, NATIVE)** | **32,768** | **42/64** |

Every YaRN rung keeps a 4,096 ruler BY CONSTRUCTION -- that is exactly what YaRN
guarantees, and it is why the ruler never moved across x4, x8 or the 680M's x32.
Dims 32..41 completed a rotation for the first time in this run.

![[fig1_native_vs_yarn.png]]

## NIAH at native positions (YARN=1)

| length | n | native | Wilson 95% | x8 YaRN rung |
|---|---|---|---|---|
| 4,096 | 100 | 99.0 | [94.6, 99.8] | 100.0 |
| 8,192 | 100 | 88.0 | [80.2, 93.0] | 100.0 |
| 16,360 | 100 | 50.0 | [40.4, 59.6] | 100.0 |
| 32,744 | 400 | **27.8** | [23.6, 32.3] | **84.8** [80.9, 87.9] |

NATIVE LOSES DECISIVELY AT THIS BUDGET. 27.8 against 84.8, intervals nowhere
near overlapping.

![[fig2_depth_cliff.png]]

### depth at 32,744 -- a cliff, not a slope

| depth | hits | pct |
|---|---|---|
| 0.00 | 0/80 | 0.0 |
| 0.25 | 0/80 | 0.0 |
| 0.50 | 0/80 | 0.0 |
| 0.75 | 31/80 | 38.8 |
| 1.00 | 80/80 | 100.0 |

Perfect retrieval from the last quarter of the context, ZERO from the first three
quarters. Read together with 99.0 at 4,096 and 88.0 at 8,192, the honest summary
is that after 400 steps the USABLE ruler is about 8,192, not 32,768. It grew 2x,
not 8x.

![[fig3_training.png]]

## Grad norm -- the measurement the run was designed around

14.8508 at step 8600, 0.5088 by 8619, 0.0968 by 8679, 0.05-0.08 thereafter.
A 97% decay in 19 steps, settling BELOW the x8 rung's 0.0967 STARTING value.

That retires a decision made on 2026-09-25: at 16k I measured grad norm 1.3248
without YaRN versus 0.2996 with, read it as instability, and chose YaRN for every
rung after. It was a transient, and 1.3248 is ELEVEN TIMES SMALLER than what this
run absorbed inside twenty steps. The proxy was wrong; four rungs were built on it.

## Reach vs steps at 32,744 -- still rising

| checkpoint | steps | accuracy |
|---|---|---|
| 8600 | +0 | 0.0 |
| 8700 | +100 | 20.0 |
| 9000 | +400 | 27.8 |

The 4,096 parent scores EXACTLY ZERO at 32,744 native, as it must -- those
positions are entirely untrained. 400 steps took it to 27.8 and the curve has not
flattened. Contrast the x8 rung, which plateaued by step 210 (61.0 -> 84.0 -> 84.8).
So native is UNDER-TRAINED, not wrong: the new 87.5% of the ruler has had 0.047x
the training the original 4,096 received (400 steps / 0.42B tokens against
8,600 / 9.02B).

Points at +200 and +300 were launched but killed by the teardown; their eval
directories exist with no summary.csv.

## What this means

YaRN compresses 32,744 real tokens into a 4,096-wide angular range the model
already understands, and gets 84.8 for almost no adaptation cost. Native training
must teach 32 dimensions an 8x wider angular range from scratch, and 0.42B tokens
buys only the first 8k of it. Both facts can be true: YaRN is the better use of a
small budget, and only native training can ever produce a ruler longer than 4,096.

The open question is where the curves cross. That needs more steps at
YARN_SCALE=1, and the curve slope says the information is worth buying.

## Errors made this session

- Guarded the link step with `pgrep -f "blurun-build"`, which matched the shell
  that WROTE the script -- its command line contains the heredoc, including that
  string. The guard never cleared, the exe never linked, the GPUs idled ~6 min.
- Wrote the eval wait as `pgrep -x bluscript_zero_ckpt_exec`. Linux truncates
  comm to 15 chars and that name is 24, so it can never match; pgrep prints a
  warning saying so. The loop fell through instantly, the eval died on a missing
  step_9000 at 15:26, and the curve blocked behind it forever. Both rewritten to
  file-based waits.
- Staged an eval that scored only step 9000, which would have lost the
  reach-vs-steps curve at teardown. The user caught this, not me.
- Predicted a native run "could land below 84.8 and still be the right
  direction". Direction right, magnitude badly understated: it landed at a third.

## Hardware note for whoever picks this up

The restored instance is 1x RTX PRO 6000 Blackwell Server Edition, 97,887 MiB,
compute capability 12.0 -- NOT the 2x H200 (143,771 MiB, sm_90) this work ran on.
Every binary here was compiled -gencode arch=compute_90a,code=sm_90a and will not
execute on sm_120 without a full rebuild. The 32k config also needs 117.6 GiB per
GPU, which exceeds 97.9 GiB, so it will not fit at world=1 without activation
checkpointing.

## Next

- More steps at YARN_SCALE=1, T=32,768. The curve is rising and has not plateaued.
- Memory, anchored to the measured 117,625 MiB/GPU at 32k on H200: 64k needs
  217 GiB without activation checkpointing but 44 GiB with it; 128k needs 422 GiB
  without, 76 GiB with. The 64k/128k "memory wall" is a flag never turned on.
  Cost +33% compute: 64k 400 steps ~6.0 h, 128k ~10.3 h on 2x H200.
- 128k remains DATA-blocked regardless: 10,997 shards / 2.0 TiB by filtering.
