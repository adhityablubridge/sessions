# Session report - 2026-08-31 - IN2 arm rerun, reach measurement, and a metric-orientation correction

Workspace: BluTrain / dist/Context_Parallelism (submodule)
Hardware: 4x H200 for the rerun, then 1x H100 for the probe-position sweep
Files: run_in2_experiment.sh, Tests/bluscriptcp/needle_gen.py, needle_score.py,
       needle_sweep.sh, probe_1b.sh, in2_gen.py, run_probepos_experiment.sh (new)

## 1. Why the first attempt failed, and the five driver fixes

The 2026-08-30 attempt trained all three arms and produced nothing. Two independent causes:

- Disk. One checkpoint step costs 17 GB across 4 ranks (3.9 GB model + 2 zero_g sidecars,
  per rank). At CP_CKPT_FREQ=100 x KEEP=2 x 3 arms that is up to 102 GB on a 193 GB disk.
  Saves then failed with a bare [WARN] while training continued, so arm C ran all 400
  steps, reported "done", and had zero checkpoints on disk.
- The learnability gate. val_loss is forward-filled and CP_VAL_FREQ defaults to 250, so a
  400-step arm yields as few as one real evaluation. Comparing first to last of a
  forward-filled column reported "did not fall" and aborted the run after every arm had
  already trained.

Fixes applied to run_in2_experiment.sh:
1. CP_CKPT_KEEP=1, CP_CKPT_FREQ=200 (peak disk bounded at one checkpoint per arm).
2. Free-space preflight before each arm; abort under 40 GB.
3. Prune orphaned zero_g sidecars between arms. CP_CKPT_KEEP prunes .ckpt files but not
   the sidecars, which accumulated 50 GB silently on the earlier 600M run.
4. Verify the checkpoint file exists after each arm, and abort if any "checkpoint save
   failed" appears in the log. A zero exit code is not proof the artifact exists.
5. Learnability gate requires >= 2 distinct validation points before judging a trend.

All five held. Arm B reproduced its first-attempt loss to four significant figures
(1.451 vs 1.450279), which matters because the IN2 result is a difference between arms.

## 2. Results

Final training loss, 400 steps each, all initialised from run7_step_750:
  arm A (plain text, no injection)   1.349215
  arm B (needles at SHORT distance)  1.450279
  arm C (needles at IN2 far distance) 1.536691

Reach at T=16384, measured against a shared absolute NLL bar (11.055) rather than the
pipeline score, because the score's floor is defective (section 3):

  run8  4k base, probed at 16k with YaRN cache only, no extension training   5,271
  run7  + 750 steps YaRN 16k extension training                              6,047
  armA  + 400 steps 16k plain text                                           7,435
  armB  + 400 steps 16k, needles adjacent                                    5,924
  armC  + 400 steps 16k, needles far (IN2)                                  12,976

  YaRN extension training   run8 -> run7 :  +776   (1.15x)
  IN2 vs matched control    armB -> armC : +7,052  (2.19x)
  IN2 vs the YaRN base      run7 -> armC : +6,929  (2.15x)

arm B vs arm C is the clean comparison: same init, steps, batch, LR, YaRN, and
bit-identical answer-position histograms by construction in in2_gen.py. Only needle
distance differs. IN2 contributed roughly 9x more reach than YaRN extension training did,
on a quarter the step count.

## 3. Two defects found in the measurement, not the model

Metric orientation (my error, corrected mid-session). needle_gen.py defines p_frac as
p=0 -> needle maximally far from the probe, p=1 -> needle adjacent (the retrieval floor).
Reach is therefore (1 - p_crossing) * T: a LOWER crossing p means retrieval over a LONGER
distance. I initially read low p as "fails early" and reported arm C's reach as 958-3,407
tokens, i.e. the worst arm, when it is the best by more than 2x. Any future analysis must
respect this orientation.

Floor defect. needle_score.py:41 hardcodes FLOOR = "p1" and computes
score = (nll_absent - nll) / (nll_absent - nll_floor). This assumes p1 is the easiest
case. Mean answer-span NLL, block b0:

           p0.75    p1     floor holds
  base     9.723   7.939   yes
  armA    10.041   8.306   yes
  armB    10.018  11.738   NO
  armC     6.217  12.033   NO

Arms B and C are about 4 nats WORSE with the needle adjacent than 4,087 tokens away, so
denom collapses (1.606 and 1.772 vs base 6.266) and scores exceed 1.0, producing the
corrupt pipeline reach of 958 for arm C. Verified this is not a scoring-window bug: the
answer span is read at the correct rows, and mid-document filler NLL is 3.53 in every arm,
so general LM ability is intact. The models really did lose adjacent retrieval while
gaining distant retrieval. Arm A, same steps and LR on plain text, did not degrade.

Unexplained: arm B trains with the needle immediately before the probe (ns = p - span),
which is exactly the p1 eval layout, yet fails at it. Out-of-distribution at test time
explains arm C (D_MIN=64 means distance 0 is never seen) but not arm B.

## 4. Confound in the whole sweep

With probe_pos unset - the default, and what probe_1b.sh uses - the probe sits at the end
of the trial, so needle absolute position and needle-probe distance are perfectly
anti-correlated. needle_gen.py's own docstring flags this: it makes "far from the question"
and "early in the document" indistinguishable. No U-shape (lost-in-the-middle) can be
detected by the sweep as run. All four models show monotone recency with no primacy bump
at p0, but that observation cannot distinguish position from distance.

## 5. Running now

run_probepos_experiment.sh on 1x H100, single GPU (probes are already CP=1, inference
only). For PROBE_POS in {8192, 12288} and models {base, armB, armC}, variants are
p<fixed-distance> p1 absent null, 3 blocks. This answers both open questions with one job
set: p1 at each position is the adjacency-vs-end-of-document diagnostic, and the
fixed-distance variant (D=4087) with a moving probe breaks the position/distance confound.

## 6. Also worth fixing

- probe_1b.sh:25 hardcodes MODEL_KEY=1b_run8, so every reach.csv claims to be 1b_run8
  regardless of which checkpoint was scored. All five existing reach.csv files carry the
  wrong model id.
- Arm LR was CP_MAX_LR=1e-3 with CP_REWARMUP_PEAK=0.3, so an effective peak of 3e-4
  (confirmed in the logs: step 99 lr 3.0000e-04). That is 0.3x the base pretraining LR and
  aggressive for a 400-step context adaptation; 0.1x is the more common choice and is a
  candidate cause of the lost adjacent retrieval.

## 7. Archived

Everything is on HuggingFace at unparallelled/BluScriptCP: source tarball, full vault,
driver and training logs, all training CSVs, all five probe directories with reach.csv,
and four checkpoints (arms A/B/C plus the extended base) under checkpoints/in2/.
