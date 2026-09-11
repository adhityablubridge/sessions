# 2026-09-09 - l3long step-count arm at 16k, silent checkpoint loss, HF teardown

2026-09-09 - Ran the step-count lever at 16k (run 17 / l3long), found and fixed a
silent checkpoint-loss bug, evaluated at steps 1000 and 2000, then stopped early
and uploaded everything to HF for instance teardown - Context_Parallelism /
b3_long_run.sh, eval_at_1000.sh, eval_at_2000.sh, ckpt_janitor.sh

## Question

l1/l2/l3 all landed on an 8k->16k slope of -18 to -29 pp/doubling and tied at 16k
(l2 35.0, l3 35.0). Document length was exhausted as a lever inside a 16k window.
Step count was the remaining untested one.

## Setup

    parent          blumodelcp_run6_step_750.ckpt (run7', same parent as l3@400)
    context_length  16384  (YaRN scale 4.000 x orig_maxpos 4096)
    data            lendata/b3_16384_32768 (16k-32k docs, TRUNCATED to the window)
    global_batch    524288 tokens (grad_accum 32)
    max_lr/min_lr   6e-4 / 6e-5
    schedule        WSD, warmup 100, stable 3800, decay tail 100 (decay from 3900)
    rewarmup        CP_REWARMUP_PEAK 0.3 -> flat 1.8e-4
    rate            11.56 s/step measured

The "32768" in the data path is the DOCUMENT-LENGTH BUCKET, not the context
window. This is a 16k run.

## NIAH single-needle (%), N=100, b2 val haystack

| prompt_len | armA_c | l1 | l2 | l3 | l3long@1000 | l3long@2000 |
|---|---|---|---|---|---|---|
| 4096  | 73.0 | 75.0 | 80.0 | 74.0 | 65.0 | 68.0 |
| 8192  | 53.0 | 48.0 | 62.0 | 64.0 | 55.0 | 54.0 |
| 16360 | 33.0 | 30.0 | 35.0 | 35.0 | 36.0 | 43.0 |

Step count IS a live lever at 16k: 36.0 -> 43.0 at 16,360 between step 1000 and
2000, +8 pp over the ladder best. Short context recovered (4k 65.0 -> 68.0)
rather than degrading further. Caveats: still 12 pp behind l2 at 4096; +7 pp is
~1.5 sigma at N=100; and EVERY l3long number is UNDECAYED (decay would not have
started until step 3900) while l3/l2 at step 400 are decayed, so the comparison
is biased AGAINST l3long.

## Cross-comparison with the separate 32k extension (ext32k_b4)

| arm | what changed | 4096 | 8192 | 16360 | 32744 |
|---|---|---|---|---|---|
| l2 baseline | --                    | 80.0 | 62.0 | 35.0 | cannot run |
| l3long@2000 | more steps at 16k     | 68.0 | 54.0 | 43.0 | cannot run |
| ext32k_b4   | widened window to 32k | 75.0 | 60.0 | 45.0 | 27.0 |

Both levers buy a similar gain at 16,360 (+8 and +10, neither resolvable at
N=100) but only the widening unlocks 32,744 at all and it costs less short
context. Widening looks the better buy, noting ext32k is decayed.

## Operational finding: SILENT checkpoint loss

CP_CKPT_KEEP prunes .ckpt files but NOT their .zero_g1 optimizer sidecars. At
CP_CKPT_FREQ=25 that leaked ~49 GB of orphans and filled a 193 GB disk. Every
save from step 1275 to 1625 then failed while training carried on normally,
because a failed save only emits

    [WARN] checkpoint save at step NNNN failed: Failed to flush checkpoint to: ...

No crash, no backpressure, no exit code. 75 minutes of checkpointing was lost and
the run would have rolled back to step 1250 had it died. Freed 63 GB (14.3 GB
abandoned .tmp plus the orphaned sidecars) and saves resumed immediately.

Mitigation: ckpt_janitor.sh, a detached 5-minute loop that deletes sidecars with
no surviving .ckpt, removes .tmp older than 10 min, and warns below 25 GB free.
It reclaimed 928 MB per cycle, exactly offsetting the leak.

The real fix belongs in CheckpointManager: prune the sidecar with its .ckpt, and
treat a failed flush as fatal rather than a warning.

## Teardown

Stopped at step 2450 of 4000 at a clean checkpoint boundary. Uploaded to
unparallelled/BluScriptCP and verified every checkpoint by local-vs-remote byte
size, 0 failures: results doc, session report, training + janitor logs, both
eval ladders, 5 scripts, a 123-file source tarball, and checkpoints for steps
1000 / 2000 / 2450 with their optimizer sidecars so the run is resumable.

The final audit caught one genuine gap: l3 (blumodelcp_run1_step_400.ckpt,
8.18 GB) was NOT on HF - the only run1_step_400 there was
lenbucket_l1_blumodelcp_run1_step_400.ckpt at 5.23 GB, a DIFFERENT file. l3 is a
column in every eval table, so teardown would have destroyed a baseline the
conclusions rest on. Now uploaded. Also archived the local dirs with no HF copy
(b3_logs, setup_logs, mspoe_*, probe_run4, context_rot_*,
CP_BluScreipt_Training_logs) as logs/misc_local_artifacts_2026-09-09.tgz.

## My own errors this session

1. Told the user "no 32k extension has been run" when asked where the 32k eval
   was. WRONG - ext32k_b4 had run and been fully evaluated to 32,744 that same
   day; its outputs were on HF and cleared off local disk. I had only checked the
   local directory, where I found the unrun ext_32k.sh (a different variant).
   Checking HF first would have caught it. The user was right.
2. Predicted the resumed LR should read 3.0e-4 and flagged 1.8e-4 as suspect.
   1.8e-4 was correct all along: CP_REWARMUP_PEAK 0.3 x max_lr 6e-4. I had
   assumed max_lr=1e-3; the configured value is 6e-4.
3. Read "resumed: step 1665" as a step number and called it a puzzle. It is
   grep -ac, a LINE COUNT. No progress was lost at either eval.
4. sed-edited eval_at_2000.sh while it was RUNNING, which can desync bash byte
   offsets. It completed cleanly but that was careless.
5. pkill -f 'b3_long_run\.sh' matched my own shell command line and killed my
   session mid-teardown. Same mistake pattern as the 2026-09-09 hf_upload pkill.
   Used explicit PIDs after that.
6. Advised that l1/l2/armA_c could be pruned if disk got tight, then found l2 is
   ext_32k.sh's PARENT. Retracted.

## Open

1. Decay check: STEPS=1100 STABLE=900 from the step-1000 ckpt, to size how much
   of the 4096 deficit is just the missing decay tail. Not started.
2. ext_32k.sh (parent l2, b3 data) - written, never run. Distinct from the
   ext32k_b4 run that did complete.
3. Resume l3long from step 2450 to 4000 if the trend is worth confirming;
   sidecars are on HF so optimizer state survives.
4. Against peer context from the same day (Qwen3-0.6B-Base flat-maxes NIAH to
   32,736 where ours is at 43-48%), +8 pp from step count is movement inside a
   regime already far short of peers; the identified bottleneck is the base
   pretrain, not the extension.
