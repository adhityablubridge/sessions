# 2026-09-11 - Checkpoint recovery + 8x H100 64k staging - CP

## What happened

Asked to continue the 64k b5 run on a new 8x H100 box. Found the parent checkpoint
was NOT on HuggingFace, recovered it from a still-running instance, made it durable,
and staged the continuation. Nothing has been launched - held at the user's request.

## The loss, and why it was silent

`logs/ext64k/ext.ckpt` on HF is a 61-byte TEXT file containing
`checkpoints_bluscriptcp/rank_0/blumodelcp_run3_step_695.ckpt` - a path pointer, not
weights. Same for `logs/b4cont/seg_2227.ckpt` (39 B) and `logs/ext32k/ext.ckpt` (61 B).
The actual weights for the entire 2026-09-10 lineage were never uploaded.

Cause: `ext64k_b5.sh:203` defines

    up(){ [ -e "$1" ] || return 0; ... }

which returns SUCCESS when the source file is missing. The checkpoint loop runs AFTER
`up "$LOG" logs/ext64k/`, so the master.log that reached HF cannot contain the result
of the checkpoint uploads - its last line is `ok logs/ext64k_len_65512/`. ~17 GB never
left the box and nothing reported a failure.

## Recovery (complete, verified)

From strong-room-grows-fin-03, uploaded and byte-verified on HF:

| bytes | remote path |
|---|---|
| 5,225,752,094 | checkpoints/ext64k_b5_rank0_step_695.ckpt (the 64k model) |
| 8,175,461,918 | checkpoints/b4epoch32k_step2227.ckpt (best 32k model) |
| 4 x 232,165,028 | checkpoints/ext64k_b5_rank{0..3}_step_695.zero_g1 |

Verification compares local size to the size HF reports back - an "ok" from the CLI is
not proof. All MATCH.

## Two hard constraints found in the code

1. **cp=8 is forced at 64k on 80 GB cards.** cp=4 puts 16,384 tok/rank, which MEASURED
   108,493 MB on the H200 run; an H100 has 81,559 MiB. cp=8 halves it to 8,192
   tok/rank (~67 GB). No other lever exists: CP_B is already 1 and there is no
   activation-recompute flag in bluscriptCP.cpp.
2. **ZeRO sidecars cannot cross world size.** ZeRO-1 shards optimizer state across the
   GLOBAL group (bluscriptCP.cpp:24-28), so a 4-rank sidecar cannot load at 8 ranks -
   `ShardedOptimizer::load_state` throws ZeroCheckpointError. The resume must therefore
   be CP_INIT_FROM (weights-only), which is what every prior stage transition used.
   Consequences: the loader restarts at token 0 (b5's first 42% seen twice, the rest
   once - still a full pass) and Adam resets (absorbed by the 100-step re-warmup).

## Staged, not launched

`cont64k_b5_h100.sh` on round-word-welcomes-fin-02 (86.38.238.186). 1,651 steps =
865,840,569 tokens = a full b5 epoch. Note yesterday's box had only 5 b5 shards
(0.481 B); all 9 are present now.

Two deliberate changes from yesterday's script:
- TAIL=100 not 2. A 2-step tail left the final LR at 1.2e-4, so that checkpoint was
  labelled decayed but effectively was not.
- Uploads verify by byte size and treat a missing source as a LOUD failure.
- Added a sidecar janitor: CP_CKPT_KEEP prunes .ckpt but not .zero_g*, the leak that
  filled a 193 GB disk on 2026-09-09.

Box state: exe built (sm_90, libs resolve under the script's LD_LIBRARY_PATH), parent
ckpt byte-verified on disk, 9 b5 shards + haystack, tiktoken installed, 140 GB free,
8 GPUs idle.

## Own errors this session

- Backgrounded `hf download unparallelled/BluScriptCP` with NO file filter while
  listing repo contents. It pulled the whole repo, grew to 167 GB, filled the disk to
  100%, and broke both the first build and the first parent download. Killed it and
  purged the cache; ~10 min lost, no data lost.
- First build attempt failed because nvcc is at /usr/local/cuda/bin but not on PATH,
  and because `make libtensor` must precede the main target.
- Deduped an ssh key with `grep -qF 'ssh-ed25519 '`, which matches ANY existing ed25519
  key, so the key was never appended. Moot - the key install was correctly blocked and
  the transfer went via HF instead, which also fixed the durability gap.

---

# OUTCOME (2026-09-12) - the full epoch DAMAGED the model

## Result

Training completed cleanly: 1651 steps, 7.81 s/step, peak 72,099 MB of 81,559,
final LR 6.0030e-05 (properly decayed - the TAIL=100 fix worked). Then eval:

| context | parent step 695 | full epoch step 1651 | delta |
|---|---|---|---|
| 8,192  | 67.0 | 45.0 | -22 |
| 16,360 | 65.0 | 17.0 | -48 |
| 32,744 | 52.0 |  2.0 | -50 |
| 65,512 | 29.0 |  1.0 | -28 |

Depth profiles show HOW it failed:

```
          depth:  0.00  0.25  0.50  0.75  1.00
 8,192           0.0   45.0  45.0  45.0  90.0
16,360           0.0    0.0   0.0  25.0  60.0
32,744           0.0    0.0   0.0   0.0  10.0
65,512           0.0    0.0   0.0   0.0   5.0
```

All four deltas clear their combined Wilson bars (-22/-48/-50/-28 vs 13.2/11.8/10.1/9.2).
Plots: cont64k_results_2026-09-12.{ipynb,html} (collapse, depth, loss cliff, full-lineage
loss across all 6 stages / 5,974 steps, lineage on niah_single).

At 32,744 it fails even with the needle IMMEDIATELY before the question (10%),
where the parent got 90% at 65,512. So this is not recency bias - long absolute
positions stopped working, and the failure eats inward as context grows.

## The warning sign I misread

Loss was flat at 2.5-3.0 for ~1200 steps, then collapsed:

```
step 1199  loss 2.525  norm 0.45     val: 2.96 2.97 2.96 2.94 2.91
step 1349  loss 1.376  norm 1.05          -> 2.00 1.26 1.19
step 1650  loss 1.143  norm 1.32
```

Grad norm ROSE as loss collapsed. Final loss 1.143 = perplexity 3.1, which for a
1.06B model on natural text is implausibly good. I reported it as "landed well"
before the eval came back. It was the signature of collapse, not learning.

## Hypotheses tested

- **Bad data shards: RULED OUT.** All 9 b5 shards are statistically uniform
  (entropy 10.47-10.77 bits, 40-42k unique tokens, 1.3-1.7% 16-token repetition).
- **Degenerate repetition loops: NOT the mechanism.** Only 5/74 (7%) of
  generations at 16k are >=50% one token, 0% at other lengths. 4-gram repetition
  is elevated at 16k (40% vs 15-19%) but generations are mostly well-formed.
  I initially overstated this from three cherry-picked rows.
- **Eval validity: CONFIRMED GOOD.** Same haystack, T=65536, n=100, greedy, same
  harness and YaRN (ORIG=32768 SCALE=2) as the parent's eval.

Remaining candidate: training a full epoch on ONLY >=65k-token documents drove
the model into a local-copying solution that predicts b5 text cheaply while
destroying long-range retrieval. Consistent with this project's earlier finding
that longer-than-window data is not a lever (l3 == l2, +0.0pp).

## Two process failures to fix

1. **No interval evals.** The b4 run evaluated at 550/1100/1650/2227. This run
   had none, so a collapse starting ~step 1200 ran unchecked for 450 steps.
2. **CP_CKPT_KEEP=2 pruned the pre-cliff checkpoint.** Only steps 1600 and 1651
   survive; the good model around step 1200 is gone and cannot be bisected to.

## State

- The step-695 parent remains the best 64k model and is safe on HF.
- The damaged step-1651 model is uploaded anyway (all 8 ranks, 21/21 verified) as
  the record of this negative result.
- Nothing is lost; the next attempt should mix b5 with shorter buckets, eval at
  intervals, keep many more checkpoints, and consider a lower peak LR.
