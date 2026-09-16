# 2026-09-15 / 2026-09-16 - 680M tied-head bf16 fix, phase-1 context extension (4k -> 16k), and the baseline correction

Workspace: Context_Parallelism (GPU box, now torn down) / CP (laptop)
Files: Scripts/Blutrain/bluscriptCP.cpp, p1_680m.sh, p2_680m.sh, p1_shipper.sh,
p1_finish.sh, p1_eval2.sh, p1_plots.sh, mk_p1_plots.py, dl_p23.sh, mix_buckets.py,
run_longeval.sh, HANDOFF_H200_32K.md

## Why this session existed

Two long-context extension runs on the 1.06B collapsed, both at roughly 80 percent
of a cumulative b5 epoch, tracking cumulative long-document exposure rather than
data composition. Working hypothesis: the 1.06B is undertrained at 0.99 tok/param,
so copying is a cheaper solution than retrieval. That predicts a better-pretrained
base has a higher ceiling. varybatch is a 680M at 10.5 tok/param - the test.

## The bug that had to be fixed first

varybatch scored NLL 12.1295, worse than uniform. Bisected with CP_DUMP_ACTS
against a PyTorch reference. GPT::forward's weight_tying path cast the TRANSPOSED
embedding matrix to bf16; the cast ignores strides, so the tied head multiplied a
garbled weight.

    before: transpose -> to_compute(bf16)          [strided cast, silently wrong]
    after:  to_compute(bf16) -> transpose -> contiguous()

Confirmed by mask bisection: mask=15 gave 12.1295, mask=7 (head in fp32) gave
3.0496. After the fix the 680M scores 3.0497 (PyTorch reference 3.0793) and the
1.06B is unchanged at 3.1719 (was 3.1717). Never surfaced before because every
model trained in this repo uses weight_tying=0 and takes the separate contiguous
lm_head path. bluscript_zero_ckpt.cpp documents the same hazard in a comment.

## Phase 1: 4k -> 16k

2x H100, cp=2, parent varybatch step 3400.

    window     16,384 via YaRN SCALE=4 x ORIG_MAXPOS=4096
    data       p1mix, 640,009,970 tokens, b3 50 / b2 30 / b1 20
    steps      1,200 of 1,220 usable at global_batch 524,288 (grad_accum 32)
    measured   50,117 MB of 81,559 per card, 7.83 s/step, clean finish
    loss       2.5969 @step 99 -> 2.6006 @step 1199   (i.e. flat)
    val        2.8726 -> 2.8560 over the whole run

rope_theta 500000 and context_length 4096 were verified against
bluscript_zero_ckpt.cpp:121 and .mdfiles/config.md - the trainer's
"provenance UNVERIFIED" warning only means no sidecar exists to compare against.

## Result: a null

niah_single, N=100, greedy, haystack b2_8192_16384/val_shard_00000.bin.

    prompt len   base680 (untrained, zero-shot YaRN)   p1final (1200 steps)
    8,192        84                                    77
    16,360       40                                    42

Both deltas sit inside the +-9-13pp Wilson band at N=100. 1,200 steps and 2.6 GPU
hours changed nothing measurable. The flat loss predicted this.

Depth breakdown at 16,360 (both models): 0 percent when the needle is at the start,
100 percent when it is at the end, monotonic in between. That is recency, not
retrieval, and training did not move it.

## Four explanations I gave, three of them wrong

1. LEARNING RATE - WRONG. I claimed phase 1 ran at 1.8e-4 against the 1.06B's
   3e-4. CP_MAX_LR and CP_MIN_LR appear ZERO times in bluscriptCP.cpp; max_lr is
   hardcoded 6e-4 at line 151. Every run in this project, both models, every
   context stage, trained at 6e-4 x REWARMUP_PEAK 0.3 = 1.8e-4. The scripts that
   set CP_MAX_LR=1e-3 were setting a variable the trainer never reads, and
   ext64k_b5.sh:23 already said so. THIS IS THE SAME ERROR I MADE AND LOGGED ON
   2026-09-09 ("my 3.0e-4 came from assuming max_lr=1e-3").

2. TOO FEW LONG TOKENS - WRONG. At the same 524,288 global batch, phase 1 saw
   314,572,800 b3 tokens; mix64k saw 188,743,680. Phase 1 had 1.67x MORE.

3. TRUNCATION - BACKWARDS. b3 documents are 16,384-32,768 tokens, at or above the
   16,384 window, so every phase-1 window is contiguous single-document text. That
   is the good case and the reason b3 was chosen. The 65,536-window runs are the
   ones that packed several b3 docs plus boundaries per window.

4. BASELINE - WRONG, AND MY CORRECTION WAS ALSO PARTLY WRONG. I reported 42 as
   "far below the 1.06B's 65-70", but ext64k had been through three extension
   stages. The stage analogue is the 1.06B's own 16k ladder: l1/l2/l3 tied at 35.0
   at 16,360, best l3long2k (run17 step 2000) 43.0. So 42 is at the 16k-stage best.
   But then the comparison table I built to show that was itself invalid, because
   eval directories used DIFFERENT haystacks.

## Haystack tiering (the reason cross-run eval comparison is unsafe here)

    val_shard_00000.bin   mix64k_1200, mixcont/cliff1400      <- same as phase 1
    b2val_haystack.bin    decide, ext32k, ext64k              <- probably same, UNVERIFIED
    b3val_haystack.bin    lenb
    flux_val_000000.bin   newck, longeval_yarnfix (N=50)

Proof this matters: checkpoint blumodelcp_run2_step_400 scores 78 under lenb and
62 under decide - same weights, 16pp apart, different haystack. The haystack files
are not on HF, so b2val_haystack.bin cannot be byte-checked against
val_shard_00000.bin despite the suggestive name.

Only mix64k (83 @8192 / 70 @16360) and mixcont (80 / 54) are certainly comparable
to phase 1.

## What actually stands

- The 680M base reaches 84 @8192 zero-shot, equal to the fully-laddered mix64k's
  83 on the SAME haystack. That is the one real signal for the higher-ceiling
  hypothesis.
- At 16,360 the 680M sits at 40-42, at the level of the 1.06B's 16k-stage ladder
  and well below its fully-laddered 65-70.
- 1,200 steps of training added nothing at either length.
- Therefore: the hypothesis is neither confirmed nor refuted. Phase 1 did not fail
  to reach a ceiling; it failed to move at all.

## YaRN rule correction (HANDOFF_H200_32K.md)

The doc said ORIG_MAXPOS must be "the length the weights were ACTUALLY trained at".
Right for a single extension, wrong for a ladder. build_rope_cache takes rope_theta
and is rebuilt from base theta every run, so SCALE never composes with the previous
stage. Re-basing freezes interpolation at theta/2 while the window doubles:

    stage  as-run (re-based)   interp   last tok -> virtual   ramp
    16k    4 x 4096            theta/4        4,096          [14,32]
    32k    2 x 16384           theta/2       16,384          [21,39]
    64k    2 x 32768           theta/2       32,768          [24,42]

    stage  corrected           interp   last tok -> virtual   ramp
    16k    4 x 4096            theta/4        4,096          [14,32]
    32k    8 x 4096            theta/8        4,096          [14,32]
    64k   16 x 4096            theta/16       4,096          [14,32]

find_dim(num_rot, L, head_dim, base) does not take scale, so the ramp moves only
when ORIG_MAXPOS moves. The 1.06B ladder followed the old rule from its 32k stage
on. This is NOT thought to be the collapse cause - a wrong cache is a static defect
while the collapse was progressive - but it plausibly lowered the ceiling.

## Operational errors this session

- The eval aborted in 49 s: run_longeval.sh defaults CP_DATA_ROOT to
  Data_Loader/Data, which on that box held only subdirectories, and the trainer
  builds a data loader even for pure generation ("no .bin shards found for split
  train"). Fixed with GEN_DATA_ROOT.
- up() verified logs/upload.log against a file it was itself appending to, so it
  always reported a mismatch. Now uploads a snapshot copy.
- A sed heredoc produced an EMPTY p2_680m.sh that still passed bash -n. test -s is
  now checked alongside it.
- run_longeval.sh prefixes OUT with $PWD, so an absolute OUT becomes
  ".../Context_Parallelism//tmp/...". Caught by an N=2 smoke test before it cost a
  second full cycle.
- I called the NLL position analysis a "copying fingerprint that explained why"
  retrieval dropped. It does not explain it: NLL shape and retrieval ANTI-correlate
  in that very data (the degraded model has the steeper context gain). The evidence
  for degradation is NIAH alone; the NLL work ruled out broken long-range
  processing and showed where the loss gain sat, nothing more.

## State

Everything byte-verified on HF unparallelled/BluScriptCP: full resumable phase-1
checkpoint both ranks (11.2 GB), both plots, all eval CSVs and generations, every
script, corrected handoff doc. p2mix and p3mix built and verified to read shards
disjoint from phase 1 and from each other. p2_680m.sh written and refuses to run
without a completed phase 1. Phase 3 needs >= 4 GPUs (~115 GB at cp=2).

## Open

- Why training moved nothing. Remaining candidate: next-token loss on natural text
  may not reward reaching backwards, so there is no gradient toward retrieval.
  Untested.
- Whether b2val_haystack.bin == val_shard_00000.bin, which would widen the
  comparable set to ext64k/ext32k/decide.
- CP_EVAL_PPL broken for weight_tying=1, deferred by decision.
