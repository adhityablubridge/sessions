# Session report - 2026-09-02 - Generative RULER/VAL/NIAH results, and two root causes for the reach ceiling

Workspace: /root/Adhi/BluTrain/dist/Context_Parallelism (box loud-life-embraces-fin-03)
Hardware: 4x H100 80GB
Files: Scripts/Blutrain/bluscriptCP.cpp, run_longeval.sh, run_len_sweep.sh,
       Tests/bluscriptcp/longeval_{gen,score}.py, compare_nll_vs_generative.py,
       longcontext_results.ipynb, EVAL_COMPARISON.md

## 1. It built and ran

The CP_GENERATE patch written yesterday had never been compiled. It compiled CLEAN on
the first attempt (bluscriptCP.o, 0 errors) with
`make CP_FUSED_ROPE=1 CP_ATTN_SPLIT=1 SM_ARCH=90 bluscript-cp`. Verified sm_90a image
present in GQA_causal_cp_fwd_sm90.o and the flag stamp correct before running anything.

Four rounds of review before execution found 14 harness defects, ALL of which would
have failed silently as plausible-looking low scores. The ones that mattered most:
  - PROMPT_LEN+MAX_NEW must equal 16384 exactly (pad_up(16368+24)=16640 > CP_T aborts
    every job)
  - DEPTHS was never forwarded to longeval_gen.py, so the depth grid would not have
    matched the NLL probe's p_frac grid
  - a signature mismatch wiped prompts but NOT $OUT/state/*, so stale .gen markers
    would have scored old tokens against a new meta.json
  - the summary parser split "arm_b_niah_single" on the first underscore -> name="arm"
  - early stop on the first newline would convert HITs to MISSes (the answer prefix
    ends "... is" with no newline, so a base model may emit "\n" first); needs a
    per-task minimum, 14 for niah_multivalue which must emit three 7-digit values
  - all N prompts shared byte-identical filler (hay loaded once, hay_pos reset to 0),
    making them replicates rather than independent draws

## 2. Results

3,600 prompts across 52 jobs. niah_single (recall a random 7-digit number given its
key; answer space 10^7):

  context   base   arm_a   arm_b   arm_c
    1,024   84.0    78.0    64.0    76.0
    2,048   82.0    66.0    56.0    70.0
    4,096   74.0    62.0    48.0    60.0
    8,192   40.0    36.0    36.0    40.0
   16,360    0.0     0.0     0.0     0.0

At 16,360: 2 hits in 2000 across all models and 5 tasks. Verified NOT a harness bug --
the depth-1.0 prompts are well formed, needle present at 99.8% depth (~30 tokens before
the question), no <|endoftext|> between needle and question, and the same harness HITs
at 1024.

Reach in tokens (50% hit-rate crossing, reach = (1-depth_cross)*context):

  context   base   arm_a   arm_b   arm_c
    1,024  1,024   1,024     938     938
    2,048  2,048   1,740   1,536   1,792
    4,096  3,481   2,730   1,706   2,560
    8,192  3,276   2,048   2,730   2,730
   16,360      0       0       0       0

REACH SATURATES AT ~2,700-3,500 TOKENS and does not grow with more context -- 4,096 to
8,192 slightly DECREASES base's reach (3,481 -> 3,276).

Forward vs backward: val_backward (name the key given its number, 35-word answer space)
is 0-8% at EVERY length while forward retrieval of a 7-digit number hits 84%. The model
resolves key->value but not value->key. That is the one VAL Probing axis implemented and
it fails outright.

## 3. NLL proxy vs generation

Pooled niah_single over 1024/2048/4096 (n=150/model, +-95% Wilson):

  base   80.0% +-6.4    NLL reach  6,047
  arm_a  68.7% +-7.3    NLL reach  7,435
  arm_c  68.7% +-7.3    NLL reach 12,976
  arm_b  56.0% +-7.8    NLL reach  5,924

  generative best->worst: base, arm_a = arm_c, arm_b
  NLL reach  best->worst: arm_c, arm_a, base, arm_b     -> DISAGREE

VERDICT: the NLL proxy is a weak DIRECTIONAL signal, not a magnitude and not a ranking.
  - overstates reach 2x (base) to 4.7x (arm_c)
  - ranks arm_c first; generation ranks the untrained base first
  - agrees only that arm_b is worst (true in every measurement we have)
  - IN2 advantage shrinks from 2.19x / 9.5 sigma to +12.7pp / 2.3 sigma (arm_c vs arm_b)
  - the 6.87-sigma "arm_c better at depth 0.25 than 0.5" anomaly comes out REVERSED
    generatively: -40pp at -3.8 sigma
  - BOTH metrics agree there is NO U-shape: depth curves rise monotonically toward the
    question, no primacy bump at depth 0. Pure recency.

## 4. Two root causes for the ceiling

CAUSE 1 -- a YaRN config bug, fixable with one env var and no retraining.
YARNOps.cpp:57 defaults YARN_ORIG_MAXPOS to 1024, but the base model (run8) trained at
context_length 4096. So every run of this project emitted:
    [YaRN][warn] seq_len(16384) != scale(4.000)*orig_maxpos(1024.0)=4096.0
The interpolation ramp targeted 4096 positions while the cache was SIZED 16384.
Setting YARN_ORIG_MAXPOS=4096 (ramp moves from low=7/high=25 to low=14/high=32):

  niah_single @16,360, N=100/cell:
    model   before   after
    base      0.0%   17.0% +-7.3
    arm_a     0.0%   17.0% +-7.3
    arm_b     0.0%    4.0% +-4.1
    arm_c     0.0%   18.0% +-7.5

  depth breakdown (after): d0.50 ~0-5%, d0.75 20-30%, d1.00 55-65%

0/100 -> 17/100 is p<0.001. BUT the recovery sits only at depths 0.75-1.0, so reach at
the 50% bar is STILL 0 -- it un-suppresses the last ~15% of the window, not the whole
thing. And it is a train/eval mismatch (checkpoints trained under the 1024 default), so
a model actually TRAINED with the correct value should do better. vt barely moves
(0-5%), so the effect is specific to single-needle forward retrieval.

CAUSE 2 -- the training data has no long documents. This is the real ceiling.
flux document lengths (split on <|endoftext|>, 20M tokens):
    median 826   p90 2,721   docs >=4k 4.50%   >=8k 1.00%   >=16k 0.27%
Measured reach saturates at 2,700-3,500. The p90 document length is 2,721. Those are the
same number. Each 16k training window was ~7 unrelated documents, so there was never a
gradient signal for attending past a document boundary. YaRN extends positional RANGE;
it cannot manufacture long-range dependencies that were never in the data. This is also
why the 750-step YaRN extension bought only +776 tokens on the NLL metric.

## 5. Errors of mine, recorded

  - PRE-REGISTERED THRESHOLDS WERE ILL-POSED. "refuted iff |diff| <= 5pp" has no power
    precondition, so at the 16k floor it is satisfied automatically and mechanically
    returns "refuted" when the honest answer is "untestable". And no sign handling, so
    prediction 3 coming out -40pp at -3.8 sigma (a significant REVERSAL) got mislabelled
    "inconclusive". Should have required at least one arm above ~10% before evaluating.
  - Reported the two sweep tasks BACKWARDS in one message (claimed val_backward was the
    64-76% task and niah_single the 2-4% one; it is the reverse), which inverted the
    answer-space caveat.
  - Timing estimate was 12x off in one direction then 5x off in the other: first ~7 s
    per forward from an fp32 H200 probe, then 0.23 s from a 2-equation/3-unknown fit
    that over-attributed 35.6 s to model load. Measured truth: ~1.3 s/forward.
  - Advocated against a mid-run restart for oversubscription, was pushed, recomputed
    +21 min, executed -- real gain ~1 min. GPU time-slicing without MPS gave 1.42x not
    1.8x, and 13 simultaneous 3.9 GB checkpoint loads cost ~6 min not 2.
  - The degenerate-task exclusion rule emptied the primary table when ALL tasks were
    degenerate; patched to fall back to pooling everything and label it a floor effect.

## 6. Deliverable

longcontext_results.ipynb -- executed, 6 figures embedded (695 KB). Reads the result
CSVs from disk; only the NLL reference values are hard-coded and they are labelled.
Followed the dataviz method: form first, categorical slots 1-4 in fixed order, no
dual-axis (NLL nats and generative % are separate panels), direct labels plus printed
tables (relief rule for the low-contrast slots). Could not run validate_palette.js --
no node runtime on the box -- so relied on palette.md's documented pass for the
unmodified reference palette on the adjacent pairlist.
Executing and LOOKING at it caught three defects I would otherwise have shipped: a log
y-axis rendering reach=0 as a spike off the plot, an x-tick labelling 16,360 as "15k",
and colliding end-labels in two figures (fixed with a collision-avoiding label placer
that must run AFTER set_ylim).

## 7. Not covered

RULER's CWE / FWE / QA / multiquery; VAL's code and structured-data contexts and
bi-directional retrieval; niah_multikey / niah_multivalue / vt in the length sweep.
And the bounding limitation: the two metrics never overlap at a context length where
both have signal -- NLL only at 16k (generation floored), generation only at 1-4k (NLL
never measured there, and measuring it would need a different YaRN cache).

## 8. Next

  1. Set YARN_ORIG_MAXPOS=4096 everywhere AND retrain with it.
  2. Source genuinely long documents, or pack related documents so cross-boundary
     attention earns reward. This is the only thing that raises the ceiling.
  3. IN2 properly resourced: far needles DID beat near needles (+12.7pp), but 400 steps
     at 3e-4 on random-token needles cost more general ability than it bought -- longer,
     lower LR, meaningful-text needles.
  4. Batch the decode loop (CP_B>1). Util was ~40% at B=1; worth ~4-8x, vs the 1.42x
     that GPU oversubscription bought.
