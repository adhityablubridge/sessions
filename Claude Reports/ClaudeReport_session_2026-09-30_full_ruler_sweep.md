# Full RULER sweep - 1.59B - 2026-09-30

Workspace: BluTrain / dist/Context_Parallelism
Files: stage_ruler/run_ruler.sh, Tests/bluscriptcp/longeval_score.py, results_2026-09-30/
Source of every number: the 24 scored CSVs at hf.co/unparallelled/blutrain-680m-reach
under ruler_2026-09-30/ (89 files, verified by re-download).

## Config

One config across all 24 cells. Only the checkpoint and the task vary.

    real length   32744
    YARN_SCALE    8          YARN_ORIG 4096      virtual window 4093
    theta         500000
    N             100 per cell,  MAX_NEW 24,  bf16
    hardware      1x H200, JOBS_PER_GPU=3, 117.6 of 143.8 GB in flight
    wall clock    11:33 -> 17:1x UTC

Checkpoints: base8600 (8.6B-token pretrain at 4096, NO long-context training),
nat9000 / nat9800 (+400 / +1200 steps at native positions, YARN_SCALE=1),
ext8590 (16k YaRN x4 -> 32k YaRN x8, highest short-document exposure).

## LINEAGE - base8600 and ext8590 are SIBLINGS, not parent and child

    base8000 --(600 steps, 4096, NO long context)--> base8600 --> nat9000 --> nat9800
    base8000 --(16k YaRN x4, 8000->8190)-----------> 8190 --(32k YaRN x8, 400 st)--> ext8590

The step numbers give it away: 8590 is BELOW 8600. ext8590 never passed through
base8600. So nat9000 and nat9800 are true before/after measurements against
base8600 (12 of the 18 comparisons); the six ext8590 cells are a sibling
comparison sharing only the base8000 ancestor, and ext8590 additionally carries
190 steps of the 16k rung.

Its cumulative long-context token count is NOT recorded in any data file on
hand - the 32k rung is 0.42B, the 16k rung's gbatch is unrecorded. Earlier in
this session it was plotted at 0.90B, which was invented, and that point has
been removed from the token axis.

## Results

    task              base8600  nat9000  nat9800  ext8590
    niah_single           99.0     99.0     95.0     80.0
    val_forward           99.0     99.0     95.0     80.0
    niah_multikey         83.0     74.0     76.0     45.0
    niah_multivalue       36.0     29.7     26.0     22.0
    val_backward          13.0      8.0      6.0      6.0
    vt                     1.0      4.0      3.0      0.0
    ---------------------------------------------------------
    RULER mean (partial)  55.2     52.3     50.2     38.8
    RULER mean (strict)   49.2     47.3     45.8     35.2

Wilson 95% CI at n=100 is about +/-9 points.

![[fig1_profile.png]]

## Findings

1. The untrained base is the best checkpoint in five of six columns, and best
   overall on both scoring conventions. Of the 18 trained-vs-base comparisons,
   14 move DOWN, 2 up, 2 equal. The two that move up are vt (1.0 -> 4.0 and
   1.0 -> 3.0), inside the noise band.

2. The RULER mean decreases monotonically along the native lineage with tokens
   of long-context training: 55.2 -> 52.3 -> 50.2 at 0 / 0.42 / 1.26B.
   ext8590 sits at 38.8, far below, but it is NOT on this axis - see LINEAGE.

![[fig2_decline.png]]

3. Two tasks are SATURATED at this config and carry no signal: niah_single and
   val_forward, both 99.0 on a model with zero long-context training. Every
   reach conclusion in this project to date was measured on niah_single alone.

4. niah_multikey is the only cell with headroom in both directions. Training
   spends it downward, 83.0 -> 45.0 at worst.

5. The base scores 99.0 on niah_single at x8 and 0.0 at native positions on the
   same weights. Inference-time YaRN alone accounts for a 99-point swing, which
   is what the compression finding was measuring.

![[fig3_instrument.png]]

## ALL 24 CELLS ARE READ AT x8 - what that does and does not mean

Confirmed from the gen logs, identical on every cell:

    [YaRN] build_rope_cache: seq_len=32768 hd=128 scale=8.000 low=14 high=32 m=1.2079

Real length 32,744 compressed into virtual 0-4,093. Train-vs-eval match differs
by checkpoint:

    base8600   trained 4096, no long context   -> x8   (n/a, never trained long)
    nat9000    trained native x1 to 32,768     -> x8   MISMATCH
    nat9800    trained native x1 to 32,768     -> x8   MISMATCH
    ext8590    trained YaRN x8 at 32,768       -> x8   MATCH

So these numbers are NOT a measure of what the native checkpoints were built to
do: x8 compresses their extended ruler back into the 4k region and never asks
for a position beyond 4,093.

Two checks show the mismatch does not explain the result:

1. x8 is the FAVOURABLE reading for the native checkpoints, not the penalty.
   nat9800 at 32,744, same weights, same real length:
       native positions (its own training config)   40.5%   [2026-09-29]
       x8 (this sweep)                              95.0%
   x8 is worth +54.5 points to it, and it still loses to base8600.

2. ext8590 is the ONLY checkpoint evaluated in its own training config, so it
   should be the advantaged one. It is the worst at 38.8.

What remains true: at x8 no checkpoint is asked to use a position beyond 4,093,
which is the window base8600 was pretrained on. The config is fair in RULER's
sense - one fixed config across the sweep - but it is home turf for the base and
it is not a test of extended reach for anyone. A sweep that tests reach would
read each checkpoint at scale=1 across a length ladder; for the native arm that
already exists (2026-09-28/29) and is much worse.

## Scoring convention - a result that changed on re-reading the CSVs

The scored CSVs carry both `ok` (all expected values retrieved) and
`n_hit`/`n_expected` (credit per value). They coincide on the five
single-answer tasks. On niah_multivalue, which asks for 4 values, `ok` reads
0.0 for EVERY checkpoint while the per-value ratio separates them cleanly:
36.0 / 29.7 / 26.0 / 22.0, the same monotone ordering as every other task.
longeval_score.py prints the strict number, so niah_multivalue was reported
mid-session as a dead task floored at 0.0. It is not dead; strict scoring
destroyed the signal. Both conventions are kept in data.py and both give the
same ranking.

## Corrections to numbers reported earlier this session

- base8600 val_backward was given as 5.6% mid-sweep, read off an incomplete
  generation file. Complete, it is 13.0%.
- niah_multivalue was reported as 0.0 for all four checkpoints. That is the
  strict reading only; per-value it is 36.0 / 29.7 / 26.0 / 22.0.
- The trained-vs-base tally was given as "ten of twelve move down". There are
  18 comparisons, not 12: 14 down, 2 up, 2 equal. Only 12 of those 18 are true
  before/after (the native lineage); the ext8590 six are sibling comparisons.
- ext8590 was plotted at 0.90B long-context tokens and described as scoring
  lower "at LESS training". That token figure was invented - the 16k rung's
  gbatch is not in any data file on hand - and the comparison also assumed a
  lineage ext8590 does not have. Both removed.

## Open, untested

ext8590 is the worst checkpoint and also the one with the most intra-document
boundary exposure (~25 boundaries per 32,768 window, full mutual attention, no
intra-doc masking in MixedDataloader.h). Degradation is monotone with exposure.
But ext8590 differs from the native runs in TWO ways at once - masking exposure
and YaRN - so this sweep does not separate them. The cheapest separating test
is ext8990, trained on --long-frac 1.0 (about 0.5 boundaries per window), at
this same config.

Never measured: the base's perplexity on 32,768-token windows. No step-0 val
exists in any log (val_freq=250).

## base8600 against peer BASE checkpoints

Peer numbers measured 2026-09-09 through OUR harness (peer_gen/peer_run/
peer_score): raw token ids as pure completions, no chat template, same
answer-prefix protocol our own base gets, one shared haystack TEXT tokenized by
each model's own tokenizer.

    task           Qwen3-0.6B  Qwen3-1.7B  Qwen3-4B  Qwen3-8B  Spark-4B  base8600
    niah_single         100.0       100.0     100.0     100.0     100.0      99.0
    val_backward         24.0        55.0      52.0      63.0      63.0      13.0

Peers at 32,736 of their own tokens; base8600 at 32,744.

![[fig4_peers.png]]

TWO CAVEATS, neither removable from this data:

1. The peers reach 32,736 at their NATIVE context - Qwen3 is trained to 32k+ and
   no YaRN was applied to any of them. base8600 reaches 32,744 only under YaRN x8
   compression into its 4,096 pretraining window. Same length, same tasks, same
   harness, but not the same demand on the model.
2. Peer coverage is TWO of our six tasks. There is no peer measurement for
   niah_multikey, niah_multivalue, vt or val_forward. None is invented here.

What the two tasks say:

- On niah_single we are effectively AT peer level: 99.0 against a flat 100.0
  across a 13x parameter range. But that is the task this sweep just showed is
  SATURATED - an untrained 4k model scores 99.0 on it - so matching peers here
  means very little.
- On val_backward we are FAR below every peer, including a 0.6B model that is
  smaller than ours: 13.0 against 24.0-63.0, where chance is ~2.9%. This was
  already identified as the discriminating task on 2026-09-09 and it still is.

The gap is therefore NOT closed by the base's 99.0 on niah_single. The two peer
tasks split exactly the way our own six tasks split: peer-level on the saturated
one, far behind on the one that requires work across the span.

A six-task peer comparison does not exist. Producing one means running the five
peers through this same suite at this same length, which needs a GPU.
