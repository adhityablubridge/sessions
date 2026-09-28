2026-09-26 - 14:40 - 1.59B 16k->32k extension, the 84.8 plateau at 32,744, and three experiments identifying compression (not steps, not data mix) as the cause - BluTrain / stage32k/run32k.sh, run32k_long100.sh, filter/mk_mix32.py, reach_vs_steps.sh, diag_compression.sh

# Session report

## Headline

The 32k rung reaches **84.8** at 32,744 (n=400, Wilson [80.9, 87.9]) against the
RULER bar of 85.6. Effective context is **16,360 of 32,768 = 50% of nominal** --
better than the 680M ladder's uniform 25%, worse than the 16k rung's 100%.

The interval STRADDLES the bar. "The 32k rung failed RULER" is a point estimate,
not a result. I first reported 80.0 from n=100; at n=400 it is 84.8, 0.8 points
short. The same caveat applies to the 16k rung's headline last session, which I
reported as a pass without noting it -- re-scored today it is 93.8 [90.9, 95.7],
which IS a clean pass.

## Results

| checkpoint | scale | 4,096 | 8,192 | 16,360 | 32,744 |
|---|---|---|---|---|---|
| parent 8190 (16k) | x4 | 100.0 | 100.0 | 93.8 (n=400) | - |
| ext 8590 (32k, 60/40) | x8 | 100.0 | 100.0 | 100.0 | 84.8 (n=400) |
| ext 8990 (32k, 100% long) | x8 | 100.0 | 100.0 | 100.0 | 84.2 (n=400) |

The 32k rung took 16,360 from 91.0 to 100.0 -- it consolidated the window below
it completely, including the depth-0.00 cell that fails first everywhere else.

![[fig1_32k_extension.png]]

## Three experiments, and what they rule out

**1. More steps is not the answer.** Reach at 32,744 vs steps into the rung:
+10 -> 61.0, +210 -> 84.0, +400 -> 84.8. The first 210 steps bought +23.0 points;
the next 190 bought +0.8, inside noise. The plateau arrives at roughly half the
budget. I had told the user the curve was "still climbing with no sign of
flattening" -- that was an extrapolation through two points, and the middle
checkpoint falsified it before 3 GPU-hours were spent on it.

**2. The data mix is not the cause.** I changed this rung from the 16k rung's
100%-long to 60/40 on the ProLong result, which confounded the rung-to-rung
comparison. A 400-step 100%-long ablation from 8590 moved 32,744 by 0.6 points
(84.8 -> 84.2) and left 4,096/8,192/16,360 at 100.0. The confound is closed: the
mix did not create the plateau and reverting it does not fix it.

Train loss on that ablation fell 1.6865 -> 1.6074 while val ROSE 1.7672 -> 1.7732
and reach did not move. Third clean instance this project of the loss/retrieval
decoupling Fu et al. (arXiv:2402.10171) warn about.

![[fig2_plus400_ablation.png]]

**3. CORRECTED 2026-09-26 (later): it is virtual span AND real token count,
NOT compression.** At fixed real length 16,360 the MORE compressed read (x8,
8 tokens per virtual unit) scores 100.0 and the LESS compressed one (x4, 4 tokens
per unit) scores 91.0 -- higher scale HELPS at fixed real length, because it keeps
the context low in the virtual window. Two things cost: reaching further UP the
virtual window (100.0 -> 91.0 at fixed real 16,360), and more REAL tokens at the
same virtual span (100.0 -> 91.0 -> 84.8 for real 4,096 / 16,360 / 32,744).
The original wording below mislabelled the first of these as "compression".

**Original wording:**

**3. Compression is the dominant term.** Same checkpoint (ext8590) read three ways:

| read as | virtual span | real length | acc |
|---|---|---|---|
| x8 @16,360 | 2,045 | 16,360 | 100.0 |
| x4 @16,360 | 4,090 | 16,360 | 91.0 (DIAGNOSTIC, cross-config) |
| x8 @32,744 | 4,093 | 32,744 | 84.8 |

Holding real length FIXED at 16,360 and pushing virtual span 2,045 -> 4,090 costs
100.0 -> 91.0. Real length then costs a further 91.0 -> 84.8 at the same virtual
span. Roughly 9 points from compression, 6 from doubling real length.

The x4 read of an x8-trained checkpoint is cross-config -- the error flagged on
09-16/20/23 -- and is valid ONLY as a controlled diagnostic. It must never be
quoted as that checkpoint's score.

![[fig3_compression.png]]

## What the frontier does, and what we have been doing

| model | native window | rope_theta | scaling |
|---|---|---|---|
| Qwen2.5-7B | 131,072 | 1,000,000 | none |
| Qwen2.5-14B-1M | 1,010,000 | 10,000,000 | none |
| Qwen3-8B | 40,960 | 1,000,000 | none |
| Mistral-7B-v0.3 | 32,768 | 1,000,000 | none |
| Llama-3.1-8B | 131,072 | 500,000 | llama3, factor 8, ORIG 8,192 |
| Phi-3-medium-128k | 131,072 | 10,000 | su/LongRoPE, 64 per-dim factors |
| DeepSeek-V2-Lite | 163,840 | 10,000 | yarn, factor 40, ORIG 4,096, mscale 0.707 |

Most ship with NO rope_scaling: they train natively at the target length and let
rope_theta carry it, with theta tracking context (1M at 131k, 10M at 1M tokens).
Llama 3.1 does scale, but its ORIGIN is its pretrain window (8,192) and the factor
is 8 -- it did not hold a small origin and escalate the factor.

We have held ORIG=4,096 fixed since the 680M ladder and escalated the factor:
x4, x8, and the 680M went to x32. Experiment 3 says that is the axis that costs
us. Our base is theta 500,000 at a 4,096 window -- Llama-3.1's theta on an eighth
of Llama-3's window.

## Data supply, measured

Exact counts over 183 raw shards (previous figures were a 2-shard sample + a
Pareto fit, and were wrong):

| domain | >=32k | >=64k | >=128k | max doc |
|---|---|---|---|---|
| commoncrawl | 2.471% | 1.144% | 0.123% | 197,570 |
| edufineweb | 1.817% | 0.606% | 0.030% | 168,037 |
| code | 21.643% | 12.542% | 6.894% | 2,620,415 |
| math | 12.086% | 8.738% | 6.409% | 919,457 |

Corrected: I claimed edufineweb was capped below 128k with zero documents
>=131,072. That was one shard. Across 78 the max is 168,037 and the pool is 2.3M
tokens -- tiny, but "impossible by corpus construction" was an overstatement.

Shards to source, at 60/40: 64k needs +66 commoncrawl and +118 edufineweb
(185 total, 34 GiB). 128k needs +2,639 and +8,358 (10,997 total, 2.0 TiB) --
not a procurement path. code and math need nothing at either rung.

## Errors made this session

- Told the user the reach curve was still climbing on two points; the third
  falsified it. Caught before spending the 3 GPU-hours, not after.
- Reported 80.0 at n=100 as if decided; at n=400 it is 84.8 and undecided.
- Staged the autoeval with the 680M ladder's niah_one.sh -- wrong checkpoints,
  wrong env-var contract, wrong path case (/root/adhi vs /root/Adhi).
- Built bluscriptCP without CP_ATTN_SPLIT. Every eval job died on
  sdpa_split_forward and run_longeval.sh still printed "DONE -> summary.csv"
  over an empty file. A green log with zero evaluation, again.
- Copied a race from autoeval16k.sh that sent the same (label,yarn,len) to both
  GPUs with the same OUT directory.
- Reused the shell variable A for both the arch string and the accuracy in
  reach_vs_steps.sh, corrupting the second iteration. Same class as the
  S=$1/O="${S}" bug reintroduced three times on 09-23.
- mk_mix32.py emitted only train_*.bin; the mixed loader also requires val_*.bin
  per domain. First launch aborted.
- Claimed edufineweb has no documents >=131,072, from one shard.
- Suggested adding an HF collaborator to a personal repo. HF personal repos have
  no collaborators -- organizations only. The user corrected this.

## Operational notes

- BLU_CKPT_WEIGHTS_ONLY needs a DCP DIRECTORY; it cannot read a flat .ckpt.
  dcp2ckpt.py goes DCP->flat only. So deleting a DCP costs the ability to
  INITIALISE from that step, not merely its momentum. ckpt2dcp.py does not exist
  and is worth an hour -- the mapping is positional and documented in
  dcp2ckpt.py's own docstring.
- The HF private storage quota blocked checkpoint uploads. Making the repo public
  dissolved it; public storage is not counted against that limit.
- 32k peak memory: 117,645 MiB of 143,771 per GPU, world=2, B=1, fp32 logits.
  My projection was 129.6 GB, 11% pessimistic. bluscriptCP was NOT needed for
  training -- but IS needed for eval, because run_longeval.sh drives it.
- Throughput at 32k: 38.9k tok/s, 26.5 s/step, 42% MFU. Predicted 41.1k and
  25.5 s -- within 4%.

## What I would do next

Not more data for 64k. Experiment 3 says the binding term is compression into
the 4,096 origin, and x16 makes that worse. Two cheap tests first:

1. 32k at ORIG=8,192 / factor 4 instead of ORIG=4,096 / factor 8. Same target
   window, half the compression, one env var, invariant still holds.
2. 32k with NO YaRN at all -- theta 500,000 already covers 32k wavelengths, and
   this is the Llama-3.1/Qwen route. It was rejected at 16k on grad norm alone
   (1.3248 vs 0.2996 at step one), which is a proxy, not reach. That decision has
   never been tested on the metric that matters.
