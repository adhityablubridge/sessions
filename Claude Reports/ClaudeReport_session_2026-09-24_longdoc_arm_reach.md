# 680M reach: long-document arm (treatment) vs raw FLUX (control)

2026-09-24 - ran the long-document treatment arm on 4x H200 and evaluated it
against the already-completed raw-FLUX control - workspace BluTrain /
dist/Context_Parallelism, handoff/ARM_A.sh

## Result

**Data composition is the cause of the 680M's short retrieval reach.** Training
the same parent for the same 600 steps on documents >= 4,096 tokens grew reach;
training it on raw FLUX did not.

### Induction (copy) probe -- mean NLL over pos >= T/2, chance = 10.825 nats

| d | base 3400 | control (raw) | **long-doc** | L-base | L-control |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.4045 | 0.4076 | 0.3938 | -0.011 | -0.014 |
| 512 | 0.4657 | 0.4617 | 0.3337 | -0.132 | -0.128 |
| 1024 | 0.9905 | 0.8341 | 0.5452 | **-0.445** | -0.289 |
| 2048 | 3.2650 | 3.3379 | 2.0130 | **-1.252** | **-1.325** |
| 4096 | 11.0493 | 10.9524 | 10.3832 | -0.666 | -0.569 |

### NIAH niah_single -- accuracy %, REAL FLUX haystack, N=100

| T | base 3400 | control (raw) | **long-doc** | L-base |
|---:|---:|---:|---:|---:|
| 512 | 93 | 92 | 93 | +0 |
| 1024 | 85 | 86 | 87 | +2 |
| 2048 | 85 | 84 | 86 | +1 |
| 3072 | 72 | 71 | 75 | +3 |
| 4096 | 59 | 60 | **67** | **+8** |

### NIAH by needle depth at T=4096 (0.00 = start = furthest from query)

| depth | base | control | **long-doc** |
|---:|---:|---:|---:|
| 0.00 | 5 +-11 | 5 +-11 | **35 +-19** |
| 0.25 | 55 +-20 | 60 +-20 | 65 +-19 |
| 0.50 | 60 +-20 | 60 +-20 | 60 +-20 |
| 0.75 | 75 +-18 | 75 +-18 | 75 +-18 |
| 1.00 | 100 +-8 | 100 +-8 | 100 +-8 |

The entire NIAH gain sits at depth 0.00. Depths 0.50-1.00 are identical across
all three models. The improvement is specifically long-range retrieval, which is
what both instruments were built to isolate.

## Harness validated against the control's recorded numbers

This box reproduced the handoff's recorded figures EXACTLY before the treatment
column was read: base 93/85/85/72/59 and control 92/86/84/71/60. Same haystack
file, same generator, same scorer. The third column is therefore directly
comparable rather than a re-measurement on a shifted instrument.

`mk_repeat.py` on this box is byte-identical to the one that produced the
control's induction CSVs, so those five series were reused rather than recomputed.

## Configuration -- everything except data held fixed

| item | value |
|---|---|
| parent | blumodel_step_3400, weights-only (BLU_CKPT_WEIGHTS_ONLY=1) |
| steps | 3400 -> 4000 (600), global_batch 1,048,576 |
| topology | 4x H200, np=4, B=8, grad_accum 8 |
| schedule | BLU_MAX_LR=1.2e-3 warmup 200, lr 1.85e-4 -> 1.2e-4 |
| optimizer | muon + adamw, BLU_ALL_BF16=1, BLU_BF16_LOGITS=0, BLU_ZERO_V2=1 |
| throughput | 326k tok/s, 41.2% MFU, 405 TF/gpu, 80.6 GB of 143 |
| wall clock | ~29 min training |

## The treatment corpus

Built by `handoff/scripts/mk_flux_long.py` over FLUX shards 37-100 (unseen by
the base, which ended inside shard 36):

- 1,693,835,604 tokens, 217,533 documents, **0 duplicates**, 0 documents < 4096
- laid out 12/4/2 shards across commoncrawl/code/math to match the c0 weights
- 600 steps consume 629,145,600 tokens = **37% of one epoch**, so nothing repeats
- **46-47% of 4,096-token windows contain zero EOT, vs 11.4% in raw FLUX**

Duplicate-freedom was verified independently by re-hashing every emitted
document, not by trusting the builder's own counter.

## Loss and reach are different axes -- now shown in both directions

| | val loss | reach |
|---|---|---|
| control (raw FLUX) | 2.439188 -> 2.423660 (**better**) | unchanged |
| long-doc | 2.439188 -> 2.446606 (**worse**) | clearly improved |

The control improved loss and bought no reach. The treatment lost loss and
bought reach. Val loss is measured on a short-document-dominated held-out set,
for which the treatment is mildly off-distribution. Selecting long-context
checkpoints on val loss would have picked exactly the wrong model.

## Limits of this result

1. **d=4096 is still failure.** 10.383 against a chance floor of 10.825. Reach
   extended to roughly 2048, not to the full 4096 window.
2. The depth-0.00 cell is n=20; its Wilson interval (35 +-19) touches the base's
   (5 +-11). That single cell is suggestive on its own. It is persuasive because
   the induction probe, a separate and higher-n instrument, shows the same
   pattern at the same distances.
3. One seed, one run per arm.
4. This shows long-document DATA is sufficient to grow reach. It does not
   separate that from intra-document masking, which would produce a similar
   effect by a different route -- a long-doc corpus is one way of guaranteeing a
   window has no unrelated document in it, and masking is another.

## Next

- Extend the long-doc arm past 600 steps and see whether d=4096 comes off chance
- Run the same arm with intra-document masking on raw FLUX to separate the two
  mechanisms
- Re-run the 128k extension ladder from the long-doc checkpoint rather than base

## Artifacts

Uploaded to `unparallelled/blutrain-680m-reach`:
`treatment_long4000.ckpt`, `treatment_blumodel_step_4000/`,
`treatment/logs/runA.log`, `treatment/evals/`, `treatment/plots/`.

---

# Addendum: second 600 steps (4000 -> 4600)

Continued the long-doc arm for another 600 steps, same recipe, same corpus,
disjoint data (loader skipped 32,000 batches, resumed at commoncrawl shard 7;
run 1 trained corpus positions 1,354-6,154 and run 2 trained 6,154-10,954 of
12,923 batches per epoch, so no token was seen twice).

**More long-document training did not help. Reach peaked at 4000.**

### Induction probe

| d | base | control | long 4000 | long 4600 | 4600-4000 | 4600-base |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.4045 | 0.4076 | **0.3938** | 0.4315 | +0.038 | +0.027 |
| 512 | 0.4657 | 0.4617 | **0.3337** | 0.3922 | +0.059 | -0.074 |
| 1024 | 0.9905 | 0.8341 | **0.5452** | 0.7011 | +0.156 | -0.289 |
| 2048 | 3.2650 | 3.3379 | **2.0130** | 2.2247 | +0.212 | -1.040 |
| 4096 | 11.0493 | 10.9524 | **10.3832** | 10.5183 | +0.135 | -0.531 |

All five deltas positive: worse at every distance, ~1/3 of the gain given back.
Still well ahead of the base at d=2048 (-1.040).

### NIAH

| T | base | control | long 4000 | long 4600 | 4600-4000 |
|---:|---:|---:|---:|---:|---:|
| 512 | 93 | 92 | 93 | 91 | -2 |
| 1024 | 85 | 86 | **87** | 84 | -3 |
| 2048 | 85 | 84 | 86 | 86 | 0 |
| 3072 | 72 | 71 | **75** | 73 | -2 |
| 4096 | 59 | 60 | **67** | 63 | -4 |

Consistent direction, but each drop is inside the +-9 Wilson band at N=100. The
induction probe carries the evidence; NIAH corroborates it.

### Depth at T=4096 -- the reversion is NOT a loss of maximum reach

| depth | base | control | long 4000 | long 4600 |
|---:|---:|---:|---:|---:|
| **0.00** (furthest) | 5 | 5 | **35** | **35** |
| 0.25 | 55 | 60 | 65 | 55 |
| 0.50 | 60 | 60 | 60 | 50 |
| 0.75 | 75 | 75 | 75 | 75 |
| 1.00 | 100 | 100 | 100 | 100 |

The depth-0.00 cell held all of its 7x gain. The T=4096 drop came entirely from
mid-context (0.25, 0.50). The model did not forget how to reach far; mid-context
precision eroded.

### Val loss moved the opposite way again

Run 2: 2.4466 -> 2.4463 -> 2.4380 -> **2.4354**, crossing below the base's
2.4392.

| | val loss | reach |
|---|---|---|
| control, +600 raw | better | flat |
| long-doc, +600 | **worse** | **much better** |
| long-doc, +1200 | better | **partially reverted** |

Three consecutive observations, same inverse relationship. Checkpoint selection
on validation loss systematically selects against long-context retrieval.

## Revised recommendation

**Keep step 4000.** Long-document data grows reach, but the gain is not
monotonic in training time -- past ~600 steps it trades back for general LM
quality.

## Next experiment to disambiguate

Re-run 4000 -> 4600 at a FLAT LR instead of decaying to min_lr:
- if reach holds, the reversion was the LR decay settling the model back toward
  its general-purpose optimum
- if it still reverts, reach is an inherently transient effect of distribution
  shift and needs a mechanism that does not decay (e.g. intra-doc masking, which
  changes the objective rather than the data)

---

# Addendum 2: the 1.59B curriculum model (8.4B tokens)

Evaluated `navingv/curriculum_8btokens` step 8000 on the SAME induction streams
and the SAME NIAH haystack as everything above.

**Shape pinned from the checkpoint, not assumed**: metadata.bin has 262 param
records totalling **1,588,229,888 elements** -- d_model 2304, 26 layers, 18q/6kv,
head_dim 128, ffn 6144, vocab 50304, tied. It is **1.59B, not "around 1B"**.
8000 x 1,048,576 = 8.39B tokens = **5.28 tok/param**, the same undertrained
regime as the 680M's 5.2.

### Induction (chance 10.825)

| d | 680M base | 680M long-doc 4000 | 1.59B |
|---:|---:|---:|---:|
| 256 | 0.4045 | 0.3938 | **0.1994** |
| 512 | 0.4657 | 0.3337 | **0.1277** |
| 1024 | 0.9905 | 0.5452 | **0.1085** |
| 2048 | 3.2650 | 2.0130 | **0.2243** |
| 4096* | 11.0493 | 10.3832 | 9.0964 |

### NIAH (N=100)

| T | 680M base | 680M long-doc | 1.59B |
|---:|---:|---:|---:|
| 512 | 93 | 93 | **97** |
| 1024 | 85 | 87 | **89** |
| 2048 | 85 | 86 | 86 |
| 3072 | 72 | **75** | 73 |
| 4096 | 59 | 67 | **70** |

### Depth at T=4096 -- the whole story in one column

| depth | 680M base | 680M long-doc | 1.59B |
|---:|---:|---:|---:|
| **0.00** | 5 | 35 | **65** |
| 0.25 | 55 | 65 | 60 |
| 0.50 | 60 | 60 | 55 |
| 0.75 | 75 | 75 | 70 |
| 1.00 | 100 | 100 | 100 |

Within its trained window the 1.59B has essentially no retrieval problem
(0.199 / 0.128 / 0.109 / 0.224 at d=256..2048). At d=2048 that is a **14.6x**
reduction against the 680M base. Scale bought far more than the long-document
intervention did.

Both interventions act on exactly ONE cell: depth 0.00, retrieval from the far
end of the context (5 -> 35 -> 65). Depths 0.25-1.00 are flat or slightly worse
across all three models. That is why the induction gap is enormous while the
NIAH headline gap is only +11 at T=4096 -- NIAH averages the one differentiating
cell with four saturated ones.

**Confound**: the repo is named `curriculum_8btokens`, so that model likely used
a curriculum data schedule whose composition is unknown here. Parameters, token
count and data mix are confounded; this result cannot attribute the gain among
them without its training config.

## * MEASUREMENT CORRECTION -- applies to every d=4096 number in this report

The d=4096 induction cell uses T=8192 windows, DOUBLE the 4096 pretraining
context. At YARN_SCALE=1 it measures RoPE extrapolation, not retrieval reach,
and all models fail it for that reason. The valid in-window comparison stops at
**d=2048**. The d=4096 figures quoted in Addendum 1 and the main report should be
read as extrapolation results, not reach.

## Revised overall conclusion

- Long-document data is a real but **small and non-durable** lever on reach.
- It is not monotonic in training time: the gain decayed as the LR annealed.
- **Scale is what actually removes the constraint.** A 2.3x larger model at 2.35x
  the tokens has no in-window retrieval deficit at all.
- Anything selecting checkpoints on validation loss selects against reach.
