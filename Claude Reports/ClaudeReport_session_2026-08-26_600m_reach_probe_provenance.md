# Claude Report - 600M reach probe: numbers + provenance (recovered)

2026-08-26 - Recovering the 600M step-500 reach/denominator probe from the
destroyed instance's vault, and pinning exactly which model it describes, because
it is NOT the base trained on 26 Aug - CP / probe_run4, needle_sweep.sh,
needle_score.py

The original report lived at /root/claude-vault on instance
ripe-face-embraces-fin-02, which has been torn down. These figures are recovered
from the session transcript. Recording them here so the vault holds them.

## The numbers

600M, run4, checkpoint at step 500 (524M tokens, 0.77 tok/param).
Probe: T=4096 NATIVE (no lambda/YaRN extension), CACHES=ntk, BLOCKS=3,
NEEDLE_LEN=16, HEAD_DIM=128 -- directly comparable to run100's 4k row.

| variant | distance | nll | significant |
|---|---|---|---|
| p0 | 4076 | 14.0787 | **no** |
| p0.25 | 3061 | 11.9778 | yes |
| p0.5 | 2046 | 9.8476 | yes |
| p0.75 | 1031 | 7.3560 | yes |
| p1 | 16 | 3.6960 | yes |

floor(p1) = 3.6960, absent = 14.4211, null = 14.4566, **denom = 10.7251 nats**

## Against the 120M

| | 120M (run100, final) | 600M (run4 @ step 500) |
|---|---|---|
| reach | 3045 | >=3061 |
| denom | 8.84 | **10.73** |
| val loss | 3.90 | **3.805** |
| tok/param | 6.4 | 0.77 |

**Reach did NOT improve with 6x the parameters** -- 3045 vs ~3061 is inside the
measurement's resolution. What improved is the DENOMINATOR (contrast between
needle-present and needle-absent), and val loss. Those are different quantities:
a bigger denom means the measurement has more headroom, not that the model
retrieves from further away.

This is the empirical basis for the project's "model size is not the dependency
for reach" line, and the reason IN2 is interesting: it points at SUPERVISION
rather than scale as the lever.

### How hard to lean on it

- **Unequal training.** 0.77 vs 6.4 tok/param. The comparison existed to establish
  that the base was SUFFICIENT for the IN2 test, not to benchmark reach fairly.
- **Coarse resolution.** Five probe points. ">=3061" means p0.25@3061 was
  significant and p0@4076 was not, so true reach is bracketed in [3061, 4076] and
  not pinned. Same for the 120M's 3045. The resolution cannot support a claim of a
  small difference in either direction.
- Both are 4k-native. Says nothing about behaviour after context extension.

## PROVENANCE -- what model this describes, and what it does not

The probe describes a 600M trained as:

| property | run4 (probed) |
|---|---|
| precision | fp32 |
| optimizer | AdamW |
| parallelism | DataParallel (DDP) |
| LR schedule | pure cosine (CP_STABLE did not exist yet) |
| data | edufineweb (15 shards, 1.5B tokens) |
| script | bluscriptCP |

It does NOT describe anything built on 25-26 Aug. Everything since differs on all
five axes: bf16, Muon, ZeRO-1 v2 + symmetric memory, WSD, and FLUX data, via
bluscript_zero.

**Consequence:** 3061 / 10.73 is not a valid baseline for the base completed on
26 Aug. That model needs its own probe before any long-context result is
interpretable.

**Blocker:** the probe harness runs the model through bluscriptCP's eval path, and
bluscriptCP is currently broken on the new box (pristine 2449e5c, fp32+DDP, gives
gradient norms 5e7-1.7e10 and a flat loss). So measuring the new base's reach is
blocked by the same defect that blocks the long-context test.

## Method note (for reproducing)

denom = nll_absent - nll_floor. needle_score.py withholds both score and verdict
below 0.5 nats, so a usable denom is a precondition for any reach claim. Reach =
the furthest distance at which retrieval is still significant. Cross-arm
comparisons must renormalise on a COMMON denominator -- not doing so is what
nearly produced a false "+8.8% reach" reading in the Ms-PoE work.
