# ClaudeReport_session_2026-08-12_context_rot_probe

2026-08-12 - Built a retrieval (context-rot) probe for long-context evaluation and ran the first gates
and reach curve - Workspace: CP - Files: Tests/bluscriptcp/needle_gen.py, needle_score.py,
needle_sweep.sh, Scripts/Blutrain/bluscriptCP.cpp

---

## Why

Long-context quality was being judged by **PPL-vs-position only**, which is blind to the failure the
question is actually about. The 114M shows a *flat, improving* held-out PPL curve at 128k
(`ppl_prog_s32.csv`, ratio 0.87) while nothing in the harness could say whether it can retrieve a fact
planted at position 20k. PPL is dominated by local next-token statistics, so a model can score well while
using only a bounded recency window.

## Key discovery: no C++ changes were required for the probe

`CP_EVAL_TOKENS_BIN` (`bluscriptCP.cpp:1519`) already windows a raw int32 token file, and `CP_EVAL_PPL`
already emits **per-position** `mean_nll`. So synthesising trials with the answer span at a known offset
yields retrieval NLL straight out of the existing, already-verified CSV path. The probe is pure Python.

## What was built

| file | role |
|---|---|
| `Tests/bluscriptcp/needle_gen.py` | fixed-slot induction/copy trials -> int32 LE `.bin` + JSON manifest |
| `Tests/bluscriptcp/needle_score.py` | span NLL -> raw-NLL significance vs a per-cache null, guarded score |
| `Tests/bluscriptcp/needle_sweep.sh` | group-slot fan-out (`slots = NGPU/CP_SIZE`), markers, resumable |
| `Scripts/Blutrain/bluscriptCP.cpp` | `CP_ROPE_THETA` env knob + per-run `.meta` sidecar + precedence/die |

Design points that matter:

- **Fixed-slot layout.** `S` needle-sized slots at fixed offsets; the needle occupies exactly one (or
  none). The probe prefix and answer span therefore sit at **byte-identical absolute offsets in every
  variant**, so NLLs are always differenced at the same point on a position-varying curve.
- **7 variants per cell**, not 15: `p_frac in {0,.25,.5,.75,1}` + `absent` + `null`. `p_frac=1.0` *is* the
  retrieval floor, so conditions and positions are one axis.
- **Off-by-one.** The eval reports NLL at the INPUT position, so the answer token at trial offset `a` is
  scored on CSV row `a-1`. The manifest carries `csv_positions`; the scorer never uses `answer_offsets`.
- **Stdlib only** (no numpy) after finding the pyenv `python3` lacks it while `/usr/bin/python3` has it -
  that mismatch would have broken the sweep on one box.

## Two flaws found by testing my own code (not at runtime)

1. **Scorer called 0.01-nat differences "significant".** With low block-to-block scatter, s.e. shrinks
   until a trivial difference clears `2*s.e.` Added a pre-registered `MIN_EFFECT_NATS = 0.1` materiality
   floor, and made a failed A4.1 gate withhold **both** score and verdict rather than emit a
   significant-looking result on a degenerate denominator.
2. **The θ sidecar was being written from forward-only paths.** `checkpointing` is on during
   `CP_EVAL_PPL`, so evaluating a *legacy* checkpoint would have stamped the assumed default into the
   sidecar as authoritative - laundering an assumption into a fact that later runs trust, and which would
   then `die` against a correct env value. Sidecar writes are now gated to training paths only.

## Verification results

| check | result |
|---|---|
| Fixed-slot invariants (answer span invariant; `p_X` differs from `absent` only at its own slot; `p1` adjacent; blocks differ) | PASS |
| Scorer logic on 3 fabricated scenarios (works / total rot / no dynamic range) | PASS |
| A4.1 dynamic range | PASS on the 114M (denom ~9 nats at every T). On the 48M it DEGRADES with length and FAILS at T=32,768 (0.21 < 0.5) - see Finding 1 |
| A4.2 in-distribution retrieval | **as worded (`p_frac=0`), FAILS on both** models at blocks=3. Under the corrected wording ("some `p_frac` > 0 is significant") both PASS. The earlier "PASS on 114M / FAIL on 48M" was a blocks=1 artifact |
| A4.3 CP invariance (CP=1 vs CP=2 identical CSVs) | PASS |
| Verification #2: `CP_ROPE_THETA` unset -> byte-identical CSV | PASS |
| Verification #3: θ precedence, 4 cases incl. die-on-mismatch + legacy warn | PASS |

A4.1 deliberately gates only `floor < absent`. A distant `p_frac` sitting at the `absent` level is **total
context rot - the finding**, and must not trip a sanity gate; the scenario test confirms it scores ~0.01
and passes.

## Findings

### 1. The capacity axis is real: the 48M reaches 2-3x less far, and its probe range COLLAPSES with length

This finding went through two wrong versions, both caused by **blocks=1 runs having no error bar**:

- v1 (wrong): "the 48M cannot retrieve at distance -> drops out of the capacity axis." Over-read a single
  failing position (`p0`, the maximally distant slot) as no retrieval.
- v2 (wrong): "the 48M reaches as far as the 114M, only more weakly." Used the 48M's **blocks=1** reach
  (3,045). At blocks=3 the same quantity is **1,015** - the missing spread had inflated it 3x.

Final, matched-stimulus comparison (identical `.bin` files, seeds, needles and λ; NTK; **blocks=3**):

| T | 48M `run200` reach | 114M `run100` reach | ratio | 48M denom | 114M denom |
|---|---|---|---|---|---|
| 4,096 | 1,015 | 3,045 | **3.0x** | 2.86 | 8.78 |
| 8,192 | 2,039 | 4,078 | **2.0x** | 2.05 | 9.28 |
| 16,384 | 4,087 | 8,174 | **2.0x** | **0.88** | 8.88 |
| 32,768 | **UNUSABLE (denom 0.21 < 0.5 floor)** | 8,183 | - | **0.21** | 9.35 |

Two signatures, the second sharper than the first:

1. **Reach**: the 48M is consistently 2-3x shorter at matched length. Both models scale reach
   proportionally with T through 16k (48M: 1,015 -> 2,039 -> 4,087, exactly x2 per doubling), so neither
   had saturated by 16k; the 114M's stall appears only at 32k.
2. **Probe dynamic range collapses for the 48M, to the point of total failure**: denominator
   2.86 -> 2.05 -> 0.88 -> **0.21 nats**, crossing the 0.5-nat A4.1 floor at T=32,768, versus a flat
   ~8.8-9.3 for the 114M at every length. At 32k the scorer correctly marked the cell **UNUSABLE** and
   withheld score and verdict. This is not "reach shrank": at 32k the 48M cannot distinguish a needle
   sitting IMMEDIATELY BEFORE the probe from no needle at all - it has stopped using the planted
   information at any distance. The collapse accelerates (ratio 0.72, 0.43, 0.24 per doubling), so
   extrapolating it would have gone unusable near ~24k.

**Rot thresholds measured:**

| model | usable through | dead at | threshold |
|---|---|---|---|
| 48M `run200` | 16,384 (4x, reach 4,087) | 32,768 (8x) | **between 4x and 8x** |
| 114M `run100` | 32,768 (8x, reach 8,183) | not reached | **above 8x** (beyond a 12 GB card) |

Capacity buys three separable things: ~2-3x reach at matched length, ~10x probe dynamic range, and
survival at an extension factor where the smaller model fails outright.

Also corrected: **A4.2 as worded is too strict and non-discriminating.** It tests `p_frac=0` - the furthest
possible slot - which fails BOTH models once a real s.e. exists (the 114M's `p0` is non-significant at
blocks=3, score 0.026; the earlier "A4.2 PASS on 114M" came from a blocks=1 run). It should read "**some
`p_frac` > 0 is significant**", which is what "can this model retrieve at all" actually means.

**Methodological lesson: never report a needle verdict from blocks=1.** With no spread, significance rests
solely on the 0.1-nat materiality floor, which is far too permissive - it inflated the 48M's reach 3x and
manufactured a `p0` pass for the 114M. blocks>=3 for anything that informs a decision.

### 2. Retrieval reach saturates at ~8.2k tokens (114M, 64k-adapted)

Defining **reach** = furthest distance from the probe still significant vs `null`:

| T | reach (tokens) | % of window | growth when T doubles |
|---|---|---|---|
| 4,096 | >=3,045 | 74% | - |
| 8,192 | >=4,078 | 50% | x1.34 |
| 16,384 | >=8,174 | 50% | **x2.00** |
| 32,768 | >=8,183 | **25%** | **x1.00 (+9 tokens)** |

Doubling 16k -> 32k bought **nine tokens**. At 32k, `p0.75` barely clears significance (0.096) and `p0.5`
fails - **three quarters of the window is dead weight for retrieval** while the denominator holds at ~9
nats, so this is not the probe running out of range. Usable reach is roughly **2x the 4k pretraining
length**.

Process note: I twice drew a trend from too few points (first predicting saturation from 2 points, then
denying it from 3). The reported curve is the 4-point one.

## Open caveats - these gate the conclusion

1. **Cache mismatch - PARTLY RESOLVED by the YaRN control (result below).** λ *family* is ruled out; the
   model's *adaptation* length is not.
2. **`run100` is adapted, not the base.** The `run1` (4k base) curve separates "the model has ~8k reach"
   from "64k fine-tuning bought only 8k of reach". The latter would be materially more consequential.
3. **Reach resolution is coarse** - 5 `p_frac` points bracket it to a quarter-window. Worth refining only
   at whichever T sits on the knee.

### 3. λ choice modulates retrieval STRENGTH, not retrieval REACH (YaRN control)

Ran the two decisive rows again with YaRN λ (piecewise: `1.00` x8, ramp, flat at `s`) instead of NTK
(smooth geometric ramp) - structurally very different vectors, same checkpoint, same trials.

| T | cache | reach (tokens) | p0.75 | p0.5 |
|---|---|---|---|---|
| 16,384 | NTK | **8,174** | 0.204* | 0.063* |
| 16,384 | YaRN | **8,174** | 0.285* | 0.168* |
| 32,768 | NTK | **8,183** | 0.096* | 0.041 |
| 32,768 | YaRN | **8,183** | 0.135* | 0.046 |

Reach is **identical to the token** at both lengths, so the ~8.2k ceiling is a **property of the model, not
an artifact of the NTK cache**. What the cache *does* change is the strength of the surviving signal at a
given distance (YaRN is consistently better mid-range, up to ~2.7x at 16k/p0.5) - it moves the gradient,
not the boundary.

### 4. PPL and retrieval disagree, on identical tokens

Background PPL measured from the `absent` variant on filler only (excluding probe/answer), T=16,384:

| cache | background PPL | retrieval @ p0.5 |
|---|---|---|
| NTK | **56.04** (better) | 0.063 |
| YaRN | 56.21 | **0.168** (2.7x better) |

**PPL calls the two caches indistinguishable (0.1% apart in nats) while retrieval differs 2.7x - and PPL
points the wrong way.** This is the plan's A4.4 divergence, observed directly, and it invalidates the
criterion used to pick NTK as C1's single cache (a PPL ranking). Consequence: **C2's formula arms are
promoted from optional to required.** Caveat: this PPL is on the synthetic filler+needle stream, so it is a
valid relative ranking between caches, not a clean held-out PPL.

## Infrastructure facts worth remembering

- **Local `CP_BluScreipt_Training_logs` is NOT a mirror of the server's, and run numbers collide across
  boxes.** Local `run1` is a *48M* model; the server's `run1` is the 114M 4k base. Reading the local
  config would have configured the architecture twins as a 48M model.
- **`run100` is the 64k fine-tune of `run1`**, not a pretraining reference - derived from
  `longrope_progressive.sh:127` (`ft_run = FT_RUN_BASE + i`) with `BASE_RUN=1 FT_RUN_BASE=100`.
- **`run200`'s θ sidecar is now recorded** (500000), so it is no longer a legacy checkpoint.
- A rebuild mid-sweep swapped `build/bluscriptCP_exec` under a running sweep. Harmless here (verified
  byte-identical), but build to a separate path when a sweep is in flight.

## Pending (server)

1. **Pre-flight 1** - recover `run1`'s config and pretraining budget. **Blocking fork**: if the budget is
   unrecoverable the twins cannot be budget-matched.
2. **Pre-flight 2b** - `CP_MEM_PROBE` the C1 rows and check the logits dtype. The plan's CP table rests on
   the 26.4 GB rows fitting; a bf16 model triggers an `as_type(Float32)` copy (`:1603-1604`) that adds a
   second buffer (~39.6 GB) and would not fit.
3. **`run1` reach curve** (the base-model comparison) and the **matched-λ test** at T=65,536.
4. Pre-flight 2 (`CP_DUMP_ROPE` per `(θ, s)`) and Pre-flight 3 (hybrid 2x2 @ CP=4 parity gate, blocking
   for the 48M 512k row).

Consequence for the plan: since the 48M fails A4.2, the only C1-compatible searched-λ cell (48M@16x)
cannot answer "does searched λ help *retrieval*". **C3's extra searches are now required, not optional,
and must be on the 114M.**
