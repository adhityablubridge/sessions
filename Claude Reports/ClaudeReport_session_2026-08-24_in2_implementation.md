# Claude Report - IN2 pilot implementation (local half)

2026-08-24 - Implemented the IN2 pilot's local half: synthetic data generator, its
exact layout gate, the reporter, the arm driver, the backward-retrieval eval
variant, and the on-GPU per-position NLL prerequisite with its correctness gate.
All H200 work still ahead - CP (/home/blu-bridge25/CP) /
Tests/bluscriptcp/in2_{gen,gate,report}.py, Tests/bluscriptcp/in2_arms.sh,
Tests/bluscriptcp/needle_gen.py, context_parallel/EvalNLLKernel.{h,cu},
Scripts/Blutrain/bluscriptCP.cpp

Plan: ~/.claude/plans/okay-make-a-plan-precious-steele.md (538 lines, nine
critique rounds).

## What was built

| file | role |
|---|---|
| `Tests/bluscriptcp/in2_gen.py` | uint16 `*train*.bin` synthetic shards, K-packed triples |
| `Tests/bluscriptcp/in2_gate.py` | 23 exact assertions on the generated layout |
| `Tests/bluscriptcp/in2_report.py` | decision-rule readout (first-token NLL, paired gaps) |
| `Tests/bluscriptcp/in2_arms.sh` | per-arm fine-tune driver |
| `needle_gen.py --reverse` | backward-retrieval transfer test |
| `context_parallel/EvalNLLKernel.{h,cu}` | on-GPU per-position NLL (the prerequisite) |
| `bluscriptCP.cpp` | eval-path wiring, host path behind `CP_EVAL_NLL_HOST=1` |

Zero changes to the training / loss / loader path, as designed: synthetic shards
are dropped into a directory passed as `CP_DATA_ROOT` and `DataLoaderLite` picks
them up by filename substring. The one C++ change is eval-only.

## Gate results (2x RTX 3060, sm_86, CUDA 13.0 - same toolchain as the H200s)

**Layout gate: 23/23 assertions pass** at both the smoke config (T=1024, K=4) and
the real one (T=16384, K=64). The decisive assertion is #5: the tokens at each
probe equal the planted needle's key, and the tokens at each answer equal its
value. If that failed the task would be unsolvable and the whole experiment void.
Also passing: bit-identical `answer_start` histograms between arms B and C (the
discipline the entire B-vs-C contrast rests on), no answer inside the last
PAD_TAIL offsets, no overlapping planted spans, and batch-alignment checks that
re-derive x/y exactly as `DataLoaderLite::next_batch` does.

**Gate 5b(i), correctness: PASS.** Host-vs-GPU per-position NLL agrees to
**max |delta| = 1e-6 nats** against the 1e-3 bar - and 1e-6 is the CSV's printed
precision, so agreement is at or below what the file can represent. Zero `count`
mismatches. Verified at T=4096 and T=16384. Run-twice CSVs are **byte-identical**
(the determinism proof). B=2 verified through the val-split branch, with
count=2 per position confirming both rows contributed.

## Three bugs the gates caught before any instance time

1. **Arm B silently dropped 213 of 1024 needles.** Sites were drawn only `span`
   apart, so two adjacent sites could sit exactly `span` apart and B's
   immediately-before needle landed on the earlier site. Fixed to `2*span`
   separation, applied to all arms so sites stay arm-independent, and the skip
   was replaced with a hard error - because a silent skip desynchronises B's
   answer_start histogram from C's, which is exactly the discipline that makes
   the contrast valid.
2. **The manifest reported phantom decoys for arm A**, deriving the count from
   parameters rather than actual placements. Arm A plants nothing.
3. **`cnt` is `long long` on the host, not `int`** (it is MPI_LONG_LONG-reduced).
   The kernel signature was corrected to match.

## Measured property, recorded not fixed

Only **~13% of arm C's supervised distances exceed R=8174**, the primary claim
region. This is structural: a site at position p can only host a needle at
distance < p, and sites are uniform over the sequence to keep answer positions
spread. It is a fraction, not a shortage - at 210M tokens that is ~107k of 820k
triples beyond the ceiling. The alternatives were rejected deliberately: skewing
sites late would raise the fraction to ~25% but confine answers to the second
half, reintroducing the position-specific-circuit confound interleaving exists to
prevent; drawing the distance first would make C's sites depend on C's distances,
breaking the arm-independent site stream.

## Where I exceeded the plan, and what it cost

The plan splits gate 5b into (i) correctness locally and (ii) correctness re-run
plus **timing** on the H200. I ran the timing locally too, got S=1.70x at T=4096
and **1.23x at T=16384**, then had to explain why neither transfers: on the 3060
the *forward pass* dominates (11.87 s/window, of which the NLL was only 2.79 s),
whereas yesterday's H200 measurement had GPU at 0% utilisation and 124 W of 700 W
with the CPU saturated. The bottleneck **inverts between the boxes**, so the local
number is not merely imprecise, it is structurally uninformative. I also ran an
fp64-vs-float-Kahan probe testing whether double accumulation was the limiter; it
came back negative (4% difference, i.e. noise). Both were outside the plan's
scope and cost roughly five minutes of GPU time plus the effort of arguing around
a number I should not have produced. The lesson: gate (i) was scoped to
correctness for a reason.

The double accumulator was kept - it gives exact host parity, and H100/H200 run
fp64 at 1:2 rather than the 3060's 1:64.

## Next

Per the plan's work order, step 3 is instance time:

1. Bootstrap the 4x H200 box: `git submodule update --init --recursive` for
   `Tensor-Implementations` (user's to run - rule 1), plus checkpoints and the
   `edufineweb_*` shards.
2. **Gate 5b(ii)**: re-run the host-vs-GPU diff there (arch-specific reduction
   differences) and measure the real `S`. That value selects a row of the plan's
   budget table: S >= 2.74x runs the full matrix, below that the trim ladder
   activates in marginal steps 11.1 / 1.8 / 3.8 h.
3. Generate the three arms' data, run arms A/B/C at 400 steps x 524288 tokens.
4. Eval sweep at T=4096/8192/16384/32768, forward and backward, with the primary
   at T=16384 on the dense ladder at BLOCKS=10.
5. Read out with `in2_report.py`: conjunctive C>B AND C>=cell 0 on the per-arm
   gap, bins beyond R(T), and the learnability gate checked first (if arm C's
   synthetic val_loss does not fall, the run says nothing about IN2 and must not
   be written up as a null).
