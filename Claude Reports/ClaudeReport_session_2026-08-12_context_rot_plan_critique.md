# ClaudeReport_session_2026-08-12_context_rot_plan_critique

2026-08-12 - Three rounds of adversarial critique on the context-rot / architecture-scaling plan (retrieval probe + arch twins + sweep + 4B gating) - CP / .mdfiles/polished-cuddling-graham.md

## Scope
Read-only review of the top plan in `.mdfiles/polished-cuddling-graham.md` (Parts A-D). The three
plans below it in the same file (candidate-parallel search, CP>1 LongRoPE eval, original LongRoPE
implementation) were read only for cross-consistency.

## Round 1 findings (all subsequently addressed by the user)
1. Gate A4.1 asserted `near < far < absent` "by construction" - would have auto-failed the headline
   finding (total rot = `far` at `absent` level).
2. Twin `n_head=6/head_dim=128` claimed as clean `head_dim` isolation; at fixed `d_model` the two are
   algebraically locked, and `n_kvhead` also moved - a joint contrast.
3. Part-D decision rule required ">=32x searched-lambda retrieval evidence" while Part C forbade new
   searches and only one such vector existed - rule unevaluable.
4. `CP_ROPE_THETA` env-only knob with no train/eval agreement assert - silent wrong-base evaluation.
5. YaRN lambda seeds are `base`-dependent; reuse across theta twins makes cells non-comparable.
6. Three conditions were not position/token-matched - differencing NLLs at different points on a
   position-varying curve.
7. No variance source; `CP_EVAL_PPL` emits count/mean_nll only. "clearly >0" undefined.
8. `near`/`far` overloaded as both condition and position.
9. Grid overcounted ~2x (near/absent do not depend on p).
10. Sweep cost unestimated ("the long pole").
11. Topology parity gating inconsistent (CP=8 hybrid listed verified vs "pure ring >=4 broken").

## Round 2 findings
1. Internal contradiction: `run100` treated as 4k pretraining reference in Part B and as the
   64k-fine-tuned checkpoint in Part C - invalidates the capacity axis.
2. C1 arithmetic (~130) assumed both caches in all 10 cells; actual ~82.
3. Twin B moves capacity in the same direction as the effect under test - upper bound, not isolation.
4. Null threshold at B=3 blocks is a 2-dof estimate; null scoped per-(model,T) but normalised by
   cache-specific denominators.
5. `die` on missing theta metadata would brick all existing checkpoints.
6. Unfilled placeholders: denominator margin; C2 cost (works out to ~1.2x C1, i.e. the largest line).

## Round 3 findings (current, open)
1. **Run counts undercount by 5x** - `--blocks 5` needs 5 separate .bin files and 5 eval invocations
   (packing them into one file would let per-position aggregation average across blocks and destroy
   the spread). C1 = 385 evals not 77; C2 = 840 not 168; the cost-gate calibration cell measures 1/5
   of a cell. "Reduce --blocks at >=64x" is missing from the trim order despite being the largest lever.
2. A4.2 / A4.4 / C-Q1 / D3 still gate on the score after A2 moved significance to raw NLL.
3. `null`-per-cache: right conclusion, stale argument (cites the score normalisation A2 abandoned;
   real reason is that `NLL_null` itself shifts with the RoPE cache).
4. `--trials N` never pinned (8 / N / 16 appear in three places) - 2x swing in the UNKNOWN line item.
5. C1 selects NTK on a PPL ranking, in a plan whose thesis is that PPL does not predict retrieval.
   Suggested: carry YaRN at one mid factor as a spot check.
6. `run1`'s config/step count is an open dependency stated parenthetically while twins are
   budget-matched to it - should be a blocking pre-flight item.

## Assessed sound
Fixed-slot needle layout; `null` variant design; theta precedence table incl. the legacy warn-not-die
row; per-topology parity gating with the "a CP bug mimics context rot" rationale; the `run1`-based C1
and the `longrope_progressive.sh:127` derivation that established it; A/B bracket framing;
forward-vs-training memory split in Part D.

## Files touched
None - review only.

## Round 4 findings (current, open)
1. Stale lines in A1 - the section A2/Part C both cite as source of truth: the variant table still says
   `null` is "once per (model,T), not per cache" (contradicted by A2 and Part C, which now specify
   per-cache), and the header still says "6 runs, not 15" (it is 7 variants).
2. Trim ladder rung 1 is a no-op: "--trials 8->4 at >=64x" but N is already pinned to 4 at >=64x.
3. "N and B trade off at fixed cost" is false - cost is N*B forwards PLUS B model loads (each block is
   a separate invocation), so at fixed N*B a larger B costs strictly more. C1 implies ~455 model loads.
   "Cut N before B" is right for dof, wrong for wall-clock; say both.
4. Cost gate calibrates on one 512k variant-cell and multiplies by the total count, but cost is
   superlinear in T and 6 of 10 (model,T) cells are <=128k - the projection overestimates and may
   trigger unnecessary trimming. Use two anchors (32k + 512k) and fit the T scaling.
5. Forward-pass total is 3,080 not ~2.9k (63 variant-cells at N=8, 28 at N=4, x B=5).

Round-3 items all closed: eval counts restated with the B multiplier, N pinned, raw-NLL threshold
propagated to A4.2/A4.4/Q1/D3, null-per-cache justification corrected, YaRN spot check added at 32x,
run1 provenance promoted to a blocking pre-flight fork.

## Round 5 findings (current)
Round-4 items all closed (A1 stale lines, trim rung 1 corrected to 4->2 at >=64x, N/B cost asymmetry
stated with the 455 model loads, two-anchor cost gate, 3,080 forwards). Plan self-added a good item:
re-measure an anchor on twin B, whose 1536 attention inner dim changes per-forward cost.

Remaining, both in the cost model and both the same confound pattern the plan polices elsewhere:
1. The two cost anchors differ in three variables at once (48M/32k/N=8 vs 114M/512k/N=4), so the
   fitted "T scaling" absorbs model size and N into the T exponent. Anchor both at the same model and
   divide out N, or add a third point.
2. Cost is not a function of T alone: CP degree rises with T across C1's rows (CP=1/2 at 32k ->
   CP=8 at 512k) and the topology switches uly -> hybrid 4x2, so per-rank work is ~O(T^2/CP) plus
   comm. Fit over (T, CP, topology) points, or anchor at a fixed CP degree.

Verdict: nothing structural remains; safe to start. Pre-flight 1 (recover run1's provenance) is the
only item that can still fork the plan and should go first.

## Round 6 findings (current)
Round-5 items closed by dropping the T-exponent fit entirely: cost is now measured row-by-row (all 7
variants within a (model,T,cache) cell cost the same), anchors double as real C1 data, rows ordered
cheap->expensive. New CP/topology table added per row.

Remaining, both raised by that new table:
1. The "hybrid 4x2 @ CP=8 gate is not blocking for C1" claim is contingent on 26.4 GB/rank of fp32
   logits fitting on a 49 GB card at the 128k/256k/512k rows - the same figure Part B calls marginal,
   and double the 13.2 GB it calls comfortable. If any row OOMs, C1 falls back to CP=8 -> hybrid 4x2
   for the 114M -> the deferred gate becomes blocking mid-sweep. Fix: run CP_MEM_PROBE on the C1 rows
   in Pre-flight (minutes), or state the fallback explicitly.
2. Unclaimed ~8x throughput: the <=32x rows are CP=1 and account for 63 of C1's 91 variant-cells, so
   8 can run concurrently on the 8 GPUs. Wall-clock would then be dominated by the two multi-GPU rows.
   Worth a line in A3 - it may remove the need for the trim ladder entirely.

Verdict: executable. Start with Pre-flight 1 (run1 provenance); fold the C1 mem-probe into Pre-flight 2.

## Round 7 findings (current)
Round-6 items closed: Pre-flight 2b added (CP_MEM_PROBE on the C1 rows, explicit CP=8 fallback that
re-promotes the hybrid 4x2 gate to blocking), and A3 gained the CP=1 8-wide fan-out with a
no-determinism-argument-needed rationale. Plan self-added a sharper catch than mine: the CSV path's
`logits.as_type(Float32)` (:1603-1604) allocates a SECOND buffer if logits are bf16 - 26.4 GB fp32 on
top of 13.2 GB bf16 ~ 39.6 GB, which does not fit regardless of the per-rank arithmetic.

Remaining, on the row the plan itself calls the bottleneck:
1. A3 drops to sequential for the CP>=2 rows on the reason that they "need every rank of their group" -
   but a CP=2 group needs 2 GPUs, not 8. On 8 GPUs that is 4 concurrent groups at 256k and 2 at 512k,
   same per-card memory, same determinism argument as CP=1. Since the fan-out's own conclusion is that
   wall-clock becomes dominated by those two rows, serializing them forfeits 4x/2x on the dominant term.
   Scheduler should be a GPU-GROUP slot loop (slots = 8 / CP_SIZE), not CP=1-only with a sequential
   fallback. Pin each group to a contiguous device set via CUDA_VISIBLE_DEVICES so cudaSetDevice(rank)
   binds correctly, and include the concurrent-group count in the pgrep pre-flight.

Verdict: executable; this is throughput, not correctness.

## Round 8 findings (final)
Round-7 closed: A3 now schedules on GPU-GROUP slots (slots = 8/CP_SIZE, 8/4/2 concurrent groups by
row), with CUDA_VISIBLE_DEVICES pinning, one-eval-per-card as the memory guard, and the group count in
the pgrep pre-flight.

Two items introduced by the fan-out itself:
1. Host RAM now scales with the concurrent-group count and CP_MEM_PROBE measures GPU only. The CSV path
   copies logits to host: 26.4 GB/process at 128k x 8 concurrent CP=1 evals ~ 211 GB unless the copy is
   chunked. The LongRoPE section of the same document made seq-chunking to_cpu MANDATORY for this exact
   reason (52.8 GB across 4 ranks) - confirm it landed in the C1 path and size
   chunk * V * 4 * concurrent_groups against box RAM. Add to Pre-flight 2b.
2. A3's CP-per-row table is contingent on 2b: it hardcodes CP=1 for 32k-128k, but 2b exists because
   those rows may not fit (bf16 as_type double-buffer). If 128k moves to CP=2, concurrency drops 8->4
   and the "trim ladder may be unnecessary" conclusion needs re-deriving. One line to link them.

Verdict after 8 rounds: nothing substantive left. The plan has an unambiguous unit of work, measured
rather than projected costs, pre-registered thresholds, per-topology gating, and honest labelling of
what each contrast can claim. Recommendation: stop reviewing, start Pre-flight 1.
