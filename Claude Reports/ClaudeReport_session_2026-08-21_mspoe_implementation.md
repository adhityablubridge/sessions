# Claude Report - Ms-PoE implementation on the H100 box

2026-08-21 - 07:20 to 08:05 - Implemented Ms-PoE (arXiv:2403.04797) per-KV-group
RoPE rescaling into bluscriptCP and verified it end to end; blocked on a model
checkpoint for the actual retrieval measurement - CP (/root/BluTrain/dist/Context_Parallelism) /
context_parallel/{LongRoPEOps.h, YARNOps.cpp, GQA_fused_fwd_sm103_cp.cu, FusedRoPESDPA.h},
Scripts/Blutrain/bluscriptCP.cpp, Tests/cp_mspoe_parity.cpp, Makefile,
mspoe_2026-08-21/scripts/mspoe_sweep.sh

## The design finding that shaped everything

Ms-PoE gives each attention HEAD its own position-scaling ratio. That is not
directly expressible here. RoPE's relative-position property

    q^T R(n_q)^T R(n_k) k = f(n_k - n_q)

holds only when Q and K rotate at the SAME rate. Under GQA one K head serves
q_heads/kv_heads query heads, so a per-query-head ratio would rotate Q and K
differently and destroy that property. The ratio must be constant within a KV
group, therefore:

    number of distinct Ms-PoE ratios = n_kvhead, NOT n_head

Consequences for our checkpoints:

| ckpt  | heads | kv | hd  | distinct ratios | usable |
|-------|-------|----|-----|-----------------|--------|
| run2  | 3     | 1  | 128 | 1               | NO - MQA, degenerates to uniform lambda (= the NTK arm already measured) |
| run7  | 3     | 1  | 128 | 1               | NO - also retrieval-dead |
| run6  | 6     | 2  | 64  | 2               | yes, weak diversity |
| run8  | 12    | 4  | 32  | 4               | ratios yes, retrieval already collapsed |
| run1  | 12    | 4  | 64  | 4               | BEST available (114M) |

The paper's models (Llama-2, Vicuna) are MHA, where n_kv == n_head, so the issue
never arises there. This reverses the earlier "use run2" recommendation: run2 is
the best retrieval architecture we have but Ms-PoE has nothing to vary on it.

## Mechanism as implemented

Composed on top of the existing base lambda rather than replacing it:

    lambda_g[i] = 1 + (lambda_base[i] - 1) * w_g,   w_g in (0, 1]

w=1 reproduces the base LongRoPE/NTK vector (full reach); small w moves that
group toward original, un-extended positions (short-range precision). That is
the paper's "some heads keep near-original positions, others get heavily
compressed ones" expressed in our per-dimension lambda parameterisation, and it
keeps Ms-PoE orthogonal to the LongRoPE axis (dim) instead of competing with it.

Selection is by tensor RANK, so no new argument appears in any intermediate
signature:
  2-D cache [L, hd]           -> one shared plane, group stride 0 (historical path)
  3-D cache [n_groups, L, hd] -> plane g for KV group g, stride L*hd

## Files changed

- context_parallel/LongRoPEOps.h - declares build_rope_cache_longrope_grouped;
  carries the GQA rationale.
- context_parallel/YARNOps.cpp - extracted fill_longrope_plane and
  validate_longrope_lambda out of build_rope_cache_longrope (so the grouped
  builder cannot drift from the single-plane one), added the grouped builder.
- context_parallel/GQA_fused_fwd_sm103_cp.cu - new trailing long
  cache_group_stride threaded kernel -> launcher -> host entry; one added line
  computes d_cache_g = d_cache + hkv * cache_group_stride. Indexing by hkv (not
  hq) is exactly what keeps Q and K in a block on the same lambda.
- context_parallel/FusedRoPESDPA.h - forward wrapper derives (cache_len, stride)
  from the cache rank and rejects n_groups != kv_heads; BACKWARD wrapper hard
  rejects a 3-D cache (the bwd kernel still assumes one plane and would emit
  wrong gradients silently).
- Scripts/Blutrain/bluscriptCP.cpp - CP_MSPOE_WEIGHTS / CP_MSPOE_WMIN, the
  grouped cache build, and six gates.
- Tests/cp_mspoe_parity.cpp + Makefile target cp-mspoe / run-cp-mspoe.
- mspoe_2026-08-21/scripts/mspoe_sweep.sh - arm driver over needle_sweep.sh.

## Verification (all on 1x H100 80GB, sm_90, CUDA 13.0)

Kernel parity test (make CP_FUSED_ROPE=1 cp-mspoe), B=1 Nq=4 Nkv=2 T=64 D=64:

  PASS 1 identity 3-D(A,A) == 2-D(A)     maxdiff=0 exactly
  PASS 3 separation A vs B differs       g0 1.95e-3  g1 7.81e-3
  PASS 2 routing group0 == 2-D(A)        maxdiff=0 exactly
  PASS 2 routing group1 == 2-D(B)        maxdiff=0 exactly
  PASS 2 routing group0 != 2-D(B)        maxdiff=1.95e-3
  PASS 4 guard n_groups != kv_heads      threw
  PASS 4 guard backward rejects 3-D      threw

Gate 2 is the decisive one: bit-equality between "group g inside a mixed 3-D run"
and "the whole model on plane g's lambda" can only hold if the group read its own
plane AND K was rotated with the same lambda as its query heads.

Config gates, each verified to fire with its own message:
  no CP_LONGROPE_FACTORS / kv_heads=1 (MQA) / attn_mode=ulysses without
  CP_ULYSSES_CPLOCAL / no CP_EVAL_PPL (training) / CP_ATTN_FUSION=split /
  weight-count != kv_heads.

End-to-end through bluscriptCP at T=16384, 114M arch, random init (no checkpoint
yet), CP_ATTN_MODE=ring:
  base vs CP_MSPOE_WEIGHTS=1,1,1,1  -> eval CSVs BYTE-IDENTICAL
  base vs 0.25,0.5,0.75,1.0         -> differ from line 3 on
  built planes: lam_last 1.75 / 2.50 / 3.25 / 4.00 = 1 + (4-1)*w, as specified
PPL ~58699 for all arms because the weights are random - that number carries no
information beyond "the pipeline runs".

## Environment work done on this box (it was a fresh clone)

- OpenMPI was absent -> installed 4.1.6 (openmpi-bin, libopenmpi-dev).
- BluTrain/Tensor-Implementations was an EMPTY submodule; user initialised it.
- Clean-built libprofiler.so + libtensor (.a 78 MB, .so 62 MB) and
  build/bluscriptCP_exec at SM_ARCH=90 (auto-detect is correct on H100).
- Data_Loader/Data/edufineweb_val_00000.bin transferred (190 MB, uint16, ~95 M
  tokens - matches what needle_gen.py's load_filler expects).
- Still missing: model checkpoints. checkpoints_bluscriptcp/ does not exist, so
  no retrieval number can be produced yet. Also lost with the old instance:
  context_rot_2026-08-19/results/*.csv (the 10 baseline needle CSVs); state/ and
  scripts/ survived.

## Next

1. Land blumodelcp_run1_step_1387.ckpt (114M, kv=4, hd=64) - or run6 step_1300
   (48M, kv=2) as a cheap second point.
2. Run mspoe_2026-08-21/scripts/mspoe_sweep.sh. Arms: base, w1.0 (parity),
   w0.5, w0.25. All arms share one needle set so trials are byte-identical.
   Cost on this box: ~80 s/eval at T=16384, 35 evals/arm, 4 concurrent on one
   card via GPUS="0,0,0,0" -> ~12 min/arm.
3. Prior expectation, stated before measuring: our own lambda control showed two
   structurally different lambda vectors giving IDENTICAL reach, so Ms-PoE should
   move mid-range retrieval STRENGTH (the denominator / gradient), not the
   boundary d@0.2. If d@0.2 moves, that contradicts the control and is the
   interesting outcome.
4. Backward support (cache_group_stride in gqa_fused_rope_cp_backward) is NOT
   implemented and is currently refused at runtime. Only needed if we ever want
   to fine-tune under Ms-PoE, which the paper does not.

---

# RESULTS (2026-08-21, 09:00-12:42, 1x H100)

## Checkpoints actually available

Read from the checkpoint headers, not from the run numbers (numbering is reused
across boxes):

  blumodelcp_run1_step_1300.ckpt  48,077,184 params  d=384 qk_gamma[128]
                                  -> 3 heads, kv=1 (MQA), hd=128, 6L
  blumodelcp_run6_step_1300.ckpt  48,076,416 params  d=384 qk_gamma[64]
                                  -> 6 heads, kv=2,       hd=64,  6L

Both 48M, not 114M (577 MB = 48M x 4 B x 3 for weights + Adam m/v; the 114M
model would be 1.37 GB). run1 here is the arch twin_evals.sh calls A2/run2.
run1 is MQA so it cannot express Ms-PoE at all - the gate rejects it. All
results below are run6, which gives exactly TWO distinct ratios.

## Baseline reproduction (the anchor)

s=1 / T=4096 base row vs the recorded 2026-08-19 run6 numbers:
  denom_nats 4.261935 vs 4.262 | floor 9.561222 vs 9.561 | d@0.2 in [1600,1850]
  vs 1,675 | #sig 10 vs 9 (one borderline point; RESUME already flags #sig as
  partly a noise measure and directs to d@0.2 for the boundary).
So the fresh CUDA-13.0 / sm_90 build, the Ms-PoE-patched kernel at stride 0, and
the transferred checkpoint + filler reproduce a number measured on the previous
instance. Everything below is anchored to that.

## Parity (Ms-PoE off must cost nothing)

CP_MSPOE_WEIGHTS=1,1 reproduced the base arm EXACTLY on every scored metric, at
all 11 distance points, at BOTH s=2 and s=4 - two independent 65-eval sweeps on
a trained checkpoint. With the kernel-level bit-exact gates that is as strong as
this can be verified.

## s=4 (T=16384) - abandoned as unmeasurable

Base row: denom 1.038 nats (vs 4.262 at s=1), floor 12.735, and NOT ONE
significant point except adjacency. Scores wander 0.05-0.13 with se ~0.11 and
are not monotone in distance. A 4k-native model on cache-only NTK x4 has
retrieval only at adjacency, so an Ms-PoE effect there would be undetectable.
Stopped the w0.5/w0.25 arms at s=4 after w1.0 (parity) completed and moved to
s=2. w0.5 retains 16 s=4 markers if that row is ever wanted.

## s=2 (T=8192) - the measurable row

              weights    denom     floor   absent     null  #sig
  base         (none)   2.5770   11.2922  13.8693  13.7723    3
  w1.0        1.0,1.0   2.5770   11.2922  13.8693  13.7723    3   (== base)
  w0.5        0.5,1.0   2.6394   11.2031  13.8425  13.7521    2
  w0.25      0.25,1.0   2.7150   11.1104  13.8253  13.7379    1

RAW NLL deltas vs base (positive = Ms-PoE retrieved better), se ~0.10:
  adjacency p1 : +0.089 (w0.5)  +0.182 (w0.25)   <- monotone in weight spread
  p0.7         : -0.190          -0.143
  p0.75        : -0.073          -0.203
  p0.5         : +0.063          +0.135
  absent       : +0.027          +0.044          <- GLOBAL shift, not retrieval
  null         : +0.020          +0.034

Two things must be separated here:
1. absent and null both improved monotonically, so ~0.03-0.04 nats of the gain
   is a GENERAL LM improvement at every position, not retrieval.
2. denom = absent - floor GREW because adjacency improved. Since score is
   normalised by denom (p1 == 1 by construction), a bigger denominator
   mechanically SHRINKS every other score. Comparing d@0.2 across arms with
   different denominators is therefore not apples-to-apples, and the raw #sig
   drop 3 -> 2 -> 1 is partly this artifact.

Renormalising every arm on the BASE denominator (2.5770) to isolate distance
from the denominator change:
  d@0.2 crossing p_frac:  base 0.757 | w0.5 0.767 | w0.25 0.780
Higher p_frac = needle CLOSER to the probe = SHORTER reach. In distance terms
~1,990 -> ~1,910 -> ~1,800 tokens, i.e. reach got ~9% WORSE, monotone in weight
spread, at roughly 2 sigma on a single row.

## Conclusion, stated against the pre-registered expectation

Pre-registered before measuring: Ms-PoE should move mid-range retrieval STRENGTH,
not the boundary, because our own lambda control showed two structurally
different lambda vectors giving identical reach.

Observed, with 2 groups on run6 at s=2:
- Reach did NOT improve. It degraded slightly and monotonically with weight
  spread (~9%, ~2 sigma, one row).
- Strength at DISTANCE did not improve either; deltas scatter +/-0.2 nats and
  flip sign between adjacent distance points, which is the signature of noise
  at se ~0.10.
- What did improve, monotonically and coherently: ADJACENCY retrieval (+0.18
  nats at w0.25, ~1.9 sigma) and a small global NLL shift (+0.04).

So the prior survives in its strong form: position rescaling did not move the
boundary. It did not even buy strength at distance here.

## Caveats that bound this result

1. TWO groups only (kv=2). The paper is per-head on MHA models with many heads;
   two ratios is the weakest expressible version of the method. A 4-ratio test
   needs the 114M kv=4 checkpoint (1.37 GB), which is not on this box.
2. One row, one seed, 5 blocks, 64 trial-tokens per point -> se ~0.10 nats
   against effects of 0.1-0.2 nats. Underpowered; the sign flips across adjacent
   distances say so directly.
3. run6 is a single training run, 4k-native, extended by NTK with no fine-tune.
4. Only the s=2 row is interpretable; s=1 is a no-op for Ms-PoE by construction
   (lambda_NTK(1) is all ones) and s=4 is below the noise floor.

## Performance finding (relevant to every future eval sweep)

The eval is CPU-BOUND, not GPU-bound. Measured on this box:
  GPU utilisation 0%, power 124 W of 700 W, at both 4 and 8 concurrent evals
  disk 9.9 GB/s (page-cached, 144 GB RAM free) - not I/O bound
  6.28 GB GPU memory per eval, so 80 GB fits ~12
  4 concurrent: 21.9 s/eval (65-eval arm in 1402 s)
  8 concurrent: 20.4 s/eval (65-eval arm in 1303 s)  -> only 7% faster for 2x
  solo eval, 1 window 32.2 s -> fixed overhead ~25 s, per-window ~7 s

Cause, confirmed in the code: the CP_EVAL_PPL path copies logits to the host and
does the logsumexp there, chunked, under `#pragma omp parallel for` - about
8192 tokens x 8 windows x 50304 vocab x 2 passes with exp() per eval. Each
process ALREADY spreads that across all 32 cores, so 4 concurrent processes
saturate the CPU and adding 4 more only oversubscribes it. That is why neither
more concurrency nor sequential execution helps: the GPU is idle either way.

Two real levers, neither applied (they would change the binary mid-experiment):
- The LongRoPE search path at bluscriptCP.cpp:1742 already computes "mean NLL on
  the GPU (no host logits copy, no manual logsumexp)". The eval path could reuse
  that and skip ~13 GB of D2H plus the host reduction per eval.
- The resume fast-forward (`skip_batches`, 41,600 iterations re-mapping shards)
  positions a TRAINING loader the needle eval never reads - it reads
  CP_EVAL_TOKENS_BIN. Guarding it with `if (!eval_ppl_mode)` removes a large
  slice of the 25 s fixed overhead.

## Next

1. To test Ms-PoE as designed, get the 114M kv=4 checkpoint (4 ratios) - that is
   the only way to raise head diversity beyond 2 here.
2. Power: raise blocks from 5 to ~20 on the s=2 row to halve se before drawing
   any conclusion about a 0.1-0.2 nat effect.
3. Apply both perf fixes first - they make (2) affordable.

---

# PART 2: VERIFIED AGAINST THE REFERENCE IMPLEMENTATION, AND THE NATIVE-LENGTH RUN

Checked our implementation against arXiv:2403.04797 and github.com/VITA-Group/Ms-PoE
(utils/modify_arch/llama.py, src/ms_poe.sh). Reference behaviour, verbatim:

    compress_ratio = min_ratio + (max_ratio-min_ratio) * (arange(num_heads)/num_heads)
    t = t / compress_ratio
    freqs = torch.einsum("ki,j->kij", t, self.inv_freq)
    average = attn_weights.mean(-1).unsqueeze(-1)
    outlier = - (attn_weights > 3*average).float().mean(-1)[:,-1]
    head_orders = outlier.argsort()
    cos = cos[self.head_order, :, :];  sin = sin[self.head_order, :, :]

src/ms_poe.sh: lmsys/vicuna-7b-v1.5, compress_ratio_min 1.2, compress_ratio_max 1.8,
apply_layers "2..31" (layers 0 and 1 excluded).

## What our first implementation got right

- Per-head-group ratio with one cos/sin plane per group, gathered per head. Same
  structure as our [n_groups, L, hd] cache indexed by group.
- Q and K share ONE ratio. The repo gathers a single cos/sin per head and applies
  it to both q and k. Our hkv indexing enforces exactly that, and our GQA finding
  is the correct generalisation: the repo's per-head gather is only valid because
  Vicuna is MHA (num_heads == num_kv_heads).
- Inference-only. Our backward refuses a 3-D cache.

## What was WRONG (three deviations)

1. UNIFORM vs PER-DIM. The repo divides the POSITION uniformly (t / r) and leaves
   inv_freq untouched, so every frequency dim is scaled identically - that is PI.
   Our first version used lambda_g[i] = 1 + (lambda_base[i]-1)*w_g, a per-dim
   NTK-shaped ramp. Different function. FIXED: new CP_MSPOE_RATIOS /
   CP_MSPOE_RMIN / CP_MSPOE_RMAX give lambda_g[i] = r_g constant over i, which is
   our per-dim expression of t/r, and needs NO base lambda. The repo's g/n (not
   g/(n-1)) divisor is reproduced, so rmax is approached, never reached: at kv=4
   with 1.2/1.8 the ratios are 1.2, 1.35, 1.5, 1.65.
2. LAYER SELECTION missing. FIXED: CP_MSPOE_LAYERS (accepts "2-11" ranges);
   unselected layers keep the unscaled plane.
3. RATIO ASSIGNMENT BY MEASURED POSITION-AWARENESS - still NOT implemented. The
   repo sorts heads by an attention-outlier statistic and assigns the ladder in
   that order. We assign by KV-group index. The repo ships --head_type "normal"
   as an ablation for precisely this, so OUR ARMS ARE THAT ABLATION, not the full
   method. The startup banner now says so on every run.
   NOTE, correcting an earlier claim in this report: that metric needs only the
   LAST attention row ([:,-1]), i.e. one q vector against all keys per head - NOT
   the full [T,T] matrix. It is therefore feasible alongside a fused kernel as a
   small separate op. The earlier "blocked by the fused kernel" assessment was
   too pessimistic.

## The deviation that invalidated the run6 regime

Ms-PoE uses ratios > 1 with NO base lambda: it COMPRESSES positions below the
native window to fight RoPE decay inside it. The paper runs Vicuna-7B on ~4k
MDQA inputs - native context, no extension. Because our first parameterisation
composed onto a base lambda, the s=1 row looked like a no-op and was skipped, and
everything was measured at s=2/s=4 (extension). That is not the method's regime.
The run6 result therefore reads only as "per-group diversity, index-ordered,
composed on NTK, under extension, did not help".

## THE NATIVE-LENGTH RUN (run100, the paper's regime)

run100 = 114,151,680 params, d=768, 12 heads, kv=4 (-> 4 distinct ratios), hd=64,
12 layers, weight-tied, loss 3.90073 at step 1387. Read from the checkpoint
header, not from the run number. T=4096 NATIVE, 5 blocks, 7-point grid, ring.

Baseline signal is far better than anything in the run6 sweep:
  denom 8.8434 nats, floor 4.8480, 4 of 5 points significant, curve monotone
  (0.0234 / 0.1504 / 0.2181 / 0.4460 / 0.9989). run6 at T=4096 had denom 4.262.

PARITY: ratios 1,1,1,1 reproduced the base arm EXACTLY (all metrics, all points).

  arm                       ratios   denom    floor  absent  #sig
  base                   no Ms-PoE  8.8434   4.8480 13.6914     4
  r1.0                     1,1,1,1  8.8434   4.8480 13.6914     4   (== base)
  paper    1.2,1.35,1.5,1.65 L2-11  8.4594   5.3474 13.8067     4
  wide     1.0,1.25,1.5,1.75 L2-11  8.7067   5.0973 13.8040     4

RAW NLL delta vs base (+ = Ms-PoE better), with se:
  p_frac      se    d_paper    sigma      d_wide    sigma
  p0       0.103    -0.1419    -1.4       -0.1442   -1.4
  p0.25    0.132    +0.0812    +0.6       -0.2746   -2.1
  p0.5     0.181    -0.0164    -0.1       -0.4641   -2.6
  p0.75    0.211    -0.7175    -3.4       -0.6389   -3.0
  p1       0.054    -0.4994    -9.2       -0.2493   -4.6

## Result

The paper's own configuration (1.2-1.8, layers 2-11), with ratios assigned by
index rather than by position-awareness, SIGNIFICANTLY DEGRADED near-range
retrieval and produced NO significant gain anywhere:
  adjacency  -0.499 nats at 9.2 sigma
  p0.75      -0.718 nats at 3.4 sigma
  best case  +0.081 nats at p0.25, 0.6 sigma - NOT significant

The wider dose (1.0-2.0) is monotonically worse at every distance, i.e. a clean
dose-response in the HARMFUL direction. That the harm scales with ratio spread
argues the effect is real, not noise.

TRAP AVOIDED: the renormalised d@0.2 crossing MOVED OUTWARD for the paper arm
(2,317 -> 2,520 tokens, +8.8%) and would have read as "reach improved". It is
driven entirely by sub-sigma mid-range changes plus a genuine near-range
COLLAPSE, which flattens the curve. Reporting that +8.8% as a win would have been
wrong. Always read raw NLL beside the normalised score.

## Interpretation

This is consistent with the paper's own mechanism claim, seen from the negative
side: WHICH head receives WHICH ratio is the load-bearing part. Assigning
compression ratios to arbitrary head groups costs local resolution in heads that
needed it (near-range collapse) without buying mid-range recovery. The repo
ablates exactly this with --head_type "normal". So our result does not refute
Ms-PoE; it reproduces the failure mode the paper's head-ordering step exists to
avoid.

## Caveats

1. Head-order assignment absent - the component the paper credits. Until it is
   implemented no claim about Ms-PoE proper can be made from these numbers.
2. 4 ratios (kv=4) vs 32 per-head ratios in the paper.
3. 5 blocks, 7-point grid, one seed. A BLOCKS=10 pass was launched to halve se on
   the one apparent (sub-sigma) mid-range gain.
4. run100 is a 114M base LM, not an instruction-tuned 7B. Our probe shows
   MONOTONIC decay, not a U-shape, so there is no lost-in-the-middle dip here for
   Ms-PoE to repair - which may itself explain the absent gain. The paper's method
   targets a U that our model does not exhibit.

## Next

1. Implement head ordering: per layer, one attention row (last query position)
   per head, outlier = -(attn > 3*mean).mean(), argsort, permute the ratio ladder.
   That is the difference between the ablation and the method.
2. Caveat 4 is the deeper point: verify our model even has a middle to lose
   before expecting a middle-fixing method to pay.

## CLARIFICATION: what the paper actually claims (checked against the abstract)

Ms-PoE claims "an average accuracy gain of up to 3.8 on the Zero-SCROLLS
benchmark". The target is LOST-IN-THE-MIDDLE: accuracy when the relevant
information sits in the MIDDLE of a context the model already handles. The paper
claims NO context-length extension and NO increase in retrieval distance.

This reframes our whole experiment. Ms-PoE makes retrieval more UNIFORM ACROSS
POSITIONS inside a fixed window; it does not make the window reach further. Our
needle harness measures retrieval-vs-DISTANCE (reach, d@0.2), which is not the
quantity Ms-PoE improves.

Combined with the finding already recorded in .mdfiles/CombatingContextRot.md -
that our probe shows MONOTONIC decay, not a U, because the left arm of the U is
LEARNED by instruction-tuned QA models and our base LM never had that training -
the conclusion is a SCOPING one:

  This experiment could not have demonstrated Ms-PoE's benefit even with a
  perfect implementation. run100 is a plain base LM with no middle dip for a
  middle-fixing method to repair, and our metric does not track position
  uniformity.

That is not a refutation of the paper. It is a statement that the method was
pointed at the wrong model and scored with the wrong metric.

To test Ms-PoE properly one would need BOTH:
  1. the head-ordering step (position-awareness sort), and
  2. a model that actually exhibits a U-shaped position curve - which for us
     means an instruction-tuned / QA-trained model, or at minimum a metric of
     accuracy-vs-position rather than retrieval-vs-distance.
Until (2) holds, (1) is not worth building for this purpose.

## CORRECTION: the "no U-shape" finding was measured on 114M, not 48M

.mdfiles/CombatingContextRot.md IS present on this box (post-merge path
dist/Context_Parallelism/.mdfiles/). Two corrections to Part 2 above:

1. Its anchor measurements are on the "114M, 64k-adapted" checkpoint (line 11),
   and its own warning (lines 33-37) states: "Our probe does not show a U. It
   shows monotonic decay with distance - our needle at position 0 scored the
   WORST (0.002-0.026), with no primacy bump at all." So the no-U finding was
   never a 48M-only result; it was established on 114M. Earlier statements in
   this session attributing it to 48M were wrong.
   Today's run100 base row independently replicates it: p0 = 0.0254 (10 blocks,
   se 0.082), inside the document's 0.002-0.026 range, on a different 114M
   checkpoint at native 4k.

2. The document's section 1 gives the MECHANISM, which settles the "is it model
   size?" question: the right arm of the U is architectural (RoPE decay) and we
   have it; the left arm is LEARNED and we do not. Two stated reasons, neither of
   which is parameter count:
     - our needle is RANDOM TOKEN IDs, and an attention sink receives attention
       MASS without its content being retrieved ("closer to a no-op than to a
       memory"), so sitting at position 0 buys nothing;
     - our model is a base LM, never instruction-tuned, never trained on any task
       rewarding use of position 0. The primacy arm in the literature is measured
       on instruction-tuned models doing multi-document QA.
   So a 7B BASE LM probed this way would likely also show no left arm. Vicuna-7B
   has one because it was instruction-tuned on multi-doc QA. Scale is not the
   operative variable; training regime and needle content are.

Also note the document PREDICTED this outcome on 2026-08-13, before any of
today's work: section 3 "Would it fix our problem? probably not the ceiling ...
expect a better mid-range gradient rather than more reach", and the section 7
table row "Ms-PoE - likely strength, not reach". It also predicted the exact
implementation shape (an [n_heads, seq_len, head_dim] cache plus a kernel change
to index it, needing its own parity gate) - which is what was built, modulo the
GQA constraint forcing KV-GROUPS rather than heads, which the document did not
anticipate.

Finally, section 4's blocker ("needs per-token attention weights, and our fused
kernel never materialises the [T,T] matrix") applies to Hsieh et al. attention
CALIBRATION. It does NOT apply to Ms-PoE's head-ordering metric, which reads only
the last attention row. Extending that blocker to Ms-PoE earlier in this session
was an error.

## FINAL NUMBERS (10 blocks) - this REVISES the 5-block conclusion

All four arms completed at 10 blocks (2026-08-21 14:43 UTC / 20:13 IST). Parity
re-confirmed at 10 blocks: ratios 1,1,1,1 identical to base on every metric.

  arm      denom    floor   #sig
  base    8.8827   4.8270      4
  r1.0    8.8827   4.8270      4   (== base, exact)
  paper   8.5022   5.2925      4
  wide    8.7651   5.0378      4

RAW NLL delta vs base (+ = Ms-PoE better), by DISTANCE from the question:
  distance  p_frac     se        paper           wide
     4,088      p0  0.082   -0.096 (-1.2s)   -0.116 (-1.4s)
     3,066   p0.25  0.099   +0.187 (+1.9s)   -0.205 (-2.1s)
     2,044    p0.5  0.107   +0.090 (+0.8s)   -0.379 (-3.5s)
     1,022   p0.75  0.146   -0.609 (-4.2s)   -0.530 (-3.6s)
         0      p1  0.089   -0.466 (-5.3s)   -0.211 (-2.4s)

REVISION: at 5 blocks the mid-range point was +0.081 (0.6 sigma) and I reported
"no significant gain anywhere". Doubling the blocks moved it to +0.187 (+1.9
sigma), with the middle also positive (+0.090, +0.8 sigma). So the paper's
predicted MECHANISM IS VISIBLE in our data after all:

  mid/far range IMPROVES (+0.187 at 3,066 tokens, +0.090 at 2,044)
  near range DEGRADES  (-0.609 at 1,022, -0.466 at adjacency)

That is exactly the trade Ms-PoE describes: compressing positions makes distant
text effectively closer (gain) at the cost of local resolution (loss). The gain
is borderline (1.9 sigma, not conventionally significant); the loss is large and
certain (4-5 sigma). Net effect on this model is negative, but the direction of
the mid-range effect is the paper's, not noise-shaped.

Dose-response supports this reading: the WIDE arm (1.0-2.0) is harmful at every
distance including mid-range, so 1.2-1.8 sits near an optimum and 1.0-2.0
overshoots - consistent with the paper's own choice of 1.2/1.8.

Why the net is still negative for us, unchanged from the analysis above: our
model retrieves out to ~3,000+ tokens of a 4,096 window, so the middle was never
broken (score 0.22, significant). The gain lands in a region that already worked,
and the cost lands at 0-1,000 tokens where all our real performance is. And the
one genuinely dead region (the outermost ~1,000 tokens, p0) did NOT improve
(-1.2 sigma) - that is the missing-capability regime, not the suppressed regime,
so no amount of position rescaling reaches it.
