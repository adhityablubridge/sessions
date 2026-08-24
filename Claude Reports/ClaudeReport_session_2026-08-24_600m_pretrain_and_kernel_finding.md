# Claude Report - 600M pretrain, the hd=64 kernel finding, and a measured stopping criterion

2026-08-24 - Redirected the IN2 pilot from the 120M to a fresh 600M after finding an
18x attention-kernel penalty on hd=64; pretrained the 600M and proved via probe that
the base is already sufficient at 1/8 of the planned tokens - CP
(/root/BluTrain/dist/Context_Parallelism) / pretrain_600m.sh, probe_600m.sh,
Tests/bluscriptcp/{in2_gen,in2_gate,in2_report}.py, in2_arms.sh, needle_sweep.sh,
context_parallel/EvalNLLKernel.{h,cu}, Scripts/Blutrain/bluscriptCP.cpp

Plan: ~/.claude/plans/okay-make-a-plan-precious-steele.md (593 lines, nine critique rounds)

## HEADLINE: the attention kernel, not model size, sets the cost

The 120M trains at **2,430 tok/s** on an H100 -- 216 s/step, GPU at 100% util but only
209 W of 700 W. Root cause: `hd = d_model/q_heads = 768/12 = 64`, and hd=64 cannot
reach the Stage-3 Hopper wgmma path. The CP-local fallback launches
`dim3 BLOCK(32)` -- ONE WARP per block -- achieving **2.1 TFLOPS, 0.2% of peak**.
Decomposing the measured forward: attention is 9.9 TFLOP at 2.1 TFLOPS = 4.6 s, while
the MLP/projections' 7.5 TFLOP take ~0.04 s in cuBLAS. So 99% of forward time is
attention, and the dense parts of the model are fine.

Measured A/B on a 600M at hd=128, same model, same T, only the kernel changed:

| path | fwd | bwd | tok/s | TFLOPS |
|---|---|---|---|---|
| fused (WMMA) | 10.100 s | 18.535 s | 2,289 | 13.5 |
| **split (Hopper wgmma)** | **0.523 s** | **1.080 s** | **40,883** | **241** |

**17.9x**, with step-0 loss agreeing to 1e-5 (11.151080 vs 11.151089) -- same math.
At matched T=16384 the 600M is **8.8x faster than the 120M while being 6x larger**.

`cp-causal-cp-sm90` parity: ALL PASS, including maxdiff 0.000e+00 against the base
kernel in the degenerate case. So RESUME.md's "NEVER RUN yet" meant never exercised in
training, not numerically unvalidated. Build detail that matters: SM_ARCH must be 90,
not 90a, because the Makefile gates the per-file compute_90a override on
`ifeq ($(SM_ARCH),90)`.

### Why this explains the project's history

Every base run in this project stopped near 1387 steps. run100's 727M tokens cost
**~77 h on 2x RTX 3060** (log100's config records an 11909 MB card) while paying the
invisible 18x kernel tax. The same tokens are **1.0 h on one H100 at hd=128** -- about
400x better, ~80x from hardware and ~18x from the kernel. The ceiling was never
ambition.

### Why porting hd=64 was correctly abandoned

The model roadmap: 120M is hd=64; **600M / 1.7B / 4B / 7B are all hd=128**. So hd=64
exists at exactly one scale, and the fast path already serves everything above it.
Porting hd=64 (~1-1.5 days, 28 fwd injection points, the SWIZZLE_128B atom is 64 bf16
so the two-head-half warpgroup split must collapse) would accelerate only the smallest,
most transitional model.

## THE STOPPING CRITERION -- corrected after user pushback

I proposed 9537 steps / 10B tokens / 14.7 tok/param / 3.5 days, reasoning by analogy to
run100's 6.4 tok/param. The user challenged whether an NLL-based test needs that. They
were right: the test is internally controlled (all arms share the base), so the base
must be SUFFICIENT, not good. Sufficiency is two measurable properties:

1. a usable **denom** -- needle_score withholds score AND verdict below 0.5 nats
2. a **reach boundary inside the window** -- a failing region left to improve

Both come from a 4-minute probe (7 variants x 3 blocks at T=4096, no lambda extension,
so directly comparable to run100's 4k row). At every-500-step checkpoints that is 1.9%
overhead. Pretrain length became an OUTCOME.

### Probe result at step 500 (524M tokens, 0.77 tok/param)

| variant | distance | nll | significant |
|---|---|---|---|
| p0 | 4076 | 14.0787 | **no** |
| p0.25 | 3061 | 11.9778 | yes |
| p0.5 | 2046 | 9.8476 | yes |
| p0.75 | 1031 | 7.3560 | yes |
| p1 | 16 | 3.6960 | yes |

floor 3.6960, absent 14.4211, null 14.4566, **denom 10.7251 nats**

| | 600M @ 500 | run100 final | |
|---|---|---|---|
| denom | **10.73** | 8.84 | exceeds |
| reach | >=3061 | 3045 | comparable |
| boundary in window | yes | yes | usable |
| val loss | 3.805 | 3.90 | better |
| tok/param | **0.77** | 6.4 | 1/8th |

**Sufficient at one eighth of run100's tokens-per-param. No extra shards needed.** The
44-vs-100-shard and 30-vs-68-hour questions are moot: the answer was the 15 shards
already on disk.

Notable side finding: the 600M's reach (~3061) is essentially IDENTICAL to the 120M's
(3045) despite 6x the parameters and better loss -- consistent with this project's
existing "model size is not the dependency" result, and an argument that supervision
rather than scale is the lever for reach.

## Also shipped

**On-GPU per-position NLL** (`context_parallel/EvalNLLKernel.{h,cu}` + eval-path wiring,
host path retained behind `CP_EVAL_NLL_HOST=1`). Eval was CPU-bound in a host logsumexp
(GPU 0%, 124 W of 700 W; 4->8 concurrency bought 7%). Now 7.25 -> 2.33 s/window at
T=16384 = **S 3.12x**, clearing the 1.96x threshold so the full eval matrix fits
(23.5 h -> 7.5 h at 1x). Deterministic by construction -- no atomics, because for a
single batch row the scatter is a permutation; the caller loops rows, so it holds for
any B. Gates passed on both boxes: max abs delta 1e-6 nats (0.0 exactly at W=4 on
sm_90), zero count mismatches, run-twice byte-identical CSVs, B=2 verified.

**Full IN2 toolchain, gated.** `in2_gen.py` (uint16 shards, K-packed triples,
arm-independent site RNG, interleaved, shuffled pairing + decoys, alphabet capped at
512), `in2_gate.py` (23 exact assertions incl. probe==key/answer==value retrievability
and bit-identical answer_start histograms across arms), `in2_report.py` (validated
against needle_score to 4 dp), `in2_arms.sh`, and `needle_gen.py --reverse`. Arm data
for all three arms generated at T=16384: 12816 examples x 3 shards = 210.0M tokens/arm,
820,224 triples, 114,733 (14.0%) beyond R=8174.

## Mistakes made and their fixes

- `in2_arms.sh` hardcoded the laptop's lib layout -> exit 127. The same trap RESUME.md
  logged for needle_sweep.sh:64. Fixed with BLUTRAIN_ROOT auto-detect + an `ldd`
  preflight so it fails before the 1.37 GB branch copy.
- `nohup` inside a tool call dies to the harness SIGTERM -- killed a training run
  silently. All launches now use `setsid`.
- Killed my own shell TWICE with ps/pkill patterns that matched the invoking command's
  own argv. Fixed structurally: the launcher writes `.pretrain600.pid`.
- `CP_CKPT_NEW_RUN=1` is mandatory on a fresh pretrain, or the resolver takes the
  highest run WITH a checkpoint -- run100, the 120M -- and loads it into the 600M. It
  threw on the shape mismatch; do not rely on that.
- The gate rejected 12817 examples: `examples % (B*dp)` must be 0. Regenerated at 12816.
- My step-0 loss band (3.925-4.147) was wrong twice over: the filler baseline at T=1024
  is 4.285 not run100's 3.901 (cold mid-document starts), and I dropped the seam penalty
  (+0.30) the round-6 critique flagged. Usable form is the DIFFERENCE arm C minus arm A
  at the same T (+0.478 measured).
- Predicted checkpointing at freq=25 would cost 5%. Measured: 25.3 s vs 25.2 s, i.e.
  free. My `dd -oflag=direct` test measured raw device speed and missed that the write
  is buffered into 150 GB of RAM.
- Recommended porting hd=64 before seeing the roadmap; retracted once it showed hd=64
  exists only at 120M.
- Spent ~5 min of 3060 time on timing measurements the plan explicitly scoped to the
  H200/H100, then had to argue around a number (1.23x) that does not transfer because
  the bottleneck inverts between boxes.

## State at session end

- `run4` training, step ~600+, loss 3.533, val 3.805, 41,584 tok/s, 47 GB, checkpoints
  every 25 steps (~11 min of exposure). Target 1431 steps (1.50B tokens = 15 shards,
  one epoch).
- **Criterion already met at step 500.** Continuing only to complete the cosine anneal
  (LR is 4.49e-4 of the way to 6e-5); stopping mid-schedule leaves the LR un-annealed.
- Disk: 18 GB free. Six 8.18 GB checkpoints are the steady state. 29 more shards would
  fit; 100 would not.

## Next session

1. `RESUME=4 FREQ=25 ./pretrain_600m.sh` to finish to 1431 (~6 h), or stop early -- the
   criterion is already satisfied.
2. Context extension 4k -> 16k (~8 h at 1x). Needed because the arms train at T=16384
   while this base is 4k-native, and at T=4096 the region beyond R=3061 is only ~1000
   tokens -- too thin to carry a verdict.
   NOTE: cosine bakes the horizon in (get_lr:612 is warmup+cosine, NO stable phase), so
   use CP_REWARMUP / CP_REWARMUP_PEAK for the extension rather than a larger
   CP_MAX_STEPS. A WSD schedule would avoid this; bluscript_zero_wsd.cpp exists in
   BluTrain but bluscriptCP has zero references to it.
3. Cell 0 reach curve on the extended base, then arms A/B/C (~1.4 h each at 40,883
   tok/s), then the full eval matrix.
4. Readout per the pre-registered rule: conjunctive C>B AND C>=cell 0 on the per-arm gap
   `nll_null - nll` at the FIRST answer token, paired by block, bins beyond R(T),
   learnability gate checked first.
