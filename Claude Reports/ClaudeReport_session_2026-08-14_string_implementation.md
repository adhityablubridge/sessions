# ClaudeReport_session_2026-08-14_string_implementation

2026-08-14 - STRING (Shifted RoPE, arXiv:2410.18745) implemented in bluscriptCP and verified end to end - Workspace: CP - Files: context_parallel/StringRoPE.h, context_parallel/ContextParallel.h, Scripts/Blutrain/bluscriptCP.cpp, Tests/cp_string_reference.cpp, Tests/bluscriptcp/needle_gen.py, Makefile

## What was built

- **`context_parallel/StringRoPE.h`** (new) - block-decomposed STRING forward. Per query block: a causal diagonal call, a full near-band call, and a full far call whose Q cache rows are shifted by `W - S`; merged with the existing `SDPAMerger`. `StringConfig` / `string_plan()` hold the knobs and all validation, so the startup banner and the driver cannot disagree.
- **`ContextParallel.h`** - dispatch at the `sdpa_fused_forward_rope` call site in the ring body, plus an env-gated `CP_STRING_SELFCHECK` that recomputes the baseline on the same buffers and reports `max|dout|` / `max|dlse|`.
- **`bluscriptCP.cpp`** - gates (`CP_SIZE==1`, LongRoPE `n_hat==0`, no-op refusal, and the new ring-mode requirement) and a banner echoing `S/B/W -> G/k/shift` and the fuzzy threshold band.
- **`Tests/cp_string_reference.cpp`** + `make cp-string-ref` - independent host fp64 reference for the shift semantics.
- **`needle_gen.py --p-fracs`** - override for the `S=1.5R` ablation whose predicted dead gap falls between the default grid points. Default output unchanged.

## Two real bugs found, both by controls rather than by the tests themselves

**1. The dispatch never executed.** Tests 2 and 3 had "passed" bit-identically. A control that set `shift = -3968` - a configuration that MUST change the output - changed 0 of 8192 positions. Cause: the eval defaults to `CP_ATTN_MODE=ulysses`, which routes through `forward_ulysses_fused` -> `gqa_fused_flash_attn_forward`, while the STRING branch had been added to `forward_cp`'s ring body. The earlier passes were vacuous: they measured baseline against baseline.

The ulysses path cannot host STRING - `gqa_fused_flash_attn_forward` takes a *packed* QKV tensor at one shared `T` and a single scalar `pos_offset`, so it cannot express "Qi against K[0:near_lo] with Q's rows shifted". Only `sdpa_fused_forward_rope` exposes separate Q/K tensors with independent position deltas -- and the single shared `pos_offset` is the deeper obstacle: RoPE is relative, so shifting Q and K by the same amount leaves `d = m - n` unchanged, making a shared offset mathematically incapable of remapping relative distance.

`string_attention` could in principle have been called from `forward_ulysses_fused` (it uses `sdpa_fused_forward_rope` internally regardless of caller), but then STRING-on would run on one kernel and STRING-off on another, confounding STRING with a kernel swap. On ring both arms use the same kernel. Hence a hard gate: `CP_STRING=1` requires `CP_ATTN_MODE=ring`, and it dies otherwise rather than silently reporting baseline numbers labelled STRING.

**Consequence for the experiment: none, verified.** The concern was that the existing reach curves were measured on ulysses and so could not serve as the off-arm. Measured directly: at `CP_SIZE=1` a `CP_EVAL_PPL` run on ring is **byte-identical** to the stored ulysses CSV on the real model (GQA 12/4, T=16384) — and `cp_rope_fused_parity.cpp` case A already asserts the two kernels agree bit-for-bit at deltas=0, forward and backward. At one rank there is no all-to-all and no ring rotation, so the two modes are different code paths computing the same plain causal attention. The stored reach values `R` are therefore valid inputs to the S-selection rule, and `run1`'s pending curve is not path-dependent either.

**2. `Tensor::narrow` on a non-contiguous input.** With the dispatch live, the `W=S` identity failed by 6.77 nats. `Tensor::narrow` (ParallellismUtils.cpp:161) does not consult the source's stride vector - it recomputes row-major strides from the *shape* and copies from `data()`, so on a non-contiguous input it silently returns a wrong tensor of the right shape. The ring's `local_q` *is* non-contiguous (K and V are not), which is why the baseline was unaffected: `sdpa_fused_forward_rope` calls `.contiguous()` on its own inputs, but that happens after our narrow. Fixed by materialising Q/K/V once before the block loop.

The diagnostic that localised it: block 0 - the only block making a single call - failed, and position `t=0` matched while `t=1` onward diverged. A single key gives softmax=1 regardless of the score, so only offset 0 survived a wrong stride walk.

## Verification results

| test | result |
|---|---|
| 1. off-mode byte-identity | **PASS** - byte-identical to a CSV produced 2026-08-12, before STRING existed |
| 2/3. block-split identity, `G=1` | **PASS** - `max\|dNLL\|` exactly 0.000e+00 |
| 3. `W=S` identity at `G=8` | **PASS** to bf16 noise: mean 1.6e-3, max 2.1e-2 nats |
| 4. shift semantics vs host fp64 reference | **PASS** - cos 0.9999942 at shift -256 and -448 |
| control: live shift must change output | **PASS** - 7168/8192 positions move, `max\|dNLL\|` 0.663 |

**The `G>1` residual is numerics, not a bug.** Block 0 (no merge) is exactly 0; blocks 1-7 sit flat at ~1.8e-3 with no growth as more chunks merge; there is no block-boundary concentration (0.86x, if anything lower at boundaries); and the sign is unbiased (42.8% positive, signed mean 3.7% of mean-abs). An indexing bug would be biased, boundary-concentrated and position-growing.

**Pre-register this noise floor:** STRING-on runs carry mean 1.6e-3 / max 2.1e-2 nats of bf16 accumulation noise relative to an exact baseline. Against the scorer's `MIN_EFFECT_NATS = 0.1`, the max is 21% of the minimum effect size at a single position. Effects at or below ~0.02 nats per position are not resolvable on this path.

## Test 4 needed rebuilding, and nearly passed vacuously

The planned reference (autograd `rms_norm` + `rope` + `matmul` + `tril` + `softmax`, the pattern `cp_rope_fused_parity.cpp` uses at T=64) does not survive at T=1024: `softmax` over a tril-with-`-inf` tensor returns whole NaN rows, and identical seeds disagree between runs. A reference that cannot reproduce plain attention cannot adjudicate anything, so it was rewritten as self-contained host fp64 code sharing nothing with the implementation or the op library - with a **mandatory sanity gate** that the reference reproduce the kernel on plain causal attention before any STRING verdict is admitted (it does: cos 0.9999941).

It then nearly passed vacuously. With `randn(std=0.1)` gammas the post-norm activations are ~0.1 and scores ~0.005, so softmax is effectively uniform - and under a uniform softmax *no* change of query rotation can move the output. The separation check caught it. With unit gammas (what real RMSNorm weights look like) the shift separates STRING from plain by maxdiff 0.105 / 0.088, twenty times the 5.3e-3 bf16 floor, while the `shift=0` case collapses to exactly the floor.

## Standing caveat

`CP_STRING` is CP_SIZE=1, forward-only, and requires `CP_ATTN_MODE=ring`. Composing the block decomposition with ring/ulysses sharding is a separate project.

## Next

1. `run100` STRING cells are unblocked NOW - `R` is already measured (3,045 / 4,078 / 8,174 / 8,183 at 4k/8k/16k/32k), so `S` is determined. Note the on/off pair must still be re-run at `--blocks 5`: the existing curves are `blocks=3`, and the success criterion is a 2-s.e. test. That is a statistical-power requirement, NOT a path requirement - the blocks=3 curves remain valid for setting `S`.
2. Server: the `run1` baseline reach curve - `run1` is the primary target (unadapted 4k base) and has no reach curve at all.
