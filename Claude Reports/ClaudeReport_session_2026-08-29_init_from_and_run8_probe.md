# CP_INIT_FROM, checkpoint-branching fixes, and the run8 reach probe

2026-08-29 - Built the weights-only checkpoint-branching path (CP_INIT_FROM),
fixed three defects found by testing it, and measured the 1.06B proxy's reach.
Boxes: metallic-bird-thinks-fin-03 (4x H200, torn down) then
light-sun-speaks-fin-02 (1x H200). Workspace: BluTrain / dist/Context_Parallelism.

## The 1.06B proxy run (run8) completed

`d1536 / L36 / Q12 / KV4 / hd128 / FFN4096 / tying OFF` = 1.0606B params
(Muon 905,969,664 over 216 matrices + AdamW groups 77,266,944 embedding and
77,388,288 lm_head+norms). Muon + AdamW, ZeRO-1 v2 + symmetric, bf16, WSD
100/700/200, all groups peaked at 1e-3.

| step | val_loss |
|---|---|
| 0 | 11.095013 |
| 250 | 3.940885 |
| 500 | 3.156706 |
| 750 | 2.988807 |
| 999 | **2.817160** |

1000 steps in 1h39m at mean 176,067 tok/s, peak 95,437 MB, B=4 on 4x H200.
Train loss 8.719 (steps 0-24) -> 2.625 (950-974), monotone throughout.

**B=4 beat B=2 by 4.7%** (176,600 vs 168,750 tok/s) - the opposite of what a
3-step sweep said (162k vs 165k). Three steps is too short: the allocator is
still growing its pool and the larger micro-batch has not reached steady state,
which biases *against* the larger B. Measure throughput over tens of steps.

**LR must be equal for Muon and AdamW.** CP defaulted max_lr=6e-4 with
muon_lr=1e-3, so the 906M body took 1.67x larger steps than the embeddings.
MATCH_RMS_ADAMW exists precisely to put them on one scale - ShardedMuon.h says
"setting both equal gives both groups the same step size and the ONLY difference
is the update direction" - and bluscript_zero sets both to 1e-3. Fixed with
CP_MAX_LR=1e-3 CP_MIN_LR=1e-4; the first 507-step run was discarded.

## CP_INIT_FROM: branching a new run from a trained base

The need: a long-context extension starts from base WEIGHTS but must not inherit
the optimizer state, which is ZeRO-sharded to the world size that wrote it. A
normal resume therefore cannot cross a GPU-count change, an optimizer change
(ZeRO -> no ZeRO), or a dp<->cp reshard.

**Resume and initialize-from are NOT the same operation.** Resume restores every
piece of state together, so ordering does not matter - `load_state` overwrites the
optimizer's fp32 masters after the fact. Initialize-from has no saved optimizer
state, so the masters must be DERIVED from the loaded weights, which means the
weights must land BEFORE any optimizer is constructed.

The first attempt (`CP_CKPT_WEIGHTS_ONLY`) loaded at the resume site, line 2141,
long after both optimizer construction sites (ZeRO at 1775, nn::AdamW at 1824).
Every one of them snapshots fp32 masters at construction, so the masters held the
random init and the first step() published them back over the loaded weights:

    step 1000  loss  2.481   <- weights loaded correctly
    step 1001  loss 11.084   <- masters clobbered them

Failing for exactly one step is worse than failing outright. `CP_INIT_FROM` now
loads right after `model.parameters()` (line 1612), before both optimizers, and
`CP_CKPT_WEIGHTS_ONLY` is retired with an error pointing at it.

**Why this is the universal mechanism:** only replicated, T-independent PARAMETERS
move. `cos_sin_cache` is a plain member, never `register_parameter`'d, so changing
T or YARN_SCALE does not change the parameter count. Verified across both
optimizer paths - ZeRO-1+Muon and plain AdamW both start at loss 2.753656,
identical to 6 decimals, and stay in the 2.7-3.0 band instead of jumping to 11.

## Three defects found by testing, not by reading

1. **`CP_CKPT_DIR` did not exist.** I wrote a comment telling the user to set it
   without implementing the getenv, so the loader always read the default
   directory, found nothing, and trained from random init. Silently.
2. **No abort on "checkpoint not found".** A weights-only load that finds nothing
   is always a mistake; it now dies. Same for `CP_CKPT=0`, where the resume block
   is skipped entirely and the load would never happen.
3. **`copy_: dtype mismatch`.** The weights-only loader used `p.copy_(loaded)`
   with no conversion. run8 stores bf16 (CP_BF16=1); the reach probe runs fp32.
   Crossing precision is exactly what this path exists for, so it now converts
   via `as_type` when dtypes differ. The ordinary `load_checkpoint` still has the
   limitation, but a resume always matches training precision.

## run8 reach probe - the IN2 sufficiency gate

`probe_1b.sh` (new; the 600M one is hardcoded to L24/tying=1 with stale
/root/BluTrain paths). `needle_sweep.sh` gained `INIT_FROM=<path>` because it
hardcoded `CP_CKPT_RESUME`, and a forward-only eval of a ZeRO/Muon base builds a
plain nn::AdamW that would try to parse ShardedMuon's state.

    floor(p1)=1.5181  absent=13.8241  null=12.7732  denom=12.3060 nats

| variant | nll | se | score | sig |
|---|---|---|---|---|
| p0 | 11.9470 | 0.2510 | 0.1525 | YES |
| p0.25 | 7.6774 | 0.5615 | 0.4995 | YES |
| p0.5 | 5.2811 | 0.4232 | 0.6942 | YES |
| p0.75 | 3.3513 | 0.6088 | 0.8510 | YES |
| p1 | 1.5181 | 0.0202 | 1.0000 | YES |

**Both gates pass.** denom 12.31 nats against the 0.5-nat withholding bar, and
better than run100 (8.84) and the 600M (10.73). The reach boundary is inside the
window: p0 scores only 0.1525, so a large failing region remains at distance -
which is the room IN2 supervision is meant to improve. All five points
significant at BLOCKS=3, curve monotone.

## Still open

1. **The plan's R=8174 is run100's, not run8's.** The primary claim region ("bins
   beyond R(T)") must be re-derived for this base, at T=16384, after the
   extension. The ~13% arm-C fraction is anchored to the old number.
2. **run8 is 4k-native; IN2 trains at T=16384.** The YARN/LongRoPE extension has
   to happen first - that is what CP_INIT_FROM was built for and it is the next
   step.
3. **in2_data is gone** with the earlier instance; regenerate with in2_gen.py
   (numpy is only in the venv, not system python).
4. The plan's budget assumed 4x H200.

## Environment traps recorded

- `hf` is in a venv, NOT on PATH: `/root/Adhi/BluTrain/envv/bin/hf`.
- `pgrep -f <pattern>` matches the searching shell's own argv when the pattern
  appears in the command line - use `pgrep -f 'make[ ]libtensor'`. This cost a
  wasted wait loop and a chained builder that spun forever.
- Downloading FLUX: `hf download navingv/flux --repo-type=dataset` (20.2 GB) -
  it is a separate public dataset, not in the BluScriptCP repo.
