# Claude Report — LongRoPE implementation in bluscriptCP

- **Date - time:** 2026-07-31 16:54
- **One line:** Implemented LongRoPE (searched non-uniform RoPE cache, arXiv:2402.13753) as an additive context-extension arm in the CP Llama trainer, with an evolutionary GA driver.
- **Workspace / files:** CP / context_parallel/{LongRoPEOps.h, YARNOps.cpp}, Scripts/Blutrain/bluscriptCP.cpp, Tests/cp_rope_longrope_parity.cpp, Tests/bluscriptcp/longrope_search.py, Makefile, README.md

---

## Goal

Add LongRoPE — the SOTA cache-only context-extension method that **searches** for per-dimension
RoPE rescale factors `λ_i` + an initial-token threshold `n̂` (instead of YaRN's fixed formula) — as
another arm on the existing PPL-vs-position rig, comparable to YaRN and CREAM. Fully additive
(gated behind `CP_LONGROPE_FACTORS`; unset ⇒ byte-identical). User chose full progressive scope +
overall-PPL fitness. Plan hardened through two review rounds.

## What was built

**Phase 0 — cache builder (verified).** `build_rope_cache_longrope` in `YARNOps.cpp` (declared in
new `context_parallel/LongRoPEOps.h`): `angle(n,i) = n·base_freq_i / (n<n̂ ? 1 : λ_i)`, NeoX
half-split layout identical to the YaRN builder, **no** attention-temperature `m`, with monotone /
`≥1` / size guards. `Tests/cp_rope_longrope_parity.cpp` (CPU value checks) — `λ=1,n̂=0` == plain
RoPE, `λ=s` == `pos·base_freq/s`, `n̂` threshold straddle, guards throw — **ALL PASS** (maxdiff
~1e-7) via `make CP_FUSED_ROPE=1 cp-rope-longrope`.

**Phase 1 — wiring (additive, gated).** `parse_longrope_factors` reads an `n_hat/s/S_search/lambda`
file. LongRoPE-alone installs the static cache at `CP_T` on all layers; composed-with-CREAM builds
`cream_full_cache_` via the LongRoPE builder. **Critical coupling asserts** (`S_search ==
cream_scaled_max`, `s == cream_factor`) reuse CREAM's already-hardened `factor = YARN_SCALE` /
`scaled_max = factor·T` invariant — closing the "two values must agree" bug the review flagged.

**Phase 2 — search.** Resident evaluator `CP_LONGROPE_SEARCH=1`: loads the model once, then reads
candidate factor-file paths on stdin, rebuilds only the cheap CPU cache per candidate, and prints
overall PPL — avoiding ~2560 model reloads per search. `Tests/bluscriptcp/longrope_search.py` is the
GA driver: grid `[1, s·1.25]` step 0.01, `n̂` set, `P=64/N₁=16/N₂=16/40 iters/top-32/p=0.3`, seeds
PI/NTK/YaRN (ported from `YARNOps.cpp` `find_correction_range`), mutation = per-dim **resample** at
0.3, crossover, `repair_monotone` = **running-max (never sort)**, memoized on grid indices.

## Verification status

- **Build:** clean, `CP_FUSED_ROPE=1`, exit 0 for both `bluscript-cp` and the parity test.
- **CPU parity:** green (the core cache-math correctness).
- **GPU runtime tests PENDING** — off-mode training regression, swap-takes-effect (3b),
  coupling-assert-with-model, and the search smoke all need a GPU, but **both RTX 3060s are
  saturated (11.6/12 GB, 100%) by the user's running jobs**. Commands are ready to run once a GPU
  frees up (see README + plan). No jobs were killed.

## Commands to finish verification (when a GPU is free)

```bash
# off-mode regression + produce a tiny ckpt
env CP_SIZE=1 CP_T=256 CP_N_EMBD=256 CP_N_LAYER=2 CP_N_HEAD=4 CP_N_KVHEAD=2 CP_FFN=1024 \
    CP_WEIGHT_TYING=0 CP_B=2 CP_GLOBAL_BATCH=512 CP_MAX_STEPS=30 CP_WARMUP=5 CP_CKPT=1 \
    CP_CKPT_FREQ=30 CP_DATA_ROOT=$HOME/Downloads CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=BluTrain/Tensor-Implementations/lib:BluTrain/Profiler/lib:$LD_LIBRARY_PATH \
    mpirun -np 1 ./build/bluscriptCP_exec
# swap-takes-effect (3b): feed a near-identity and an aggressive candidate; PPL must differ
# search smoke: python3 Tests/bluscriptcp/longrope_search.py ... --pop 6 --iters 3 --calib-windows 2
```

## Deferred / notes

- Progressive multi-stage fine-tune (Stages A/B) is full-length training → single-GPU memory
  ceiling; recommended path on the 3060 is composing LongRoPE with CREAM (4k physical fine-tune,
  LongRoPE cache as gather source).
- `.mdfiles/LongRoPEexplanasion.md` holds the algorithm + pseudocode; could be synced to the
  as-built resident-evaluator/λ_NTK details.

## Plan

`/home/blu-bridge25/.claude/plans/polished-cuddling-graham.md` (approved after two review rounds —
algorithm verified against the paper; all integration-layer traps closed).
