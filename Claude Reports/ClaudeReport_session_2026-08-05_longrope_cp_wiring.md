# Claude Report — 2026-08-05 — LongRoPE/YaRN Context-Parallel eval+search wiring

- **When:** 2026-08-05
- **One line:** Implemented CP_SIZE>1 support for the LongRoPE/YaRN forward-only PPL paths (search + per-position CSV) in bluscriptCP, per a plan hardened over 4 review rounds, so the LongRoPE-vs-YaRN test can run at 128k/256k across 4 GPUs.
- **Workspace / files:** CP — `context_parallel/RopeDeltas.h`, `context_parallel/YARNOps.cpp`, `Scripts/Blutrain/bluscriptCP.cpp`, `Tests/bluscriptcp/longrope_search.py`, `Tests/bluscriptcp/run_longrope_pipeline.sh`; plan `.mdfiles/polished-cuddling-graham.md`.

## Why
At 48M/8x, LongRoPE ties YaRN with the temperature m (155) and loses without it (163). The LongRoPE
paper's decisive win is at extreme extension (256k, where YaRN's fixed formula breaks). Reaching
s=32/s=64 (128k/256k) exceeds one 48 GB GPU's sequence capacity, so the two forward-only PPL paths
(resident search evaluator `CP_LONGROPE_SEARCH`, per-position CSV `CP_EVAL_PPL`) — which hard-rejected
CP_SIZE>1 — had to be wired for Context Parallelism across 4 GPUs.

## What was already there (verified, reused)
The fused CP attention kernel already reconstructs each token's GLOBAL RoPE position from its local
shard index (`RopeDeltas.h` 4-delta interface, `GQA_fused_fwd_sm103_cp.cu`, tested by
`Tests/cp_rope_deltas_test.cpp`). `GPT::forward` shards the full `[B,T]` input internally and returns
local `[B,T/cp_size,V]` logits. So LongRoPE fine-tune under CP is essentially free; only the eval/search
reduction was missing.

## Changes implemented
1. `RopeDeltas.h` — `local_to_global_pos(r,i,Tl,N,lb)`: single source of truth for the CP eval
   position map (contiguous `r*Tl+i`; HeadTail `cs=Tl/2; i<cs ? r*cs+i : (2N-1-r)*cs+(i-cs)`).
2. `YARNOps.cpp` — `YARN_NO_MSCALE=1` gates the m=0.1*ln(s)+1 temperature (default keeps m), enabling
   a YaRN-no-m arm for the 2x2.
3. `bluscriptCP.cpp`:
   - Early pure-ulysses guard: die if cp_size>1 and q_heads or kv_heads not divisible by cp_size.
   - Search evaluator: require world==cp && dp==1; rank-0 reads stdin + parses, then MPI_Bcast of the
     parsed genome (ok flag + lambda + n_hat + s) so all ranks act in lockstep (no NCCL hang on a
     parse error); fitness = GPU `sparse_cross_entropy_loss` per rank, MPI_Allreduce mean/world (no
     host logits copy, no manual logsumexp).
   - CP_EVAL_PPL CSV: require world==cp && dp==1, die on hybrid; disambiguate T_global (window/CSV)
     vs T_local (logits); MANDATORY chunked device->host copy over local positions (bounds host RAM
     to CHUNK*V, not T_local*V); map each local position to global via `local_to_global_pos`;
     MPI_Reduce sum_nll/cnt to master which writes the full [0,T) CSV; `CP_EVAL_SKIP_WINDOWS` held-out
     support on BOTH the val_loader path and the CP_EVAL_TOKENS_BIN bypass (base=(skip+w)*T_global,
     clamp subtracts skip).
4. `longrope_search.py` — launch `mpirun -np <cp_size>` (from CP_SIZE in --arch); candidate-#1 budget
   precondition (abort if fewer than one full generation ~ POP+2(N1+N2) would fit the wall-clock).
5. `run_longrope_pipeline.sh` — multi-GPU (`mpirun -np NP`, GPUS comma list, wait-all-CP-GPUs >=20GB);
   overridable CP_REWARMUP and GPUS (CP_REWARMUP was a hardcoded literal); real `ARM={longrope|yarn}`
   switch (yarn skips search, sets YARN_SCALE/ORIG_MAXPOS, m-variant via YARN_NO_MSCALE, arm+m-variant
   tagged into OUT and BEST so cells never overwrite); token-matched GBATCH/FT_STEPS/CP_REWARMUP; eval
   skip=CALIB (held-out, both arms score the same windows).

## Design decisions forced by review (4 rounds of `.mdfiles/plancritique.md`)
- attn_mode must be `ring` for CP=4 (ulysses can't split 2 kv heads 4 ways) -> HeadTail is the live map.
- Search fitness on GPU CE, not host logsumexp (infeasible at 256k: ~2.6e10 exp + 13.2 GB host copy).
- Held-out eval (skip = search's CALIB) to avoid train-on-test; must cover the bypass branch too.
- Token-invariant matching across factors (s=64 halves FT_STEPS to 150 AND CP_REWARMUP to 50).
- 2x2 phased: headline = LongRoPE-no-m vs YaRN-with-m (as published); no-m/no-m is the mechanism control.
- Pre-registered criterion vs the base 4k-run TAIL PPL ([2048,4096)), NOT the unigram floor.

## Status / not done
Code complete, NOT yet built or run. Remaining (server, needs GPUs + git):
- Build after pull; then parity FIRST (first-ever 4-rank ring run — expect bug shakeout): CP=1/2/4
  scalar + CSV parity on run 30 (HeadTail), contiguous CP=2, off-mode byte-identical, both-paths CP=1
  reconcile, swap-takes-effect, malformed-candidate no-hang.
- YaRN-cache dump pre-flight (CP_DUMP_ROPE) at s=32/s=64; measure base 4k-tail reference; fill criterion.
- mem-probe s=32 AND s=64 (CP shards T not V, so 256k logits ~13.2 GB/rank may not fit even on 4 GPUs;
  s=64 falls back to COMPOSE=none forward-only if training OOMs).
- Science Phase 1 (3 cells/factor) at s=32, then s=64 if it fits.

## Honest caveat
The result may be inconclusive at 48M if both arms saturate near the unigram-entropy floor; the
pre-registered 4k-tail criterion distinguishes "LongRoPE won" from "the model is too small." If the
latter, the next step is a larger base model, not more search.
