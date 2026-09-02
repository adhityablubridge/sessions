# Session report - 2026-09-02 - CP_GENERATE decoding + RULER/NIAH/VAL generative eval harness

Workspace: CP (/home/blu-bridge25/CP)
Hardware: local dev box only (2x RTX 3060) - NOTHING in this session was GPU-tested
Files: Scripts/Blutrain/bluscriptCP.cpp, Tests/bluscriptcp/longeval_gen.py,
       Tests/bluscriptcp/longeval_score.py, run_longeval.sh, fetch_eval_assets.sh

## 1. Why: the existing probe is an NLL proxy, not a benchmark

needle_score.py measures teacher-forced NLL over an answer span that is ALREADY in
the input. RULER / NIAH / VAL all score what the model GENERATES. The proxy is more
sensitive (continuous nats vs binary hit) but it cannot produce numbers comparable to
published results, and it can pass where real generation fails - teacher forcing hides
error compounding, and at p1 a local copy circuit can produce low NLL without any
long-range lookup. Hence a real decode path.

## 2. Generation IS feasible in bluscriptCP - three checks before writing code

The loop at gpt2_cp_test.cpp:1237-1346 ports over, but that model is GPT-2 style with
wpe added before attention, whereas bluscriptCP applies RoPE INSIDE ContextParallel.
Three things had to be verified rather than assumed:

  a) Does pre_sharded=false work in the ring at all?
     YES. ContextParallel.h:454-459 explicitly anticipates it: "Generation paths pass
     variable, often-small T (e.g. T=10 during incremental token generation) that may
     not satisfy this. When the internal kernel would be invoked (pre_sharded=false)
     and divisibility fails, fall back to a non-LB path for this call."

  b) Does RoPE stay correct when CP shards internally instead of the caller?
     YES. RopeDeltas.h derives global positions from (rank, ring step, world_size,
     T_local) - pure CPU metadata, no dependence on the caller's HeadTail perm. Its
     documented degenerate case "contiguous CP -> (r*Tl,r*Tl)/(s*Tl,s*Tl)" is exactly
     the pre_sharded=false path. So RoPE is right whether CP shards contiguously
     (short T, lb_div_ok false) or HeadTail (long T).

  c) Does the Hopper split kernel accept a growing sequence?
     NO. GQA_causal_cp_fwd_sm90.cu:579 guards hd==128 && T_q%256==0 && T_k%128==0 and
     THROWS otherwise (deliberately - the comment notes a zero-unit grid would leave
     O/LSE unwritten). SOLVED by padding each decode step up to a multiple of 256.
     Trailing pad cannot influence position cur-1 under causal masking, and cur-1 is
     the only logit read. 256 is also a multiple of 2*world_size for every world size
     we run, so it satisfies forward_cp's lb_div_ok at the same time.

## 3. The patch (3 edits, additive; training and CP_EVAL_PPL untouched)

  CausalGQA  : + generation_mode_ / set_generation_mode(); forward_cp called with
               unshard=generation_mode_, pre_sharded=!generation_mode_
  GPT        : + set_generation_mode() fanning out to blocks; forward() skips the
               pre-embed shard while generating and reads idx's ACTUAL length rather
               than cfg.T (the sharded path hard-codes cfg.T because training windows
               are always exactly T - that is why decoding needed its own branch)
  main()     : + CP_GENERATE block. Greedy by default (benchmarks must be
               reproducible; CP_GEN_GREEDY=0 opts into top-k). Rank 0 decides the
               token and MPI_Bcasts it - deciding on one rank is what GUARANTEES the
               ranks stay in sync, rather than relying on their logits being bitwise
               equal. Aborts up front if pad_up(prompt+max_new) > CP_T, since CP_T is
               the RoPE cache length.

Brace balance verified 282/282. NOT COMPILED - no CUDA/MPI on this box.

## 4. Harness

  longeval_gen.py    6 tasks: niah_single, niah_multikey, niah_multivalue (RULER NIAH
                     family), vt (RULER variable tracking), val_forward, val_backward
                     (VAL's forward/backward retrieval patterns). Writes an int32
                     prompt stream at EXACT stride + a meta.json of expected answers.
  longeval_score.py  substring match (RULER's recall criterion) + per-depth breakdown,
                     which is the lost-in-the-middle axis.
  run_longeval.sh    fans (checkpoint x task) jobs across GPUs, resumable via .done.
  fetch_eval_assets.sh  checkpoints + flux haystack + tiktoken.

PROMPT FORMAT MATTERS: the target is a ~1B BASE model with no instruction tuning, so
every task ends with an answer PREFIX ("Answer: The special magic number for yarrow
is") making the correct answer the natural continuation. A bare Question:/Answer:
would fail for reasons unrelated to retrieval.

Tested end to end in Python: all 6 tasks emit exact-stride bins, decoded prompts
confirmed to contain their needle, scorer returns 3/5 with the correct depth split on
a synthetic generation and warns about missing rows.

## 5. Cost - the defaults had to change

A forward at T=16384 fp32 measures 17.8 s on H200 (derived from probe_arm_c: 21 jobs
x 8 windows in 12.5 min at slots=4), ~25 s on H100. Naive re-forward decoding costs
ONE forward per generated token:

    fp32, MAX_NEW=32, N=50   190.4 GPU-h    <- my first draft. impractical.
    fp32, MAX_NEW=16, N=20    38.1 GPU-h
    bf16, MAX_NEW=16, N=20     9.5 GPU-h    <- shipped default
    bf16, MAX_NEW=16, N=10     4.8 GPU-h

bf16 is on for generation because these tasks are scored by SUBSTRING MATCH, where
bf16 rounding cannot move a score the way it would move an NLL. The fp32 CP_EVAL_PPL
probes are deliberately left alone.

GPU sizing: memory needs ~24 GB at T=16384 (params 4.2 + optimizer state 12.7, which
bluscriptCP.cpp:1362 allocates even in eval, + logits 3.3 + activations), so ONE H100
80GB suffices and H200 143GB is idle capacity. Time is the binding constraint, and the
24 jobs (4 ckpts x 6 tasks) are embarrassingly parallel: 4 GPUs -> ~2.4 h, 8 -> ~1.2 h.
More than 8 buys little at N=20.

## 6. Not done / next

  - NOTHING GPU-TESTED. Run the smoke test first:
      TASKS=niah_single CKPTS=arm_c N=2 MAX_NEW=8 PROMPT_LEN=2048 T=4096 GPUS=0 \
        ./run_longeval.sh
  - RULER CWE/FWE (long list outputs, different scoring) not implemented.
  - VAL's code and structured-data contexts not implemented; only document-style.
  - Still open from 2026-09-01: arm B at BLOCKS=8, the control for the 6.87 sigma
    arm C distance anomaly (retrieval better at 12,252 tokens than at 8,176).
