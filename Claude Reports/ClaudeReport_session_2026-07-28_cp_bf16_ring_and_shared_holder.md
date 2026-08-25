2026-07-28 - Implemented two flag-gated, additive, reversible CP memory optimizations on the dense context-parallel framework to close the 114M max-context gap vs USP/yunchang - CP repo: context_parallel/ContextParallel.h, Scripts/Blutrain/bluscriptCP.cpp

# Session Report: CP bf16 KV ring transport + shared cross-layer forward ring holder

## Context / why
The hybrid Ring x Ulysses systems comparison (bluscriptCP vs USP/yunchang, 4x RTX6000 Ada) showed
bluscriptCP has a LOWER per-token activation cost but a HIGHER fixed per-rank ring overhead that scales
with model width. At 114M this flips the result against us:
- pure ring-4 (r4u1): max-context 13,568 tokens vs USP 40,704
- hybrid 2x2 (r2u2): 65,536 vs 72,192
(we win everywhere at 48M).

Two confirmed root causes of the fixed overhead (from reading context_parallel/):
1. KV rotated around the ring in fp32 although the fused SDPA kernel casts to bf16 internally anyway
   (FusedSDPAOp_sm89.h:66 / FusedRoPESDPA.h:96 hard-require fp32 at the boundary, pack bf16 inside) ->
   fp32 transport buys zero precision, doubles ring buffers + NCCL bytes.
2. Forward ring buffers (send_buf[2] + rotator recv_[2]) are per-ContextParallel instance, held
   x num_layers for the whole run.

## What was implemented (both default OFF, additive, reversible)
Plan went through 4 rounds of adversarial review before any code; all load-bearing assumptions were
confirmed by reading the code, and two hardening points came out of review (baked in below).

### Opt A - CP_SHARE_FWD_ROTATOR (shared cross-layer forward ring holder)
- bluscriptCP.cpp (after GPT model construction, ~line 847): read flag (value-aware, default OFF);
  when on, make_shared<ContextParallel::SharedFwdRing>() ONCE and set it on every layer via
  model.attn[i]->cp_->set_shared_fwd_ring(holder); rank-0 log. The SharedFwdRing machinery already
  existed (ContextParallel.h:285-292) but was never called anywhere -> this is pure wiring.
- HARDENING (review's Opt A fix - the cross-layer send race): the send-completion guard exch_work[2]
  was a forward_cp stack-local, so layer N+1's reuse of the shared send_buf[s] was NOT stream-ordered
  after layer N's still-in-flight send (implicit guard via the prior call's trailing drain was fragile:
  depended on OVERLAP + same FIFO compute stream). Fix: added std::shared_ptr<Work> exch_work[2] to
  SharedFwdRing, and in forward_cp use a single alias
  `std::shared_ptr<Work>* exch_work = shared_fwd_ring_ ? shared_fwd_ring_->exch_work : local_exch_work;`
  used at BOTH the guard site (:636) and the assignment site (:647) so they cannot drift apart (a
  half-fix would silently wait on a stale/null handle). Recv side needs no fix: the holder shares the
  ROTATOR OBJECT (kv_rotator = R.rotator.get(), :573/582), so recv_/work_/pack_ev_ are one set and the
  RingRotator.h:140 self-guard spans layers for free.

### Opt B - CP_RING_BF16 (bf16 KV wire transport, forward-only)
- ContextParallel.h forward_cp: read RING_BF16 once (excluded for AllGather); ring_dt = bf16 when on.
- Staging alloc: send_buf allocated with local_k.opts().with_dtype(Bfloat16).with_req_grad(false) when
  on, else the original local_k.opts() (OFF path byte-identical). Recv slots inherit send_buf opts.
- UPCAST-BEFORE-SAVE invariant (the correctness key): the received bf16 buffer is up-cast to fp32
  (as_type(Float32)) IMMEDIATELY at unpack (:625-627), before BOTH the fp32-only SDPA and the
  saved_k_chunks clone (:714-716). So saved chunks stay fp32 -> backward is UNTOUCHED and correct in
  both recompute_k modes. bf16 lives purely on the wire.
- Pack: bf16 converting copy (curr_k.as_type(Bfloat16) -> send_buf halves, element offset +k_numel,
  bfloat16_t itemsize); the original fp32 raw memcpy is the else branch, untouched.
- Net-zero precision: fused kernel casts KV to bf16 anyway; fp32->bf16 is round-to-nearest-even on both
  CPU (Types.h:90-95) and CUDA (ConversionKernels.cu:147-149) and bf16->fp32 is exact, so
  round_bf16(round_bf16(x)) == round_bf16(x) (idempotent).
- Backward, merger, output/grad unshard reshuffle, AllGather index_chunk: all untouched (off the bf16
  path - they operate on fp32 tensors or are excluded).

## Validation
- mpic++ -std=c++2a -fsyntax-only -DWITH_CUDA -DCP_FUSED_ROPE (full Makefile include set) on
  bluscriptCP.cpp (which includes ContextParallel.h): PASS, exit 0 (only an OpenCL #pragma note).
- forward_cp is a manual-backward custom Node, so the as_type casts create no autograd edges;
  Tensor::as_type is a plain forward-only cast (AsTypeTensor.cpp) - fine because backward is manual and
  reads only explicitly-saved tensors.

## Pending (runtime verification - on the box, GPUs)
1. Correctness: 4-way flag matrix {off/off, A, B, A+B} x CP_NO_OVERLAP={0,1}; CP_EVAL_PPL mean_nll vs
   off/off within ~1e-3; ~20-step loss overlay. A-cells stressed >=3x + wider width; +compute-sanitizer
   synccheck/racecheck; +forced send-path delay probe to make a missing exch_work guard fail by
   construction.
2. Memory/max-ctx win: mem_scaling_sweep_hybrid_compare.sh (bluscriptCP arm, FINE_GRAINED=1, RESUME=1)
   for 114M r4u1/r2u2, all four flag cells; expect max_T_ok to rise vs baselines (r4u1 13,568;
   r2u2 65,536).
3. Fallback proof: both flags unset -> snapshots byte-identical to current baseline dir.

## Files modified
- /home/blu-bridge25/CP/context_parallel/ContextParallel.h (Opt A.2 struct+alias; Opt B flag+alloc+pack+upcast)
- /home/blu-bridge25/CP/Scripts/Blutrain/bluscriptCP.cpp (Opt A.1 wiring + rank-0 log)
Plan file: /home/blu-bridge25/.claude/plans/make-a-detailed-plan-clever-widget.md
