2026-06-29 - 18:55 - Analyzed the team's fused attention kernel for context-parallelism compatibility and wrote a requirements document for the kernel team - Workspace: CP / File: .mdfiles/cp_attention_kernel_requirements.md

# Session Report: CP Attention-Kernel Requirements

## Goal
The team builds a Llama-style training script (`BluTrain/bluscript.cpp`) on an in-house DL
framework (`BluTrain/Tensor-Implementations`), with a fused attention kernel (QK-norm + RoPE +
grouped-query causal attention) reached via `autograd::scaled_dot_product_attention(...)`. I am
adding context parallelism (ring attention) on top. The ask: determine what the team must keep in
mind while designing/finishing the attention kernels so they are CP-compatible (e.g. q_offset/k_offset
position indices, sin/cos rotary handling), tracing both the current naive kernel and my existing CP
implementation.

## What I did
- Traced the production fused kernel via two parallel read-only agents:
  - Team kernel: `GQA_fused_fwd_sm103.cu` / `GQA_fused_bwd_sm103.cu`, `AttentionOps.cpp`, `RopeOps.cpp`,
    `AttentionBackward.h`.
  - My CP machinery: `context_parallel/ContextParallel.h`, `ContextParallelBackward.h`, `SDPAMerger.h`,
    `RingRotator.h`, `LoadBalancer.h`, `AttentionForward_sm89.{h,cu}`, `AttentionBackward_sm89.{h,cu}`.
- Synthesized into `/home/blu-bridge25/CP/.mdfiles/cp_attention_kernel_requirements.md`.

## Key findings
Production kernel is ~70% CP-ready: it already does online softmax, emits LSE `[B,Nq,T]`, uses
absolute indices in the causal mask, and has a `pos_offset` param. Gaps for CP:
1. Single `pos_offset` (feeds RoPE only) must become independent `q_offset` + `k_offset` feeding the
   **causal mask** (`k_offset+k > q_offset+q`).
2. Square `T_q==T_k` assumption must be relaxed (load-balanced ring uses asymmetric Q vs KV lengths).
3. LSE must be a returned forward output (currently hidden in the backward node) for the cross-block merge.
4. Backward must be a pure function of supplied `(Q,K,V,O,dO,LSE,offsets,T_q,T_k,is_causal)` — not its
   own saved LSE — because CP feeds the merged LSE per ring step.
5. RoPE-fused-in-attention is the biggest friction: Q and K need different positions per ring step;
   re-rotating shipped K wastes work and entangles dk_gamma; HeadTail load-balanced shards are
   non-contiguous in global position (single scalar offset cannot express them). Recommendation:
   expose a de-fused path (apply QK-norm + RoPE with per-token global positions before attention; ship
   pre-rotated K around the ring; kernel does pure SDPA) — which is exactly what my sm89 CP kernel does.
6. Keep the primitive GQA-aware and ship K/V grouped through the ring (3x less comm).

## Deliverable
`CP/.mdfiles/cp_attention_kernel_requirements.md` — includes side-by-side trace tables, the proposed
CP-ready forward/backward API, the RoPE de-fuse vs dual-offset decision, a checklist, and a source map.

## Notes / follow-ups
- My current CP sm89 kernel takes a single `nh` (not GQA-aware) — joint design item: make the CP
  attention primitive GQA-aware so KV stays grouped in the ring.
- For CP parity testing, an fp32 math toggle on the attention kernel is recommended (prior logs show
  TF32 placement caused C++/PyTorch drift).
- No code was changed this session; analysis + documentation only. No git operations performed.
