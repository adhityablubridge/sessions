2026-07-23 - 18:09 - Long-context extension benchmark: built the harness comparing bluscriptCP YaRN vs LlamaFactory frontier RoPE-scaling methods - LlamaFactory / scripts/convert_ckpt/{ckpt2hf.py,rope_parity.py}, scripts/{gen_ext_yamls.py,eval/ppl_vs_pos.py,README_longctx_bench.md}, examples/train_full/ext/*.yaml

# Session Report: Long-context extension benchmark (bluscriptCP vs LlamaFactory)

## Goal
Compare the user's dense context-parallel long-context extension framework (bluscriptCP.cpp, default
method = YaRN) against frontier long-context extension methods, using LlamaFactory as the reference
harness. LlamaFactory provides the canonical HuggingFace transformers implementations of four dense
RoPE-scaling methods (yarn, linear/PI, dynamic/NTK, llama3) via `rope_scaling` (v0 only; v1 has none).

## Approved plan (2 tracks, same 4k base)
- Track 1 (quality): long-context PPL of CP-YaRN vs LF {yarn, linear, dynamic, llama3} across scaling
  factors {2,4,8}.
- Track 2 (systems): tok/sec, peak mem, max-ctx-before-OOM across DDP x Ring x Ulysses x ws {2,4,8},
  CP vs LF v1 Ulysses.
- Plan file: `~/.claude/plans/make-a-detailed-plan-clever-widget.md` (went through 2 external review
  rounds via plsnfix.md; all findings folded in).

## Key decisions (locked with user)
- Canonical base = CP-trained 4k `.ckpt`, converted once to HF to seed LF arms. Base lives on the
  6000 Ada server box and is kept as a FILL-IN variable; the user runs the whole pipeline on the box.
- Single evaluator = HF/Python PPL loop (weights lossless-repacked; fairness lives in eval-time rope,
  gated by a rope-parity check).
- LF methods = yarn + linear + dynamic + llama3. Factors 2/4/8. Both tracks. Up to 8 GPUs.
- LF extension arms run in v0 (`stage: pt`, `finetuning_type: full`, `rope_scaling`) because v1 has no
  rope_scaling plumbing.

## Verified facts (from CP source, not assumed)
- `.ckpt` = positional binary, no string keys. Header: `CKPT` | ver:i32 | step:i32 | loss:f32 |
  count:i32 | count tensor records | AdamW optimizer state | RNG state. Per tensor: `TNS1` | dtype:i32
  (Float32=10) | rank:i32 | shape:i64[rank] | raw row-major bytes. (`Serialization.cpp`,
  `Checkpointing.h`.)
- Param order = `NN.cpp::parameters()` emits own params first, then children in registration order.
  Per layer: q_gamma, k_gamma, attn.norm, wQ, wK, wV, wO, mlp.norm, gate_up, down; then wte, norm_f,
  lm_head (omitted iff tied).
- Linear weight stored [in,out] (forward x@W) = transpose of HF [out,in]. Embedding [vocab,d_model] no
  transpose. gate_up fused [d_model, 2*ffn] (gate=cols[:ffn], up=cols[ffn:]).
- head_dim = 32 (the cpp comment `384/12 // =64` is wrong; matches LF config q_proj [192,384]).
- CP YaRN (YARNOps.cpp): mscale = 0.1*ln(s)+1 (matches HF attention_factor), beta_fast=32/beta_slow=1
  (match HF defaults). For this config the high-clamp is INACTIVE (low=3, high=8), correcting an
  earlier plan hypothesis.
- v1 has zero rope_scaling support (grep-confirmed); it is v0-only.

## Deliverables built this session (all base-independent, box-runnable)
1. `scripts/convert_ckpt/ckpt2hf.py` - positional .ckpt -> HF Qwen3 safetensors. Orientation-aware
   self-validating placement (transpose only if reverse-shaped; hard error otherwise, catching a
   wrong-slot tensor). Fused gate/up split before transpose. Reads only the model-param section.
   VALIDATED: synthetic value-verified round-trip (69 HF tensors, transpose + split correct, trailer
   ignored) + strict load into real Qwen3ForCausalLM with ZERO missing/unexpected keys + finite fwd.
2. `scripts/gen_ext_yamls.py` (+ 12 generated YAMLs in `examples/train_full/ext/`) - one template +
   param table; token-budget parity BY CONSTRUCTION (tokens/step and total held constant, grad_accum
   solved per factor = 4/2/1). Prints the matching bluscriptCP env so the CP arm shares the budget.
3. `scripts/eval/ppl_vs_pos.py` - the single scorer. Per-position NLL = logsumexp(logits) -
   logits[target], identical to CP_EVAL_PPL. Reads a shared token-ID npy. Pinned flash_attention_2
   (fp32 full-attn at 32k would OOM). Optional rope injection for CP-converted arms.
4. `scripts/convert_ckpt/rope_parity.py` - CP YaRN vs HF Qwen3 YaRN cache diff. Uses the real
   Qwen3RotaryEmbedding (true mscale) and passes beta_fast/beta_slow EXPLICITLY. HF-side smoke passed
   (cos range +/-1.2079 at factor 8 confirms mscale baked in).
5. `scripts/README_longctx_bench.md` - runbook with all fill-in variables and the exact on-box command
   sequence for both tracks + the fairness audit.

## State of the mandatory converter gate
The converter's shape/orientation/HF-load checks are DONE and passing. The two checks that need the CP
binary + a real base on the box are still pending:
- CP-vs-HF forward parity on the same tokens (only thing that catches a same-shape permutation).
- CP_DUMP_NAMED identity guard - must source identity from LIVE NAMED MEMBERS (not a second walk of the
  serialization order) and cover EVERY same-shape tensor (incl. non-adjacent input_ln/post_attn_ln).

## Pending / next steps
- CP-side edit: add CP_DUMP_NAMED named-member dump in bluscriptCP + wire the identity guard into the
  converter.
- build_eval_stream.py + a CP_EVAL_PPL read-from-npy toggle (eval-faithfulness: both scorers read the
  same token-ID array).
- On box: convert real base -> base_hf/, run the mandatory gate, then Track 1 arms + scoring + plots,
  then Track 2 systems sweep (extend Tests/bluscriptcp/mem_scaling_sweep*.sh).
- Fairness audit before any headline: optimizer cold vs warm start (LF cold-starts Adam; decide/document
  the CP arm), CP_REWARMUP schedule shape, QK-norm eps (CP 1e-6 vs HF 1e-5) measured not assumed.

## Notes
- No trained bluscriptCP 4k base exists locally (only old non-CP bluscript .ckpt at 1.37GB and 5-step
  smokes). The local LF 917-step run only reached step 20. Real base is on the 6000 Ada box.
- No git operations performed (per user rule). All CP-source reads were confirmatory only.
