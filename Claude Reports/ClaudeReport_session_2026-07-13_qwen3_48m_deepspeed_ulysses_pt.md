# ClaudeReport_session_2026-07-13_qwen3_48m_deepspeed_ulysses_pt

2026-07-13 - 14:59 - Built a LlamaFactory (v1) DeepSpeed + Ulysses from-scratch pretraining reference for a 48M Qwen3, matched to the CP framework's bluscriptCP.cpp config, to benchmark against the in-house CP-Ulysses run - Workspace: LlamaFactory - Files: qwen3_48m/*, examples/v1/train_full/qwen3_48m_pt_ds_ulysses.yaml, data/pretrain_c4_demo.json

---

## Goal
User has an in-house DL framework (CP project) that pretrained a ~48M-param Qwen3-architecture
model from scratch using DeepSpeed-Ulysses-style sequence parallelism. They wanted an equivalent
**LlamaFactory + DeepSpeed-Ulysses** script for a like-for-like comparison. Pretraining (not SFT),
from scratch (random init), 2x RTX 3060 (24GB VRAM, 31GB RAM).

## Key findings (LlamaFactory internals)
1. **Ulysses SP is v1-only** (`USE_V1=1`). Implemented in
   `src/llamafactory/v1/plugins/model_plugins/parallelization/{sequence_parallel,ulysses}.py`
   (verl-derived). Enabled via `dist_config: {cp_mode: ulysses, cp_size: N}`. Shipped example:
   `examples/v1/train_full/train_full_ulysses_cp.yaml` (pairs Ulysses with **fsdp2**).
2. **The clean `stage: pt` + `train_from_scratch` are v0-only** (`src/llamafactory/model/loader.py:168`),
   and v0 has **no Ulysses**. So PT-stage and Ulysses cannot both be used in one native path.
3. **v1 has only sft/dpo/rm trainers** (no pt). PT objective reproduced by feeding raw text as
   **empty-prompt alpaca** (instruction/input="" , output=document) -> prompt masked, whole document
   supervised = full-sequence LM loss. Chat-wrapper neutralized via `custom_chat_template` that emits
   only message content (v1 renders through `apply_chat_template`; falls back to ChatML otherwise).
4. **v1 has NO true random-init-from-config path.** `init_on_meta` -> `from_config` under
   `init_empty_weights`, then the backend *chunk-loads a checkpoint from disk*
   (`ValueError: No checkpoint files found`). Fix: materialize a **seeded random-init checkpoint to
   disk**, then let v1 load it (this is cleaner + reproducible anyway).
5. **v1 optimizer/scheduler are barely wired**: default `AdamW(lr)` (wd 0.01) + constant LR
   (`LambdaLR->1.0`). Cosine / warmup / min_lr / wd 0.1 need a custom Optimizer+LRScheduler plugin
   (empty stubs in `plugins/trainer_plugins/{optimizer,lr_scheduler}.py`). NOT settable from YAML yet.
6. Ulysses **requires flash-attn** (monkey-patches `_flash_attention_forward`); no flash -> SP is a no-op.
7. Launcher shells out to `torchrun` from PATH — must have `.venv/bin` on PATH or torchrun picks the
   system Python 3.10 (no llamafactory). Run with `export PATH="$PWD/.venv/bin:$PATH"`.

## Config built (matches bluscriptCP.cpp lines 98-140 exactly)
d_model 384, n_layers 6, q_heads 6, kv_heads 2 (GQA), head_dim 64, ffn 1024, vocab 50304 (GPT-2/tiktoken),
tie=false, rope_theta 5e5, rms_eps 1e-5, ctx 4096. => **48.076M total** (9.44M transformer + 19.32M embed
+ 19.32M untied lm_head). Training: micro_batch 2, cutoff 4096, global_batch 128 samples (=524288 tok/step,
grad-accum 64 at dp=1), lr 6e-4, max_grad_norm 1.0, 917 steps, seed 42.
NOTE: bluscriptCP.cpp:102 writes `head_dim = 384/12 = 32` but comment/geometry say 64; used 64 (6*64=384).

## Validation (no-CP, fsdp2, sdpa — no flash-attn needed)
`USE_V1=1 FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0 llamafactory-cli train qwen3_48m/validate_nocp.yaml`
-> step1 loss 10.9605 (~ ln(50304)=10.83 => correct random init), 10.95, 10.65 decreasing, exit 0.
Cross-check: CP framework's own tiny run logged 10.89->10.23 (same baseline).

## Deliverables
- `qwen3_48m/` : config.json + GPT-2 tokenizer (model_max_length 4096) + seeded random-init model.safetensors
- `qwen3_48m/pretrain_data.yaml` : v1 dataset spec (empty-prompt alpaca)
- `data/pretrain_c4_demo.json` : 300-doc demo corpus (swap for real corpus)
- `examples/v1/train_full/qwen3_48m_pt_ds_ulysses.yaml` : the benchmark run (deepspeed z3 + ulysses cp_size 2)
- `qwen3_48m/validate_nocp.yaml` : fast sanity config

## Launch (after flash-attn finishes)
```
cd /home/blu-bridge25/LlamaFactory
export PATH="$PWD/.venv/bin:$PATH"
USE_V1=1 FORCE_TORCHRUN=1 llamafactory-cli train examples/v1/train_full/qwen3_48m_pt_ds_ulysses.yaml
```

## Open / pending
- flash-attn 2.8.3.post1 compiling from source (CUDA 13.0, sm_80/90/100/120, MAX_JOBS=2) — long build,
  may fail on cu130. Background task bc0sjmyqn.
- After flash-attn: launch the DeepSpeed+Ulysses run; verify cp_size=2 engages and loss ~10.8 start.
- Fair-comparison gaps vs CP run: (a) LR schedule/warmup/wd not matched (v1 limitation);
  (b) init distribution is HF Qwen3 normal(0,0.02) vs CP framework's own init;
  (c) small chat-wrapper only if custom_chat_template not honored.
- Environment: system RAM 31GB rules out 4B/8B full-param; 48M is trivial.

---
## Update 15:20 (exact-match refinements after user gave bluscriptCP.cpp cfg)
- head_dim CORRECTED 64 -> **32** (user confirmed; 384/12). attn width = 6*32 = 192.
  New total = **46.896M** (transformer 8.263M + embed 19.317M + untied lm_head 19.317M).
- Wrote custom v1 plugins (were empty stubs):
  - `plugins/trainer_plugins/optimizer.py` -> `@OptimizerPlugin("adamw")`: nanoGPT wd grouping
    (ndim>=2 decay, else no decay), betas/eps/lr/wd from optim_config.
  - `plugins/trainer_plugins/lr_scheduler.py` -> `@LRSchedulerPlugin("cosine_with_warmup")`:
    linear warmup -> cosine -> min_lr floor. Verified curve: step0 6.59e-6, step90 6.0e-4 (peak),
    step916 6.0e-5 (floor). Exact match to max_lr 6e-4 / min_lr 6e-5 / warmup 91.
  - YAML now sets optim_config{adamw, lr6e-4, wd0.1, betas[0.9,0.95], eps1e-8} +
    lr_scheduler_config{cosine_with_warmup, warmup_steps91, min_lr_ratio0.1}.
    betas NOT in user cfg lines -> assumed GPT-3/nanoGPT (0.9,0.95); flagged for confirm.
- Init distribution matched EXACTLY to bluscriptCP.cpp (read lines 143-341): plain normal(0,std),
  no bias, resid_scale = 1/sqrt(2*n_layers) = 0.288675. std 0.02 for q/k/v/gate/up/embed/lm_head;
  0.02*resid_scale = 0.005774 for o_proj + down_proj; all norm gains = 1.0. Checkpoint regenerated
  (seed 42), measured o_proj std 0.005769, q_proj 0.020 -> matches. 32 std-0.02 + 12 scaled tensors.
- Final validation: loss ~10.87 (~ln(50304)), trains, plugins load, EXIT 0.
- STILL PENDING: flash-attn compile (bc0sjmyqn) then the real DeepSpeed+Ulysses launch.

---
## Update 16:18 (RUN WORKING - backend forced to FSDP2)
- HARD BLOCKER: DeepSpeed + Ulysses is impossible in LlamaFactory. hub.py:65
  `shard_model_deepspeed` raises "CP currently requires dist_config.name: fsdp2" whenever
  cp_size>1. Ulysses CP is FSDP2-only. => switched backend to FSDP2 (ZeRO-3-style fully
  sharded; the combo LF ships/tests). DeepSpeed remains available only WITHOUT Ulysses.
- BUG FIXED: model_engine.py from_config (meta / from-scratch path) did not pass
  attn_implementation, so config._attn_implementation stayed default and base_trainer.py:150
  SP check "requires flash attention" failed. Patched from_config to pass
  attn_implementation=self.args.flash_attn (+ trust_remote_code).
- Also: DeepSpeed z3 does not do FSDP2's meta chunk-load, so init_on_meta gave
  "Cannot copy out of meta tensor" under deepspeed; FSDP2 handles meta materialization fine.
- Working config: examples/v1/train_full/qwen3_48m_pt_fsdp2_ulysses.yaml
  (fsdp2 + init_on_meta + cp_mode ulysses + cp_size 2). Old *_ds_ulysses.yaml removed.
- Smoke 5-step PASS: cp=2/dp=1, "Replaced _flash_attention_forward ... for sequence parallel",
  loss 10.905->10.706, CSV 14 cols + training_config.txt written, [TIMING] populated
  (fsdp2 separates clip~3ms/optim~6ms; loss=0 fused into fwd~2.3s; bwd~4.6s; dt~7s; tok/s~75k).
- FULL 917-step run launched in background (task bsc4vevqv), ETA ~2h. Outputs in
  outputs/qwen3_48m_pt_fsdp2_ulysses/{training_log.csv,training_config.txt}.
