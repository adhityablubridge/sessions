# Claude Report — YaRN in build_rope_cache

2026-07-09 - Implemented canonical (HF/Qwen3) YaRN context extension in the RoPE cache builder — Workspace: CP / File: context_parallel/YARNOps.cpp

## Goal

Extend the Qwen3-style model's context (e.g. 1024 -> 8192) without retraining, by adding
YaRN math to the pre-baked `cos_sin_cache` that the existing fused RoPE+RMSNorm GQA attention
kernel consumes. Reference: `.mdfiles/NTK based YARN research.md` (used as a guide, verified
and corrected against the canonical HF/Qwen3 implementation).

## Key findings before coding (verified via Explore agent)

- `context_parallel/YARNOps.cpp` is a copy of libtensor's `RopeOps.cpp`. The Makefile compiles
  `context_parallel/*.cpp` and links CP objects **before** `libtensor.a` (an `--start-group`
  archive). Because YARNOps.o already defines `build_rope_cache`/`rope`/`rope_packed_qk`, the
  archive member `RopeOps.o` is never pulled in — so **editing YARNOps.cpp alone** changes
  `build_rope_cache` for all CP targets. Confirmed at link (YARNOps.o precedes libtensor.a).
- Temperature math: the compiled kernel `gqa_fused_flash_attn_forward` hardcodes
  `scale=1/sqrt(hd)` applied once to QK logits, and RoPE hits **both** Q and K. Baking factor
  `m` into cos/sin therefore multiplies logits by `m^2`. YaRN wants logit temperature `1/t`
  with `sqrt(1/t)=0.1*ln(s)+1`, i.e. `1/t=m^2`. So baking `m=0.1*ln(s)+1` is exactly correct
  and the kernel needs **no** change. The research doc's `adjusted_scale=1/(m^2*sqrt(hd))`
  Python snippet is self-contradictory (it cancels the baked `m^2`); ignored (our kernel has
  no scale arg anyway).
- The device cache kernel `rope_build_cache_cuda(ptr,seq_len,head_dim,base)` has no
  per-dimension frequency hook, so it is bypassed: build on CPU (`DeviceIndex(Device::CPU)`),
  fill, then `.to(device)` (Tensor.h:180). Cost negligible (once at init).
- CP: the cache is built once at the global extended `seq_len` with global scale; existing
  `pos_offset`/RopeDeltas plumbing slices it per rank. No CP-side change. (Caveat: the per-rank
  in-kernel offset path is only reached by the unbuilt `-DCP_FUSED_ROPE` kernel; the standin
  oracle applies RoPE pre-shard at offset 0, so it validates the cache but not per-rank offset
  indexing.)

## Implementation (build_rope_cache rewrite)

- **Params, env-overridable, hardcoded defaults**: `YARN_SCALE=1.0` (=> standard RoPE),
  `YARN_ORIG_MAXPOS=1024`, `YARN_BETA_FAST=32`, `YARN_BETA_SLOW=1`.
- **Temperature**: `m = (s>1) ? 0.1*ln(s)+1 : 1`.
- **Canonical correction range** (dimension-index ramp, matches HF find_correction_dim):
  `low=floor(find_dim(beta_fast))`, `high=ceil(find_dim(beta_slow))`, clamped `[0, half-1]`.
- **Freq blend**: `inv_freq = (base_freq/s)*ramp + base_freq*(1-ramp)`; ramp linear over dim
  index. High-freq (ramp=0) extrapolated, low-freq (ramp=1) interpolated by 1/s.
- **Bake m** into both cos and sin.
- **Build on CPU then copy to device**; removed the CUDA-kernel branch.
- **Robustness (all guard against silent wrongness)**: R2 warn if `|seq_len - s*L| > 1`;
  R3 one-line init log `[YaRN] ... scale/low/high/m` (exposes silent link-order revert); R5
  throw on bad correction range (e.g. inverted betas).
- Scale=1 reduces bit-identically to standard RoPE (`base_freq/s==base_freq`, `m==1`).

## Verification (2x RTX 3060, sm_86, ws=2)

- Syntax check + `make cp-rope-standin` + `make cp-ulysses`: clean (exit 0).
- **Standin parity (MHA; exercises correct kernels + full CP ring fwd/bwd + gamma):**
  - No env (scale=1): ALL 12 gates PASS, `cos=1.0000000`; log `scale=1.000 m=1.0000`.
  - `YARN_SCALE=8`: ALL 12 gates PASS, `cos=1.0` (both ref and CP read same YaRN cache);
    log `scale=8.000 m=1.2079` (= 0.1*ln(8)+1, exact). Analytic hd=128 range: low=11, high=36
    (matches plan).
- **Robustness:** R2 warning fires on `seq_len=64 != 8*1024`; R5 throws
  "bad correction range ... low=17 high=6" on inverted betas.
- **Ulysses parity:** 71-72 PASS; the only failing gates are `dQ`/`dq_gamma` (never
  forward/dK/dV/dk_gamma) — the **pre-existing, documented true-GQA backward-kernel bug**
  (Claude Logs entries 133/137/138). The no-env (scale=1 ≡ standard RoPE) run reproduces the
  identical failures, proving YaRN introduced no new failure class. (One borderline dq_gamma
  gate at cos=0.98 crosses threshold at scale=8 — same buggy quantity, m-scaled.)

## Files touched

- `context_parallel/YARNOps.cpp` — rewrote `build_rope_cache` (+ includes cstdlib/cstdio/
  string/algorithm, anon-namespace `yarn_env_f` helper). `rope`/`rope_packed_qk` untouched;
  header signature unchanged.

## Notes / follow-ups

- Callers must build the cache at `seq_len = target_ctx` and set `YARN_SCALE = target_ctx/L`
  (R2 warning catches a mismatch). For production (`bluscript.cpp:271`) set
  `context_length=8192` + `YARN_SCALE=8`.
- Real per-rank in-kernel offset indexing + long-context quality remain to validate when the
  `CP_FUSED_ROPE` in-kernel RoPE path is built and a real long-context eval is run.
- CORRECTION (supersedes earlier note): the "GQA backward dQ/dq_gamma kernel bug" was retracted.
  The dQ divergence was a bf16-vs-fp32 precision artifact in the test reference (bf16-rounding K
  on the G==1/MHA case alone reproduced the ~0.108 dQ cosine, no GQA involved). The kernel dQ path
  is correct; no true-GQA training-grad corruption. See the CP fused-RoPE kernel work for details.
