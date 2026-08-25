# Claude Report — checkpoint resume segfault fix

2026-07-10 - Fixed AdamW::load_state segfault when resuming from a checkpoint saved before the optimizer's first step() - CP / BluTrain/Tensor-Implementations/src/nn/optimizer/Optim.cpp

## Reported symptom

`CP_CKPT=1 CP_RESUME_CKPT=47 make run NP=2` segfaulted (signal 11) shortly after
backward pass on the resumed run, and separately resume picked run 48 instead of
the intended run 47.

## Two separate issues

1. **Wrong env var name**: the flag is `CP_CKPT_RESUME`, not `CP_RESUME_CKPT`. The
   mistyped name was silently ignored, falling back to auto-resume-latest, which
   picked run 48 (highest run present) instead of 47. Also, run 48's only saved
   checkpoint was `step_0` — there was no `step_5` for that run to resume from.

2. **Real segfault bug** in `AdamW::load_state`
   (`BluTrain/Tensor-Implementations/src/nn/optimizer/Optim.cpp:1736`):
   - `AdamW` lazily allocates its momentum/variance buffers (`m_`, `v_`) inside
     `step()`, on first call — empty (size 0) before that.
   - The checkpoint save (both `gpt2_cp_test.cpp` and `gpt2_fmha_ddp.cpp`) lives in
     the val block, which runs *before* that iteration's own `optimizer.step()`.
     So a fresh run's first checkpoint (typically step 0) is saved with `m_`/`v_`
     count = 0.
   - `load_state` read `count=0`, resized `m_`/`v_` to 0, and unconditionally set
     `initialized_ = true`. That disabled `step()`'s own lazy-init guard
     (`if (!initialized_)`), so the resumed run's first `optimizer.step()` indexed
     `m_[i]`/`v_[i]` against empty vectors — out-of-bounds `std::vector::operator[]`,
     undefined behavior, observed as SIGSEGV.
   - **Confirmed this is not CP-specific**: `gpt2_fmha_ddp.cpp`'s only
     `optimizer.step()` call (line 1366) is also after its val/save block (1099-1167),
     so a fresh DDP run's step-0 checkpoint has the identical exposure. A prior
     "it worked" DDP resume doesn't prove safety — UB doesn't reliably crash.

## Fix

`AdamW::load_state`: only set `initialized_ = true` when `count > 0`. A `count == 0`
checkpoint (optimizer never stepped) now leaves `initialized_` false, so the next
real `step()` call lazily allocates correctly-shaped zero buffers exactly as it would
for a brand-new optimizer. Step-0 checkpoints remain fully saveable and resumable —
no functionality lost. Rebuilt `libtensor` (only `Optim.o` recompiled) + relinked
`gpt2_cp_test_exec`.

Also fixed, incidentally: bare `make` (no target) resolves to a pre-existing
`build/objects/.cp_rope_flags` stamp rule as its default goal (textually precedes
`all:` in the Makefile, unrelated to this task) — `make all` must be used explicitly
to force a real rebuild/relink; a bare `make` can silently no-op even when the
target binary is missing.

## Verification

Reproduced the exact failure: created a genuine pre-optimizer-step checkpoint
(`CP_MAX_STEPS=1`, save captured before that step's own `optimizer.step()`), then
resumed it in a fresh process. Before the fix this configuration is the exact
crash trigger; after the fix, training resumed cleanly through 3 steps (sane loss/
grad-norm progression, token generation ran, no segfault, no NaN).
