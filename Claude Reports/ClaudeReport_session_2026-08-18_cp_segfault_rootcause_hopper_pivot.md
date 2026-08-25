# Claude Report — CP segfault root-cause + fix, and Hopper-only pivot

2026-08-18 - Root-caused and fixed the CP tensor-allocation segfault (stale TI object files vs a grown TensorOptions), re-verified Stage-1 PG parity on the clean build, uncovered the upstream fused-attention Blackwell-only regression, and pivoted the plan to Hopper-only - Workspace: CP / Scripts/Blutrain/bluscriptCP.cpp, BluTrain/Tensor-Implementations, .claude/plans/context-we-as-a-delegated-lerdorf.md

## The segfault (was blocking Stage-1 e2e) — FIXED
- Symptom: SIGSEGV in `Storage::Storage` -> `allocator_->allocate()` (garbage `Allocator*`), on the first
  `Tensor::randn`/`zeros` from any user TU; libtensor-internal allocation (`build_rope_cache`) worked.
- Isolation: a minimal probe (just `Tensor::randn`, zero CP code) crashed identically -> not the PG/CP
  work, not a genuine TI incompatibility. gdb showed `get_allocator` was never called -> the
  `allocator != nullptr` branch fired with a garbage `opts.allocator_override`. `sizeof(TensorOptions)=32`.
- Root cause: **stale/partial libtensor build.** `Storage.o` and `TensorFactory.o` were dated 07-03,
  compiled against the OLD `TensorOptions` (24 bytes, before the `allocator_override` field was added);
  `Tensor.o` had been rebuilt (08-18). A prior `make libtensor` re-archived the `.a` (fresh mtime)
  without recompiling those TUs -> 24-vs-32-byte layout mismatch -> `randn`/`zeros`/`Storage` read a
  garbage allocator pointer -> segfault. 123 of 177 TI objects were pre-header-change.
- Fix: `rm -rf BluTrain/Tensor-Implementations/lib/objects && make libtensor` (clean rebuild, exit 0).
  Probe + parity then pass. **Rule going forward: always clean-rebuild libtensor after a TI upgrade;
  its incremental (`-MMD` mtime) build silently misses struct-layout header changes.**

## Stage-1 PG re-verified on the clean build (2x3060, sm_86)
- `cp-rope-standin` ws=2: **ALL PASS** cos=1.0000000 (fwd + dQ/dK/dV/dq_gamma/dk_gamma; contiguous + HeadTail).
- `cp-ulysses` non-fused (MHA + GQA, all group ratios): **ALL PASS** cos=1.0000000.
  Confirms `CPProcessGroupNCCL` ring transport + Ulysses all-to-all are correct against the new upstream.

## Second finding — upstream fused attention is Blackwell-only (upgrade regression)
- `cp-rope-fused` + `cp-ulysses` fused case throw `cudaErrorNoKernelImageForDevice`. compute-sanitizer
  named the real (latched) culprit: `OwnTensor::cuda::launch_gqa_fused<16,32,64>` via
  `gqa_fused_flash_attn_forward`, defined in `GQA_fused_{fwd,bwd}_sm103.cu` — **Blackwell-marked, compiled
  sm_100a/sm_103a only**, hard-called (no arch dispatch). No sm_86/sm_89 image.
- Impact: CP **ring** fused path is fine (self-contained CP-local `cp::cuda::gqa_fused_rope_cp_*`, machine
  arch). CP **ulysses** path (`ContextParallel.h:1741`) routes to the Blackwell-only kernel -> breaks off
  Blackwell. The separate `gqa_flash_attn_forward` (`GQA_fwd_generic.cu:220-295`) has no correct causal
  path on sm_86 either: causal hd128 = Hopper-only (`gqa_v62`, TMA), hd64 = B300-only/maskless, else throws.

## Pivot (user, 2026-08-18): HOPPER-ONLY
- Team has moved to Hopper; all testing from now is on Hopper (sm_90), head_dim 128. Portable/sm_86
  requirement dropped. This resolves the Stage-3 precondition: on Hopper the upstream **separate**
  `gqa_flash_attn_forward` runs `gqa_v62_causal_launch_d128` natively -> **Stage 3 collapses to calling
  the upstream separate kernel; the portable `_cp` causal fork is NOT built.**

## Plan revised (`.claude/plans/context-we-as-a-delegated-lerdorf.md`)
- Context/Decisions: portability premise replaced with Hopper-only + the dispatch evidence.
- Stage 3: rewritten to "call upstream Hopper `gqa_flash_attn_forward` / causal bwd"; fork dropped.
  New open design point: ring sub-chunk masking via per-step `is_causal` {full/causal} + external LSE
  merge (vs a masked step for partial-diagonal HeadTail cases) — resolve when mapping the zig-zag schedule.
- Stage 4: ring keeps its Hopper-capable CP-local `_cp` fused kernel as the revertible fallback; **Ulysses
  is no longer optional** — must be re-pointed off the Blackwell-only fused kernel onto the upstream
  separate attention to run on Hopper at all.
- Stage 2 (CP-aware standalone QK-norm+RoPE fork of `fused_qknorm_rope_*_kernel.cu`, head/tail deltas)
  remains the one real kernel fork; build for sm_90.

## Outstanding
- Stage 2 implementation (build sm_90 here; parity run on Hopper by the team — this box is 2x3060).
- Stage 3/4 wiring per the revised plan.
