2026-09-26 - 15:30 - Ported the missing bluscript_zero_ckpt.cpp features into bluscriptCP.cpp (DCP checkpoint loading, multi-domain mixture loader), fixed the local build, verified with 2-step runs on 2x RTX 3060 - CP / BluTrain/dist/Context_Parallelism/Scripts/Blutrain/bluscriptCP.cpp

# Session report

## Build: bluscriptCP was never broken, I was building it wrong

The blurun-build recipe belongs to the ZERO trainer. Using it on bluscriptCP
produced a missing-header error and then ~12 undefined symbols (ContextParallel,
CPProcessGroupNCCL, DeviceMesh, build_rope_cache_longrope, the CP attention
kernels). All of it was an artefact of the wrong recipe.

bluscriptCP has its own target, documented at dist/Context_Parallelism/Makefile:509:

    make CP_FUSED_ROPE=1 bluscript-cp

It already declares BLUSCRIPT_DIST_SRCS (DataParallel.cpp + profiler.cpp), which
is what the last three undefined symbols needed. I missed the target initially by
truncating a grep with `head -10`; the match is ~490 lines further down.

RETRACTED: my earlier claim that "bluscriptCP.cpp was never wired into any build
target". It has one and always did.

## Second build issue, local only

This laptop's CUDA 13 toolkit ships NO cuBLAS. /usr/local/cuda-13/lib64 has
libcudart but no libcublas/libcublasLt, and libtensor.so itself reports
`libcublas.so.13 => not found`. The project venv has both, so the link needs:

    -L BluTrain/envv/lib/python3.10/site-packages/nvidia/cu13/lib

Unrelated to CP. On the server this does not arise.

## What was ported

### 1. DCP checkpoint loading  (DONE, compiles, compiled OUT locally)

CP already had CP_INIT_FROM, which loads weights BEFORE the optimizer is built --
the correct site, and the one that superseded its retired CP_CKPT_WEIGHTS_ONLY.
It only understood a FLAT .ckpt. bluscript_zero_ckpt.cpp writes DCP, a
<name>/{metadata.bin, data_rank_*.bin} DIRECTORY, so CP could not start from any
checkpoint the ZeRO trainer produced.

CP_INIT_FROM now branches on what the path IS -- file goes to the flat loader,
directory goes to DistCheckpoint::load + ZeroDcpModule, weights-only with nullptr
for the optimizer so it stays world-size agnostic (ZeRO state is sharded to the
world size that wrote it).

The include is guarded with __has_include and CP_HAVE_DCP, because this checkout
ships dist/Distributed_Checkpointing EMPTY. Without the guard the trainer would be
unbuildable here for a feature most runs never touch. When compiled out, passing a
directory to CP_INIT_FROM dies with a clear message naming the two missing headers
rather than mis-loading silently.

### 2. Multi-domain mixture loader  (DONE, built, RUN-VERIFIED)

CP read one flat CP_DATA_ROOT, so it could not hold a domain ratio while changing
sequence length -- which is what a long-context extension needs (Fu et al.
arXiv:2402.10171: changing domain balance and length distribution together is what
makes length upsampling hurt the web domains).

Added, opt-in via CP_MIX_ROOT so every existing launch is unchanged:
  * DomainConfig list + c0/c1/c2 presets with the SAME defaults and env-var
    overrides as the ZeRO trainer, so a rung moves between trainers unchanged
  * phase schedule with CP_MIX_ANNEAL_START, CP_MIX_STATIC ablation handle
  * set_weights() on phase change, fired from the step loop
  * per-domain val loaders via create_val_loaders()

SHARDING: both loaders take dp_rank/dp_size, NOT the global rank. Every CP rank in
a group must see the same tokens -- the sequence is split across cp ranks, the
batch is not. Getting this wrong would silently desynchronise a CP group.

TrainSource: DataLoaderLite and MixedDataLoaderLite share no base class and the
loop needs only next_batch/skip_batches/reset, so a thin owning wrapper (unique_ptr
members, rule of zero) is cheaper than templating the training loop.

DataLoader.h: added `int domain_index = -1` to Batch. MixedDataloader.h requires
it; this checkout's DataLoader.h predates it. Default -1 means "not from a
mixture", so every existing caller compiles and cannot confuse it with domain 0.

### 3. bf16 resid / logits / embed  (NOT DONE -- blocked, deliberately)

CP already has bf16 LINEARS via CP_BF16 + CP_BF16_MASK (QKV/WO/MLP/HEAD). My
earlier "CP lacks BLU_ALL_BF16" was too broad; what it lacks is the other three
stages.

bf16_logits CANNOT be done safely on this checkout. LossKernels.cu here is dated
Aug 17 and contains ZERO occurrences of VEC_ELEMS / Chunk16 -- the fix that lets
the sparse-CE kernels accept bf16. bluscript_zero_ckpt.cpp documents exactly what
happens without it: 16-byte cp.async staged at a stride of 4 ELEMENTS is 8-byte
aligned on bf16 (hard misaligned-address fault), and reinterpret_cast<float4*> on
row_grad overruns the gradient buffer SILENTLY. Wiring it here would produce
corrupted gradients, not an error.

## Verification

Both paths run on 2x RTX 3060 (sm_86), 2-layer 128-dim model, B=2 T=256:

  FLAT      step 0..7, loss ~10.90, no regression from the pre-port binary
  MIXTURE   preset c2 parsed, all 4 domains loaded, and the phase transition
            fires: "[mix] step 3: phase 'anneal'" with the loader re-targeting
            from {cc .58 edu .05 code .22 math .15} to the anneal weights

Synthetic corpora for the smoke test are in the session scratchpad, not the repo.

## The real problem: this checkout is stale

Three separate blockers today all had the same root cause -- local is behind the
server:
  * dist/Distributed_Checkpointing/include/  EMPTY (no DCP at all)
  * dist/zero/include/ZeroDcpBridge.h        missing
  * MixedDataloader.h                        missing until it was added mid-session
  * LossKernels.cu                           lacks the bf16 fix
  * dist/lib/libdist.so                      never built

Consequence: bluscript_zero_ckpt.cpp CANNOT build on this laptop at all -- it
includes both missing DCP headers at lines 77-78.

The private repo BlubridgeAI/BluTrain branch dist_side_merge has the headers, but
WebFetch gets 404 (private), gh is not installed, no token is configured, and the
CP repo tracks a different remote (adhityablubridge/Context_Parallelism). A
sibling checkout at ~/KurraVishnuSaiAnkannaofBlubridge/BluTrain HAS a complete
Distributed_Checkpointing tree and a built libdist.so, but its trainers are a
different lineage (bluscript_zero_ckpt.cpp is 143,155 bytes vs 158,487 here), so
copying from it risks a silently mismatched serialization format.

## Files changed

  dist/Context_Parallelism/Scripts/Blutrain/bluscriptCP.cpp   +13,208 bytes, 249 added lines
  Tensor-Implementations/include/DataLoader.h                 +1 field (domain_index)

Original saved to the session scratchpad as bluscriptCP.cpp.orig.

## Build command that works locally

  cd dist/Context_Parallelism
  make -j4 CP_FUSED_ROPE=1 bluscript-cp \
    LINK_MEMFLAGS="-Xlinker --no-keep-memory -Xlinker --reduce-memory-overheads \
      -L<venv>/nvidia/cu13/lib -Xlinker -rpath -Xlinker <venv>/nvidia/cu13/lib"

Run needs LD_LIBRARY_PATH covering Tensor-Implementations/lib, Profiler/lib AND
the venv cuBLAS dir -- the baked rpath does not cover cuBLAS.
