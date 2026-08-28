# Claude Report - CP multi-GPU bug localised: data parallelism corrupts non-zero ranks

2026-08-27 - Bisected the bluscriptCP "gradient norm 1e6 / loss frozen" failure down
to DATA parallelism specifically. Context parallelism is clean. Four separate
defects found in shared Data-Parallel infrastructure along the way - CP
(/root/Adhi/BluTrain/dist/Context_Parallelism) / Scripts/Blutrain/bluscriptCP.cpp,
dist/Data-Parallel/src/DataParallel.cpp

Box: mild-fire-glows-fin-03, 4x H100, shared with another tenant (~25 GB/GPU).
Fresh clone at 2449e5c, no local edits at the start. FLUX data (100 train + 1 val
shards, 200,000,000 bytes each, token IDs verified 0..50256 < vocab 50304).

## HEADLINE: CP is usable TODAY with pure context parallelism

    NP=4, CP_SIZE=4 (dp=1 x cp=4):  norms 34.45 / 45.99 / 3.99, IDENTICAL on all
                                    four ranks, loss 11.133 -> 10.458 -> 9.313

That is the configuration long-context testing needs anyway (T=16384 at CP_SIZE>1),
so the long-context work is NOT blocked. Only data parallelism is.

## The bisect

Per-rank instrumentation was the turning point. Every "healthy" number before it
came from the master rank only, because bluscriptCP redirects non-master stdout to
a null buffer -- which hid the failure completely for days.

| config | dp | cp | rank0 norm | rank1 norm | verdict |
|---|---|---|---|---|---|
| NP=1 | 1 | 1 | 53.78 | - | healthy |
| NP=2 | 1 | 2 | 51.71 | 51.71 | healthy |
| NP=4 | 1 | 4 | 34.45 | 34.45 (all 4) | healthy |
| NP=2 | 2 | 1 | **53.78** | **19,822,810** | rank1 broken |

The decisive run: reduction DISABLED (CP_DDP_NO_REDUCE=1) so each rank reports its
own LOCAL gradient. rank0 = 53.7790, rank1 = 19,822,810. The all-reduce is
INNOCENT -- it faithfully averages one good gradient with one garbage gradient.

Then with CP_DDP_INIT_SYNC=0 and CP_SAME_DATA=1 (both ranks read the identical
slice): **both ranks' forward losses are bit-identical (11.160570), and rank 1's
gradient norm is still 1.4e7.** So with identical weights, identical inputs and
identical forward output, rank 1's BACKWARD produces garbage. That is the bug, and
it is still not root-caused.

## Ruled out, each by direct measurement

- **the reduction** - disabling it leaves rank1 garbage
- **NCCL / ProcessGroupNCCL collectives** - standalone 2-rank all_reduce test
  through the same PG path returns [5 7 9] exactly. PASS on both ranks.
- **the data** - token IDs 0..50256 across shards and in both rank windows; and
  forcing both ranks onto the same slice does not help
- **device binding** - cudaGetDevice() reports 0 and 1 correctly per rank
- **the CP sub-group** - cp_pg reports worldsize=1/rank=0 at CP_SIZE=1 and
  worldsize=2/ranks{0,1} at CP_SIZE=2, both correct
- **grad_as_view** - CP_DDP_NO_GRADVIEW=1 does not help
- **bucketing** - the flag is IGNORED (see defect 4), so that test was a no-op
- **CP's own kernels** - parity suites pass (cp-causal-cp-sm90 ALL PASS with
  maxdiff 0.000e+00; cp_qknorm_rope all gates PASS)
- **my Muon/ZeRO/WSD/bf16 port** - the failure reproduces with CP_BF16=0 CP_ZERO=0,
  where every one of those code paths is inert

## Four defects found in shared Data-Parallel infrastructure

1. **init_sync corrupts non-zero ranks' weights.** With CP_DDP_INIT_SYNC=1 (the
   fp32 default) rank1's forward loss is 11.132493 vs rank0's 11.160570; with it
   =0 both are 11.160570. `sync_model_parameter`/`broadcast_coalesced` is writing
   wrong values. Workaround CP_DDP_INIT_SYNC=0 is safe: GPT is built from a
   hardcoded rank-independent seed (1234), so the broadcast is redundant.

2. **finalize_backward is missing the cudaDeviceSynchronize its own comment
   mandates.** The comment explains that work_obj_->wait() is GPU-side only and
   that "CPU-side reads of gradient data can race against still-in-flight NCCL",
   then the call is simply absent. Added locally; did NOT fix the main bug, but it
   is a real latent race.

3. **naive_grad_sync() is dead code AND wrong.** Never called. Its write-back loop
   reassigns elements of a LOCAL std::vector<Tensor>, so the reduced values never
   reach the parameters' gradient buffers.

4. **opts_.bucket_ is never read.** `with_bucket_data(false, ...)` is silently
   ignored -- bucketing is always on. Any bisect using that flag is a no-op.

## Also fixed this session (unrelated but required)

**Static-archive device linking.** CP linked libtensor.a and device-linked TI's
sm_90a objects into an sm_90 target, silently dropping those kernels ->
"named symbol not found" on strided_inner_vec_copy_kernel, then
convert_cuda_fp32_to_bf16_sm90. Fixed by linking libtensor.so (-ltensor); TI
already device-links it. TI *must* be sm_90a (its GQA_fwd/bwd_sm90.cu require
wgmma/setmaxnreg, which plain sm_90 ptxas rejects), and CP *must* be sm_90 so its
per-file compute_90a override fires -- the 11 sm_90 / 60 sm_90a mix is BY DESIGN.
Do not pass SM_ARCH=90a to the CP build.

Note libtensor is only an ORDER-ONLY prerequisite of the CP link, so a rebuilt
library never triggers a relink: `rm -f build/bluscriptCP_exec` before each build.

## Wrong turns worth not repeating

- Believed "the source is exonerated" after reverting to HEAD -- but HEAD (2449e5c,
  "After Muon BluScriptCP changes") CONTAINS the Muon port, so that reverted only 32
  lines of later edits. The premise was wrong.
- Chased an sm_90/sm_90a "mixing" theory and told the user to build SM_ARCH=90a,
  which broke every wgmma kernel.
- Patched ContiguousKernel_sm90.cu twice on the theory that the sm90 Views kernel
  corrupts bucket copies. It does not (disabling it changes nothing here).
- Read master-only output as whole-job health for several rounds. This is the big
  one: the per-rank print should have been the FIRST instrumentation, not the last.

## Next step

The remaining question is narrow and well-posed: with identical weights, identical
data and a bit-identical forward, why does the backward diverge on a rank whose
cp_size==1 but dp_size>1, when cp_size==2 on the same rank is fine? The degenerate
(no-op ring) CP backward path at cp_size==1 inside a >1-rank world is the only
structural difference left untested.

Practical recommendation meanwhile: run everything with CP_SIZE = NP and dp=1.
That is healthy, verified on 4 ranks, and is the configuration long-context testing
requires.
