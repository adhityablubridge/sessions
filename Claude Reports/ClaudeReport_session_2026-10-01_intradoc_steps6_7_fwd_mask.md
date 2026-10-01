# Intra-document masking - steps 6 and 7: forward dead-row guard and the element mask

2026-10-01 - Landed the forward dead-row guard and the forward intra-document element mask including the unsafe !diag fast-path fix, and proved both with a new fp32-reference test plus mutation testing - Workspace BluTrain/dist/Context_Parallelism / context_parallel/GQA_causal_cp_fwd_sm90.cu, Tests/cp_doc_mask_fwd_test.cu

---

## Step 6 - forward dead-row guard (FOUR sites, not the three planned)

Traced a fully-masked row through the online softmax and found it breaks at four
places, one more than the plan listed:

| Site | Dead-row failure | Guard |
|---|---|---|
| `cc0/cc1` | `ex2(-INF - -INF)` = NaN | `(m == nm) ? 1.0f : ...` |
| `b0/b1` | `b = +INF`, then `ex2(fma(-INF, s, +INF))` = NaN | `(nm == -INFINITY) ? 0.0f : ...` |
| `r0/r1` | `rcp_approx(0)` = +INF, `O_acc * INF` = NaN | `(l > 0) ? rcp(l) : 0.0f` |
| **`LSE`** (new) | relied on `-INF + logf(0)` | `(l > 0) ? ... : -INFINITY` |

Each branch is provably inert on a live row, which is why the golden record did
not move: `ex2_approx(0)` is exactly `1.0f` (the pre-existing `cc0 == 1.0f` warp
vote already depends on that identity, so substituting the literal cannot
differ), `nm` is finite whenever a row has any live key, and a live row always
has `l >= 1` because its own max contributes `ex2(0) = 1`.

**The fourth site exists because of `--use_fast_math`**, which is in NVCCFLAGS.
The unguarded `(m0*scale2)*LN2 + logf(l0)` cannot be trusted to evaluate to -INF
under it, and a NaN there does not stay local: it flows into SDPAMerger and
poisons the whole ring step rather than one row. Same reasoning drove writing all
four as explicit comparisons rather than leaning on IEEE INF arithmetic.

At the time this landed I stated plainly that byte-identical proves only that the
guard does not FIRE -- dead rows are unreachable under pure causal, since row 0
always has `gcol = 0 <= grow = 0`. It is validated retroactively by case 4d below.

## Step 7 - the forward element mask

**Shared memory, measured not assumed.** Queried the device directly:
`cudaDevAttrMaxSharedMemoryPerBlockOptin` = 232,448 B, pre-existing use 229,456 B,
headroom 2,992 B. `sDocK[NSTAGES][BLOCK_N]` int32 = 1,536 B leaves 1,456 B. Added
a `static_assert` against 232,448 so this can never regress silently. The size is
added UNCONDITIONALLY: `cudaFuncSetAttribute` is cached behind a `static bool cfg`,
so a conditional size would bake the first caller's choice in for the process.

**Q side in registers, K side in shared.** Each thread owns exactly two rows
(R0/R1 -> gR0/gR1) and they are invariant across the whole `n` loop, so two scalar
loads per q_tile beat any staging. K ids must be per-tile, hence smem.

**Head-invariance is a real trap.** `kv_row_base` is `(b*Hkv + hkv) * T_k`, but
doc_k is `[B, T_k]` -- head-invariant -- so the doc base is `b * T_k` with no hkv.
Folding hkv in would read another head's slice and mask the wrong keys. Called out
in a comment at the site.

**Producer staging and ordering.** All 128 producer threads pass
`mbar_wait(&empty[stage])`, so staging there is WAR-safe; one id per thread since
WG == BLOCK_N == 128 (static_assert). Ordered ahead of lane 0's `expect_tx` by
`bar.sync 3, 128` -- verified barriers 1 and 2 are taken by the baton at 256
threads each and 0 is the implicit `__syncthreads()`, so 3 is the first free id,
and the producer warpgroup is exactly 128 -- followed by
`fence.proxy.async.shared::cta` to close the generic/async proxy question.
Deliberately did NOT add an extra `mbar_arrive(&full[stage])`: `full[]` has
arrive-count 1 and a second arrival would flip the phase early, tearing the tile.

**Stage correspondence verified, not assumed.** The QK is software-pipelined
("issued one iteration ago"), so I confirmed which slot `S_acc` belongs to before
indexing `sDocK`: the prologue issues QK(0) from `sK[stage*TILE_K]`, and at
iteration n `cur == stage` with `Vt = &sV[cur*TILE_K]`. All three agree, so
`sDocK[cur*BLOCK_N + keycol]` is right.

**The `!diag` fast path was genuinely unsafe.** `diag` answers only "can the
causal test fire", which says nothing about document boundaries. Gated on
`!diag && !need_doc`.

## Verification

New `Tests/cp_doc_mask_fwd_test.cu` -- the first test that actually exercises the
mask, since `cp_causal_cp_sm90_parity` only ever passes doc = nullptr.

| Case | Result |
|---|---|
| 4a single-document ids vs null path | **BIT-IDENTICAL** |
| 4b boundary at T/2, B=2, per-row-distinct | cos 0.9999969 / 1.0000000 |
| 4c !diag counterexample (q_tile=3 vs n=0, boundary@100) | pass, incl. a targeted rows-192-255 band |
| 4d every row fully masked | **O == 0 exactly, LSE == -INF** |

4d is what retroactively proves step 6's guard: no NaN leak, exact zeros.

### Mutation testing - the fix is load-bearing

Reverted `!diag && !need_doc` to `!diag` and rebuilt. 6 failures: 4b O collapses to
cos 0.75, the targeted rows-192-255 band to cos 0.7322 (exactly the band predicted
from the counterexample), and 4d fails completely with 131,072 non-zero O elements
and 1,024 LSE entries not -INF. 4a still passed, correctly -- with a single
document the mask never masks anything, so the unsafe path is harmless there,
confirming 4a is not the discriminator and 4b/4c are.

### A weakness the mutation run exposed, now fixed

Under the mutant, 4c's **LSE check passed** at cos 0.9956. Cosine is a weak
detector on LSE: the values are large and similar, so a badly wrong row barely
moves the angle. Only the O band caught it. Added absolute maxdiff bounds
alongside the cosine (LSE 5e-02, O 2e-02) -- comfortably above the correct
kernel's 4.7e-04 / 2.8e-03 and far below the mutant's 1.50 / 6.2e-02. The test now
fails for the right reason rather than by luck.

## Full sweep after step 7

golden gate BYTE-IDENTICAL; doc-mask 4a-4d ALL PASS; one-sided guard 4/4;
doc-id scan 14/14; ring slice + selector PASS; cp-mspoe, cp-qknorm-rope,
cp-rope-longrope, cp-string-ref, cream-cache-parity all PASS; both kernels still
cross-compile to sm_86; trainer still links.

(`cp-rope-fused` remains broken by a pre-existing libtensor arch issue, proven
independent of this work by rebuilding with pristine headers. User confirmed the
fused RoPE+norm+attn path is retired and only `CP_ATTN_FUSION=split` is used, so
it is out of scope.)

## Next

Step 8: the backward element mask plus its own dead-row guard. The backward is
where `CP_BWD_PER_STEP_LSE=1` feeds `saved_lse_per_step_[i]` straight into
`fused_p_stsm`'s `l0 = sLSE[r0]*LOG2E`, so an LSE of -INF becomes `ex2(+INF)` ->
`P = INF` -> NaN grads, and the template zeroing happens AFTER the ex2 and does
not save it. Then step 9 (ring slicing + CP_DOC_MASK flag and gates).
