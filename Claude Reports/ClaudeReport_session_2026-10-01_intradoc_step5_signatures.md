# Intra-document masking - step 5: kernel signatures, nullptr plumbing, one-sided guard

2026-10-01 - Threaded doc_q/doc_k through both CP Hopper attention entry points and every internal layer including the non-Hopper stubs, added the both-or-neither guard, and confirmed against the golden record that the change is a true no-op - Workspace BluTrain/dist/Context_Parallelism / context_parallel/GQACausalCP_sm90.h, GQA_causal_cp_fwd_sm90.cu, GQA_causal_cp_bwd_sm90.cu, Tests/cp_doc_guard_test.cu

---

## Pre-work: reconciling the pull

The remote had moved `a20e3b6 -> 8604afe` (158 objects) while this tree sat at the
older base, which is exactly the divergence I had flagged as the one thing that
would change my advice against syncing. I snapshotted everything first, the user
pulled, and then:

- **Both attention kernels came through byte-identical** (`4e53b0d9...`,
  `d387034e...`), as did `cp_causal_cp_sm90_parity.cpp`, the Makefile,
  `SDPAMerger.h` and `SplitAttnSDPA_cp.h`. I rebuilt and re-ran the parity test
  against the post-pull tree: still byte-identical to the golden record. **No
  re-capture needed** -- the gate survived the pull intact.
- **Six of my seven files matched the remote byte-for-byte**, which independently
  confirms the re-port onto this box was faithful rather than a drifted copy.
- **One file needed merging.** `cp_doc_ring_slice_test.cpp` on the remote predates
  the discovery that the parity test never enters the ring loop, so it lacked
  `check_selector_equivalence`. I verified mine was a STRICT superset (the pulled
  version contained nothing mine lacked) before restoring it. Now committed.
- The pull also brought a real bug fix in `ContextParallelBackward.h`: the backward
  KV rotation packed its staging buffer on stream 0 while `alltoallv_async` issues
  sends on the dedicated `cpRingStream`, so the A2A could transmit before the
  memcpys landed -- stale bytes, wrong dK/dV, grad-norm explosion, forward exact,
  AlltoAll-only. Lands in the backward ring path where steps 8-9 will work.

## Step 5 proper

Added `const int* doc_q` / `const int* doc_k`, defaulted to `nullptr`, at **six**
points. Checked every call site first: none passes `stream` positionally (all stop
at `is_causal`), so inserting the pair between `is_causal` and the defaulted
`stream` breaks nothing.

1. `GQACausalCP_sm90.h` -- both declarations, with the contract documented.
2. FWD Hopper device kernel.
3. FWD **non-Hopper stub** -- the `__global__` no-op that keeps the TU compilable.
4. FWD host wrapper (+ pass-down at the launch).
5. BWD Hopper device kernel, non-Hopper stub, `launch_gqa_causal_cp_bwd_v44`.
6. BWD host wrapper (+ pass-down).

The parameters are threaded but **not yet read**. That is deliberate: it keeps
step 5 provably behaviour-free and makes step 7's diff purely about the mask.

**Both-or-neither guard** in both host wrappers, placed ahead of the shape check
and TMA setup so a one-sided call throws without dereferencing anything. A
one-sided doc array is always a plumbing bug; silently treating it as "masking
off" would train with the mask half-applied, which presents as a slightly worse
model rather than as a failure.

**Also fixed a pre-existing discard.** `cudaFuncSetAttribute` in the forward
wrapper ignored its return. It is the only thing raising the dynamic-smem cap
above 48 KB, so on failure the launch fails anyway -- with an opaque "invalid
argument" far from the cause. Now checked and rethrown with context.

## Verification

| Check | Result |
|---|---|
| Golden gate, `doc = nullptr` | **byte-identical** to the pre-change record |
| One-sided nullptr guard | 4/4 rejected (fwd and bwd, each direction) |
| `cp_doc_ids_test` | 14/14 |
| `cp_doc_ring_slice_test` | selector equivalence + 8 ring configs + 4 rejections |
| **sm_86 cross-compile of both kernels** | both TUs compile -- the stubs are right |
| Eval jobs | undisturbed |

The sm_86 check is the one that matters most here: the plan warns that missing the
non-Hopper stubs breaks every `SM_ARCH=86` dev-box build, and this box is sm_90,
so the failure would not have shown up locally. Cross-compiling both TUs to sm_86
proves the stub signatures track the real ones.

### Two things I got wrong and corrected mid-step

- **Built the guard test with `-arch=sm_90`**, which cannot assemble wgmma/TMA.
  The Makefile filters `-arch` out entirely for these two TUs and uses
  `-gencode arch=compute_90a,code=sm_90a`. Note the plan says plain sm_90 fails
  *silently*; here it failed **loudly** with explicit ptxas errors. Resolved by
  linking the guard test against the objects `make` already built correctly.
- **First guard test also exercised the both-null rows**, which correctly passed
  the guard and then tripped the TMA encoder on null tensors. Those rows were
  redundant: the entire parity suite passing already proves the guard does not
  fire on the default path. Narrowed to the one-sided cases only.

`Tests/cp_doc_guard_test.cu` is saved in the tree. Verified it is NOT swept into
the build: `CU_SOURCES` globs `find $(CP_DIR) $(PG_DIR)`, which excludes `Tests/`.

## Eval jobs - normal turnover, not a crash

A mid-run snapshot showed 2 of 3 jobs alive and I checked rather than asserting.
`master.log` shows a 6-job queue at 3 concurrent: `niah_single`/`niah_multikey`
finished 03:39 and were replaced by `vt`/`val_forward`, `val_backward` started
04:03, `niah_multivalue` completed 04:13. 4 of 6 tasks done, 2 in flight. The
exited pids completed normally and were replaced off the queue.

## Next

Step 6 (FWD dead-row guard alone, golden still bit-identical because fully-masked
rows are unreachable until step 7) and then step 7, which is where the golden
record can legitimately move -- and where the `!diag` fast-path at the verified
counterexample (T_q=T_k=256, boundary at 100, q_tile=3 vs n=0) has to be fixed.
