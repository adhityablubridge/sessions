# Intra-document masking - steps 2-4 re-landed on 1x H200, plus the step-1 golden dump

2026-10-01 - Ported the intra-doc masking steps 2-4 onto the H200 CP tree, captured the pre-change golden record for the sm90 kernels, and proved the two header changes are byte-for-byte behaviour-neutral - Workspace BluTrain/dist/Context_Parallelism / context_parallel/DocIdKernel.{cu,cuh}, DocMask.h, ContextParallel.h, RopeDeltas.h, Tests/cp_doc_ids_test.cpp, Tests/cp_doc_ring_slice_test.cpp, Tests/golden/

---

## Environment findings (these changed the plan)

This H200 box is **not** the machine steps 2-4 were built on. The CP tree lives at
`/root/Adhi/BluTrain/dist/Context_Parallelism`, and it was **clean** - no
`DocIdKernel.*`, no `DocMask.h`, no `sub_chunk_for`, no `doc_local`. All of
steps 2-4 had to be re-landed here from scratch.

Before editing anything I verified all nine multi-line edit anchors were present
and **unique** in this tree, so the ports applied to the same code they were
written against. One anchor is unique only as a multi-line block: the single line
`out.idx_local = OwnTensor::gather(...)` appears twice, because
`shard_sequence_pre_embed_hybrid` gathers the same way.

**The GPU was busy and stayed busy.** Three `bluscriptCP_exec` RULER-eval jobs
were running (checkpoint `sep29_8000`, 6 tasks, 39.2 GB each, 117.6 of 143.8 GB),
and critically they run with `CP_ATTN_MODE=ring CP_ATTN_FUSION=split` - the exact
kernels this work modifies. Nothing was killed, restarted or rebuilt out from
under them; all three were still alive and at the same memory at the end. The
build was safe to run in parallel because the box has 44 cores at load 3.2 - the
eval jobs are GPU-bound, not CPU-bound.

## Step 1 - the golden dump

Captured `Tests/golden/cp_causal_cp_sm90_parity.golden.txt` with a short README
explaining its scope. 20 of 20 checks PASS, cosines to 7 decimals and maxdiff to
3 significant figures, across all four cases (causal-vs-oracle, KHeadHalf,
QTailHalf, full-noncausal).

### Ordering slip, and how it was repaired

The plan puts the golden dump FIRST, before any edit. I applied steps 3-4 before
capturing it. Steps 3-4 should be behaviour-neutral - a defaulted parameter that
launches nothing, and a selector refactor with an identical truth table - but
"should be" is precisely the claim the golden dump exists to test, so taking the
dump after the edit would have made the gate circular.

Repaired rather than hand-waved: pristine copies of both headers had been kept, so
the sequence became restore-pristine -> build -> dump A -> re-apply -> rebuild ->
dump B -> compare. **Result: byte-identical, md5 `2a4f0fe2...` both sides.**

### What the golden record does and does not cover - correction to my own claim

It covers the kernels. It does **not** validate the `sub_chunk_for` extraction.
`Tests/cp_causal_cp_sm90_parity.cpp` calls
`OwnTensor::cp::cuda::gqa_causal_cp_{forward,backward}_sm90` **directly**
(lines 212/245/286/312) and never enters `ContextParallel::forward_cp`, so it
never executes the ring loop where the selector was substituted. The case labels
"KHeadHalf" and "QTailHalf" refer to the tensor shapes, not to the selector
having chosen them.

I checked this before claiming the refactor was validated, not after.

The gap is closed by `check_selector_equivalence` added to
`Tests/cp_doc_ring_slice_test.cpp`: it compares `sub_chunk_for` against a verbatim
transcription of the expression it replaced over **every** `(i, r, lb)` for
N = 1..16, roughly 3,000 cases. The selector's domain is finite, so exhaustive
comparison is strictly better than sampling it through a kernel.

## Steps 2-4 re-landed, all verified here

- `DocIdKernel.{cu,cuh}` - built for **sm_90** this time (sm_86 on the laptop);
  `cp_doc_ids_test` passes all 14 cases identically on both.
- `ContextParallel.h` - `doc_local`, defaulted `eot_id = -1`, both sharding
  branches. Passes `-fsyntax-only` against the full project include set.
- `RopeDeltas.h` - `sub_chunk_for`; ring-loop call site substituted.
- `DocMask.h` - `DocIdRing`, `build_doc_id_ring`, `doc_slice_for_step`,
  `validate_doc_ids`.
- `cp_doc_ring_slice_test` passes: selector equivalence, 8 ring configurations
  (N = 1/2/4/8 x both layouts, B > 1), and 4 host-rejection cases.

**Confirmed a plan assumption on real hardware:** `build/objects/context_parallel/DocIdKernel.o`
appears in the parity test's link line without any Makefile edit, which is what
the plan predicted - `CU_SOURCES` globs `context_parallel/*.cu`. It is correctly
absent from `SM90A_SRCS`, so it is not forced to `compute_90a`.

## One self-inflicted false alarm

The backgrounded build reported "failed with exit code 1" while having actually
succeeded. Cause: my own wrapper - a shell variable did not carry into the `tail`
after the `nohup ... &` compound, so `tail` opened `/build_pristine.log` and
failed. `make` itself reported `Done`, the log had zero real errors (the only
`grep -i error` hits were the filename `Error_logs.o` in the link line), and the
binary existed and ran. Reconciled before trusting the result.

## Not done

Steps 5-11 are untouched: kernel signatures including the non-Hopper stubs,
`validate_doc_ids` wiring, the FWD dead-row guard, the FWD element mask plus the
`!diag` fast-path fix, the BWD element mask and its dead-row guard, ring wiring,
and the `CP_DOC_MASK` / `CP_DOC_EOT` flags and gates. No Makefile targets were
added for the two new tests; both are still built by hand.

Scope reminder, unchanged: these kernels are reached only via
`CP_ATTN_MODE=ring CP_ATTN_FUSION=split`. The 16k/32k extensions were trained
with `bluscript_zero_ckpt.cpp`, which does not use them, so this will not
retroactively explain those runs' scores.
