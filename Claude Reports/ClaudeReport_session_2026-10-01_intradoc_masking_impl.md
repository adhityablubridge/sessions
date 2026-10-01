# Intra-document masking for the CP Hopper attention kernels - implementation, steps 1-4

2026-10-01 - Built and verified the first four landing steps of the intra-doc masking plan (doc-id scan, sharding plumbing, ring slicing, two no-GPU tests) on the local RTX 3060 laptop - Workspace CP / context_parallel/DocIdKernel.{cu,cuh}, context_parallel/DocMask.h, context_parallel/ContextParallel.h, context_parallel/RopeDeltas.h, Tests/cp_doc_ids_test.cpp, Tests/cp_doc_ring_slice_test.cpp

---

## What this session did

Implemented landing-order steps 2, 3 and 4 of the approved plan
(`.claude/plans/wild-chasing-stroustrup.md`). These are deliberately the three
steps that need **neither Hopper nor MPI**, so all of it was built and run to
completion on this laptop. Step 1 (golden dump) and steps 5-11 (kernel
signatures, the FWD/BWD masks, the ring wiring, flags) still need the H200.

Nothing is wired into the attention kernels yet. `CP_DOC_MASK` does not exist
yet. The default path is unchanged and no training run behaves differently.

## Step 2 - `DocIdKernel` (new)

`context_parallel/DocIdKernel.{cu,cuh}`. Block-per-row tiled **exclusive** scan
of `tok == eot_id` along T, with a running block-uniform carry.
`DOCID_BLOCK = 256`; warp-shuffle inclusive scan, shared-memory scan of the warp
aggregates, converted to exclusive by subtracting the element's own flag.

Two alternatives were rejected in the header comment rather than silently:
one-thread-per-row (B is 1-8, so a single-warp kernel doing up to 128K dependent
loads) and CUB `DeviceScan` (correct, but a device-wide scan, a temp allocation
and tile-order nondeterminism for a few-million-element problem).

Launchers: `launch_doc_ids_i64` (the loader's Int64 token tensor) and
`launch_doc_ids_u16` (the raw UInt16 buffer). Built for **sm_86** -
`DocIdKernel.cu` is deliberately **not** in the Makefile's `SM90A_SRCS`, which
would force `compute_90a` and break the dev-box build.

### The exclusive convention, and why it is load-bearing

Exclusive scan means an EOT token carries the id of the document it
**terminates**. That is not a style choice:

1. The EOT query attends to all of its own document, so the model trains on
   "given a complete document, this is what completion looks like". Under an
   inclusive scan the EOT position's only same-doc key is itself and the row
   degenerates to a singleton softmax.
2. It guarantees every query has at least one unmasked key at ring step `i == 0`
   (itself, via the causal diagonal). `SDPAMerger`'s online combine computes
   `new_lse = accum_lse - log(sigmoid(accum_lse - block_lse))`, which is **NaN**
   when `accum_lse` is `-INF`. The accumulator is seeded by the first merged
   step, which is always `i == 0`. So this convention is the only thing standing
   between doc masking and NaN loss.

## Step 3 - sharding plumbing

`ShardedInputs` (`ContextParallel.h`) gains `Tensor doc_local` (Int32, invalid
unless enabled). `shard_sequence_pre_embed` gains a defaulted `int eot_id = -1`;
`-1` means no kernel launch and no allocation, which is what every existing
caller gets.

Ids are derived from the **full** sequence and then carried through the **same**
permutation as `idx_local` - never scanned from `idx_local`. A shard-local scan
restarts numbering at 0 on every rank, so rank r's query ids and rank s's key ids
stop being comparable and the `doc_q == doc_k` test the kernel performs becomes
meaningless.

Both sharding branches are wired, and they take structurally different routes:

- `load_balance=false` uses `make_shards_inplace_axis(..., axis 1)` and has no
  gather index. Its result **must** be `.contiguous()` - slicing a non-leading
  axis returns a strided view, and the kernel reads doc ids as flat `[B, T]`
  memory, so a view makes every `b > 0` row read row 0's data. That is
  accidentally correct at `B == 1`, which is how it would survive testing and
  corrupt training.
- `load_balance=true` reuses the existing `gather_idx`, so `doc_local[b][j]` is
  by construction the id of the token at `idx_local[b][j]`. `gather` dispatches
  on the input dtype (Int32 is covered on both the CPU and CUDA paths, verified)
  and allocates a fresh row-major result.

Guards added: Int64 input required, contiguous input required (the scan indexes
`[b*T + t]` directly, so a strided view would scan the wrong elements rather
than fail).

## Step 4 - `sub_chunk_for` and `DocMask.h`

**`sub_chunk_for(i, r, lb)`** moved into `RopeDeltas.h` and the inlined copy in
the forward ring loop (`ContextParallel.h`) replaced with a call. Identical truth
table, verified line by line. The reason to extract it: the backward pass and the
document mask both need the same answer, and three hand-copies of a three-way
conditional is exactly how forward and backward end up masking different sets of
positions - a bug that surfaces only as slightly wrong gradients.

**`context_parallel/DocMask.h`** (new) is the single slicing authority:

- `DocIdRing` - per-rank shard-local ids, **globally** numbered.
- `build_doc_id_ring(doc_full, N, lb)` - builds every rank's view locally, no
  communication. Valid because every CP rank in a group sees the same tokens
  (sharding is the DP axis; `loader_rank = dp_rank`). The permutation comes from
  `local_to_global_pos`, which `RopeDeltas.h` already declares the single source
  of truth, so this adds no fourth copy of the HeadTail arithmetic (it already
  exists in `shard_sequence_pre_embed`, `headtail_kernel.cu` and that helper).
- `doc_slice_for_step(ring, r, i, N, lb)` - driven by `rope_source_rank` and
  `sub_chunk_for`, the same two calls `compute_deltas` uses, so the doc slice and
  the RoPE deltas cannot drift apart. Returns contiguous tensors always.
- `validate_doc_ids` - rejects rank != 2 (a `[B, Hq, T]` array), dtype != Int32
  (an Int64 tensor read through `data<int>()` is half-stride garbage, not a type
  error), non-contiguous, and wrong length.

Checked while implementing: `Tensor::contiguous()` returns `*this` only when
`is_contiguous() && storage_offset() == 0`, so both the head half (offset 0) and
the tail half (offset T/2) materialise correctly.

## Tests - both run with no GPU work beyond a single device

### `Tests/cp_doc_ids_test.cpp` - 14 cases, all pass

No EOT, EOT at 0, EOT at end, consecutive EOTs, per-row-distinct patterns,
T = 1/255/256/257/1023/1024/1025/65536, all-EOT. Asserts against a CPU reference
plus three structural invariants.

**One invariant was wrong on the first run and the kernel was right.** I had
written the convention as `tok[p] == EOT => doc[p] == doc[p-1]`, which only holds
when `tok[p-1]` is not EOT: with two EOTs in a row, the second terminates an
*empty* document and legitimately gets a fresh id. Replaced with the recurrence

    doc[t] == doc[t-1] + (tok[t-1] == eot)

which is the exclusive scan written directly, and which separates exclusive from
inclusive at **every** position rather than only at EOT tokens. Strictly stronger
than what the plan specified.

### `Tests/cp_doc_ring_slice_test.cpp` - no GPU, no MPI

All `(N, r, i)` for N = 1, 2, 4, 8, both sharding layouts, B > 1 throughout.
Asserts elementwise that each rank's shard ids are the global ids of the tokens
it holds, that the K slice comes from `rope_source_rank(r, i, N)`, that
`QTailHalf` picks half **[1]**, and that every returned tensor is contiguous.
Plus four host-rejection cases.

This replaces the plan's original CP_SIZE=1-vs-2 loss comparison and is strictly
stronger: the CP ring is not bit-identical across CP sizes anyway (bf16 plus
online-softmax merge ordering), so a loss comparison can only assert "close
enough" under a tolerance nobody can derive, and a handful of wrongly-masked rows
disappears under it.

### Mutation testing - the gates were verified to actually fire

A regression gate that has never been seen to fail is only an assertion. Four
mutants were injected and the originals restored:

| Mutant | Caught by |
|---|---|
| kernel scan flipped to inclusive | 12 failures (reference check) |
| kernel **and** CPU reference both flipped to inclusive | 12 failures (invariant 3 alone) |
| `QTailHalf` takes the head half | 3 failures |
| K slice reads rank `r` instead of source rank `s` | 6 failures |
| per-rank-local ids (shard base subtracted) | 6 failures |
| unmutated control | 0 failures |

The second row is the one that matters: it proves the convention invariant stands
on its own, not merely because the CPU reference happens to agree.

## Verification performed

- `DocIdKernel.cu` + its test compile and run clean for **sm_86**.
- `ContextParallel.h` passes `-fsyntax-only` with the project's full include set
  after every edit (including the `sub_chunk_for` substitution).
- `cp_doc_ring_slice_test` links against `libtensor` and passes all 8
  configurations plus all 4 rejection cases.

## Not done / still open

- Step 1, the golden dump of today's kernel, needs the H200 and must be captured
  **before** the signature change so the `doc = nullptr` path can be byte-compared.
- Steps 5-11: kernel signatures (including the non-Hopper stubs, or every
  `SM_ARCH=86` build breaks), the FWD dead-row guard, the FWD element mask plus
  the `!diag` fast-path fix, the BWD element mask and its own dead-row guard, the
  ring wiring, and the `CP_DOC_MASK` / `CP_DOC_EOT` flags with their gates.
- No Makefile targets were added for the two new tests yet; both were built by
  hand this session.
- Scope reminder from the plan, unchanged: these kernels are reached only via
  `CP_ATTN_MODE=ring CP_ATTN_FUSION=split`, and the 16k/32k extensions were
  trained with `bluscript_zero_ckpt.cpp`, which does not use them. This will not
  retroactively explain those runs' scores.
