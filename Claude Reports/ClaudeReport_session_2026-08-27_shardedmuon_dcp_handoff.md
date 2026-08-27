# ShardedMuon: DCP ShardedOptimizerState implementation - handoff to the team

2026-08-27 - Implemented the four DCP shard-spec methods on owt::zero::ShardedMuon,
which the team's ZeRO/DCP update left unimplemented. Workspace: BluTrain /
dist/zero (ShardedMuon.h, ShardedMuon.cpp).

## The problem

The team's update moved both ZeRO-1 implementations onto `ZeroOptimizerBase`, which
now also inherits `OwnTensor::dcp::ShardedOptimizerState` (OptimShardSpec.h). That
adds four pure virtuals. `ShardedOptimizer` (v1) and `SymmetricShardedOptimizer` (v2)
implement all four; `ShardedMuon` implements none, which made it an ABSTRACT class:

    error: invalid new-expression of abstract class type 'owt::zero::ShardedMuon'
      note: dcp_state_shard_specs()  dcp_get_state_shard()
            dcp_set_state_shard()    dcp_after_load()

So `std::make_unique<ShardedMuon>` does not compile on any branch. This is not
CP-specific - it breaks any trainer that instantiates Muon.

## What was implemented

`dist/zero/include/ShardedMuon.h`
  - public: the four `dcp_*` overrides
  - public: `get_config` / `set_config` overrides. These are virtual on
    `nn::Optimizer` and v1 implements them to persist `step_count_`; ShardedMuon
    did not declare them, so a DCP resume would have silently restarted the step
    counter at 0.
  - private: `republish_masters()` and `dcp_slot_for()`
  - added `<map>` / `<string>` includes

`dist/zero/src/ShardedMuon.cpp`
  - the four methods plus `get_config`/`set_config`
  - `load_state`'s republish tail factored into `republish_masters()`, which
    `dcp_after_load()` also calls. No behaviour change to `load_state`; the two
    paths must republish identically and now cannot drift.

## The one real design decision

v1/v2 shard a flat ELEMENTWISE buffer, so each of their state tensors is one
contiguous range of one global flat space, and they report ONE spec per state_type
with `param_fqn = dcp_scope_` and a single `{flat_offset, flat_numel}`.

ShardedMuon owns WHOLE TENSORS. An owner's tensors are scattered through the flat
space rather than contiguous in it, and under `REDUCE_BROADCAST` there is no flat
space at all. A single offset/length pair cannot express that.

So: **one spec per OWNED parameter**, where the "global" state for that spec is just
that parameter - the owner holds all of it, at offset 0 - and ranks that do not own
a parameter emit nothing for it. That is exactly the ownership statement DCP needs
in order to intersect ranges on load.

This is strictly better than the stream format for resharding. `save_state` /
`load_state` pin a reader to the same world size AND the same `MuonCollective`,
because the byte order follows the local owner assignment (hence `layout_hash_`
rejecting a mismatch). Under DCP a state tensor is identified by the PARAMETER, not
by a position in some rank's shard, so a load can reshard onto a different world
size and a different owner assignment.

### Naming contract - please review this specifically

`param_fqn` is `"<scope>.<input_index>"`. The class is constructed from a bare
`std::vector<Tensor>` and never learns parameter names, so the position in that
vector is the only stable identifier available. The contract is therefore that the
model presents its parameters in the SAME ORDER on save and load - the assumption
`load_state` already makes. If the caller later passes real FQNs in, substituting
them at that one site is the only change needed. This is the piece most worth a
second opinion, since a wrong name maps state onto the wrong tensor silently.

### Other details worth knowing

- The second moment (`zero_v`) exists only on the AdamW path, and even there it is
  allocated LAZILY on that parameter's first update (`update_one`). So the spec is
  emitted only when `!use_muon && owned_v_[s].is_valid()`, and
  `dcp_set_state_shard` allocates it when a checkpoint carries `v` for a parameter
  this process has not stepped yet.
- `dcp_get_state_shard` returns a 1-D VIEW. The owned buffers carry the parameter's
  shape (2-D for every matrix Muon orthogonalizes), but a FlatSharded spec declares
  1-D metadata and the loader asks `load_tensor_region` for a 1-D region; without
  the view the saved tensor has rank 2 and the load fails with "Rank mismatch".
  This mirrors the same note in v1.
- State-type strings are deliberately identical to v1's (`zero_master`, `zero_m`,
  `zero_v`) so one archive reads with one vocabulary across all three variants.
- `dcp_get_state_shard` calls `drain_param_refresh()` first, as v1 does, so a
  staged refresh in flight cannot race the bytes being checkpointed.

## Verification

Compile: `ShardedMuon.cpp` and `bluscriptCP.cpp` both `-fsyntax-only` clean;
ShardedMuon is concrete and `make_unique` compiles. Full CP build + link clean.

**Behavioural non-regression** - the bisection the class's own header prescribes
("a disagreement between the two isolates a layout bug in the bin-packing from a
fault in the Muon math itself"), 600M, ZeRO v2 + Muon, before vs after the edit:

| step | np=2 RSAG | np=2 REDUCE_BROADCAST | np=1 |
|---|---|---|---|
| 0 | 11.163398 / 37.6721 | 11.163398 / 37.6718 | 11.163398 / 37.6718 |
| 1 | 11.100425 / 42.0405 | 11.100425 / 42.0400 | 11.100474 / 42.0408 |
| 2 | 11.103305 / 40.7067 | 11.103378 / 40.7065 | 11.103358 / 40.7046 |

Step-0 loss identical to 7 digits across all three; norms agree to ~1e-5 relative.
Unchanged from the pre-edit run, i.e. the edit is behaviourally inert.

**Checkpoint round-trip** (exercises the `republish_masters()` refactor; this is the
FIRST time ShardedMuon's checkpoint path has been run at all):

| | 4 steps straight | 2 steps -> save -> resume -> 4 |
|---|---|---|
| step 2 | 11.103245 / 40.7074 | 11.103302 / 40.7076 |
| step 3 | 11.029182 / 36.1563 | 11.029196 / 36.1568 |

Resume reported `[Resume] run 89 from step 2` and fast-forwarded the loader 4
batches. Agreement ~1e-5, i.e. run-to-run fp noise.

## NOT verified

- The DCP path itself was not executed end to end. Nothing in bluscriptCP reaches
  these methods yet: it still checkpoints through `CheckpointManager` and the
  per-group stream API, and the real DCP entry points (`DistributedCheckpointing`,
  `DistCheckpointManager`) take an `OwnTensor::dnn::DModule&`, which CP's plain
  `nn::Module` model is not. So the four methods are compile-verified and
  logic-reviewed but not yet round-tripped through DCP. Someone driving a DModule
  should confirm the spec shape - particularly that per-parameter `param_fqn` with
  `global_numel == flat_numel` is read the way intended.
- The header's claim that Muon is "bit-identical to single-process Muon" is still
  unverified. The bisection above tests internal consistency (two collective modes,
  two world sizes); it does not compare against the in-tree `nn::Muon`.

## Files / backups

Modified: `dist/zero/include/ShardedMuon.h`, `dist/zero/src/ShardedMuon.cpp`.
Backups: `*.preDCP.bak` alongside each.
Also: `dist/Context_Parallelism/Makefile` gained the include paths the update now
requires - `Distributed_Checkpointing/include` (hard requirement, since
ZeroOptimizerBase.h includes OptimShardSpec.h), plus `Tensor-Parallelism/dnn`,
`Tensor-Parallelism/tensor` and `third_party/cufile/Include` (note: capital I on a
case-sensitive filesystem).

---

# Addendum: CP checkpointing fixed at W>1 (stream path)

The bug: `bluscriptCP.cpp` gated the save on `is_master`, while its own comment
said "Loads run on EVERY rank ... so each rank restores its own shard; saves are
master-only to avoid a write race." Under ZeRO those two halves contradict. Only
rank 0's slice was ever written, so every rank then loaded rank 0's bytes as
though they were its own:

  - v1 / v2 (flat contiguous shards): SILENT. Rank r loads [0,S) into the buffer
    that owns [r*S,(r+1)*S).
  - ShardedMuon: throws, because load_state verifies the owned record indices.

Either way resume at W>1 was broken.

Fix (4 sites in bluscriptCP.cpp):
  1. `CheckpointManager(..., shard_dir=cfg.zero, ...)` -- the manager appends
     rank_<N>, so every rank saves and loads its own shard. Non-ZeRO stays
     master-only (its optimizer state is replicated, so one copy is the truth).
  2. Save condition `is_master` -> `(cfg.zero || is_master)`. No race: each rank
     writes its own directory.
  3. + 4. The g>=1 ZeRO sidecars follow the .ckpt into the rank subdirectory via a
     new `zero_side_dir`, or they would collide across ranks in the parent.

A REGRESSION this introduced and then fixed: the resume resolver scanned only
`cfg.ckpt_dir`, non-recursively, so once the .ckpt files moved into rank_<N>/ it
found nothing and every `CP_CKPT_RESUME` silently became a fresh run (observed:
"[Resume] CP_CKPT_RESUME=91 has no checkpoint; starting a new run"). The scan is
now a lambda applied to BOTH the parent and the rank subdirectory, which also
keeps checkpoints written before this change resumable.

Cost: W copies of the (replicated) model weights per checkpoint. Correctness of
the optimizer state is worth more than the bytes.

## Verification: np=2, ZeRO v2 + Muon, 600M

Written (previously only rank 0's):

    checkpoints_bluscriptcp/rank_0/blumodelcp_run91_step_2.ckpt
    checkpoints_bluscriptcp/rank_0/blumodelcp_run91_step_2.zero_g1
    checkpoints_bluscriptcp/rank_1/blumodelcp_run91_step_2.ckpt
    checkpoints_bluscriptcp/rank_1/blumodelcp_run91_step_2.zero_g1

| | 4 steps straight | 2 -> save -> resume -> 4 |
|---|---|---|
| step 2 | 11.103301 / 40.7085 | 11.103344 / 40.7086 |
| step 3 | 11.029268 / 36.1552 | 11.029279 / 36.1565 |

`[Resume] run 91 from step 2`, loader fast-forwarded 2 batches, no missing-sidecar
warnings. Agreement ~1e-5, i.e. run-to-run fp noise.

Limitation, unchanged and inherent to the stream format: same world size only, and
same MuonCollective (layout_hash_ rejects a mismatch). Resharding needs DCP.

Backup: /tmp/bcp.pre_ckptfix.bak

---

# Decision: DCP integration deferred (2026-08-27)

CP stays on its NATIVE checkpoint path -- its own fork of `CheckpointManager` at
`dist/Context_Parallelism/checkpointing/Checkpointing.h`, with the per-rank
`shard_dir` fix above. Confirmed against the built binary: 0 symbols matching
DistCheckpoint / DistributedCheckpointing / DefaultPlanner / GdsStorage, 9
CheckpointManager symbols, and ShardedMuon's four `dcp_*` methods exported but
never called.

Accepted limitation: resume requires the SAME world size and the SAME
MuonCollective. Weights survive a world-size change (they are replicated and
saved in full); optimizer momentum/variance does not. Given this project has
already lost two instances mid-run, note that recovering on a different GPU count
means restarting the optimizer state, not the weights.

The ShardedMuon `dcp_*` work is unaffected by the deferral and still belongs
upstream -- it makes the class constructible at all, which blocks any Muon run on
any branch regardless of checkpoint strategy.
