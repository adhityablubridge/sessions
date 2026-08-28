# Claude Report - bluscriptCP <- bluscript_zero parity pass (Muon + diagnostics)

2026-08-26 - Diffed bluscriptCP.cpp against the team's updated bluscript_zero.cpp
(1213 -> 1495 lines) and ported the gaps: Muon via ShardedMuon, per-group WSD LR
pairs, step ordering, a grad-norm bug of mine, and the CP_DIAG per-micro
diagnostic - CP (/home/blu-bridge25/CP) / Scripts/Blutrain/bluscriptCP.cpp, Makefile

Laptop session: code only. NOT built or run - no CUDA/libtensor build here. Syntax
verified with mpic++ -fsyntax-only against all real headers (rc=0, no errors).

## Two findings that kill earlier hypotheses

**1. The learning-rate hypothesis is dead.** Last session I proposed our bf16 gap
came from running 6e-4 against zero's 2e-4. zero has since moved to
**max_lr 1e-3 / min_lr 1e-4** -- nearly 2x OUR peak -- and trains bf16 fine. A
too-high peak is not the explanation. Good thing this was tested before acting.

**2. Muon is not the stabiliser either.** BLU_OPTIMIZER defaults to AdamW; Muon is
opt-in. And both reduce knobs (BLU_FP32_REDUCE / BLU_BF16_REDUCE) default OFF, so
zero reduces the bf16 group in bf16 exactly as we do. So zero's DEFAULT recipe is
bf16 + AdamW + ZeRO-1 + bf16 reduce at 1e-3, which works for the team.

Our +2.8 nats therefore remains UNEXPLAINED and is something specific to
bluscriptCP. Also checked and cleared: ZeroConfig's overlap_backward_comm and
overlap_param_refresh both default FALSE, so my not setting them was benign (no
overlap, just slower) rather than the silent corruption I suspected.

## Ported

| item | detail |
|---|---|
| Muon | owt::zero::ShardedMuon, CP_OPTIMIZER=muon. Shards by WHOLE PARAMETER, because Newton-Schulz is a whole-matrix operator a flat slice would cut through the middle of. |
| partition | ndim>=2 AND not wte/lm_head -> Muon; embedding + all 1-D scales -> elementwise ZeRO-1/AdamW. Under weight tying lm_head IS wte, so one exclusion covers both. |
| why the split | ShardedMuon owns whole tensors, so a group that is one ~77M embedding plus a tail of tiny scales cannot balance - one rank would hold nearly all state. Elementwise sharding splits the embedding evenly; AdamW can be sharded that way, Muon cannot. |
| MuonConfig | all 20 fields verified present in ShardedMuon.h before use: target_buckets, lr, momentum, weight_decay, ns_steps, nesterov, adjust_lr, adamw_*, exclude_1d_from_weight_decay, max_grad_norm, reduce_dtype, init_broadcast, verify_consistency, verbose |
| per-group LR | each group rides the SAME WSD curve with its OWN (peak, floor) pair, as zero does - NOT one curve scaled by a constant (equivalent only while both floors are the same fraction of their peak) |
| CP_MUON_ADJUST_LR | =original for modded-nanogpt convention. No single lr converts between conventions: the ratio is 0.2*sqrt(cols), spanning ~4.5x (wK/wV) to ~18x (gate_up) on OUR shapes, so switching means re-sweeping muon_lr. Reported at startup. |
| step order | ascending numel, biggest group LAST. Its staged refresh broadcasts issue at the end of step(); another group stepping after would enqueue a blocking norm all-reduce behind them on the same in-order stream and destroy the overlap. |
| CP_MAX_LR / CP_MIN_LR | were hardcoded 6e-4/6e-5 and therefore untestable |
| CP_ZERO_BF16_REDUCE | mirror knob + mutual-exclusion abort (same field, opposite values). Bisection only: saves neither time nor memory. |
| CP_DIAG | per-micro-batch losses with min/max/spread. Directly aimed at our symptom - the mean over 32 micros dilutes one bad micro 32x while its gradient dominates the norm almost undiluted, which is exactly how a modest loss bump sat next to a 15x norm spike at steps 108/137. |

## A bug of mine, found by reading zero

My ZeRO port reported the step's gradient norm as **max over groups**. zero uses
**sqrt(sum of squares)**. Max under-reports a spike confined to one group, which is
precisely the signal I was relying on as the divergence early-warning. Fixed.

## NOT ported, deliberately

- **Staged parameter refresh** (set_param_refresh_stages / refresh_param_stage /
  drain_param_refresh + forward pre-hooks, LAYERS_PER_STAGE=1). Performance only,
  and it needs per-block forward hooks CP's GPT does not wire. Both overlap flags
  stay false, which is ZeroConfig's default, so behaviour is correct just not
  overlapped.
- **micro_where** (shard index + token offset per micro-batch). CP's `Batch`
  (Data_Loader/DataLoader.hpp:118) carries neither field; adding them is a change
  to the shared loader, out of scope for this pass.
- **Depth probe** (layer_gnorm / gamma_absmax / w_absmax per layer). Cheap and
  relevant to the bf16 hunt, but needs model-internal access; flagged as the
  obvious next diagnostic if CP_DIAG alone does not localise the gap.
- **Defaults left alone**: our max_lr stays 6e-4 (zero: 1e-3) and B/global_batch
  stay ours. Different model size and a running experiment; the env vars now make
  them testable rather than silently changed.

## Verification done

- mpic++ -fsyntax-only with every real include path: **rc=0, no errors**
- all 20 MuonConfig field names grepped out of ShardedMuon.h before use
- ShardedMuon confirmed to derive from ZeroOptimizerBase (so the existing
  std::vector<unique_ptr<ZeroOptimizerBase>> holds it and CheckpointManager still
  works), ctor signature (params, pg, cfg) confirmed
- Makefile: ShardedMuon.cpp added to the zero source list
- brace balance checked

## Addendum - the team's intended invocation needed a THIRD change

The team runs `BLU_OPTIMIZER=muon BLU_ZERO_V2=1 BLU_SYMMETRIC=1`. I had ported the
first two; BLU_SYMMETRIC was missing, and it is NOT a ZeroConfig field:

**It is a MemoryMode argument to init_process_group, and NCCL fixes the memory mode
at process-group creation** -- so it cannot be enabled later from the optimizer
config. bluscript_zero reads it before init_process_group for exactly this reason.

CP creates its PG through DeviceMesh, and `device_mesh.cpp:140` called
`init_process_group(total_devices_, global_rank_)` with no MemoryMode -- hardwired
to Standard. So symmetric memory was UNREACHABLE in CP regardless of env.

Plumbed it:
- `process_group/device_mesh.h`: 3rd ctor param `dist::comm::MemoryMode mem_mode =
  Standard` (default keeps every existing caller working -- gpt2_cp_test.cpp and
  four Tests/*.cpp pass two args), new `mem_mode_` member + `memory_mode()`
  accessor, explicit `#include "MemoryMode.h"` so the default arg does not depend
  on the transitive chain through CPProcessGroupNCCL.h.
- `process_group/device_mesh.cpp`: capture into `mem_mode_` in the ctor and consume
  it in `initialize_process_groups()`, which is where init_process_group actually
  runs (my first attempt put it in the ctor, where mem_mode was out of scope).
- `bluscriptCP.cpp`: `CP_ZERO_SYMMETRIC` + `CP_MUON_RB`, read BEFORE the mesh is
  constructed, with zero's "has NO effect" warning: symmetric only helps when
  `zero_v2 || (muon && !muon_rb)`, because NCCL ships symmetric kernels for
  AllGather/AllReduce/ReduceScatter but NONE for Reduce or Broadcast -- which are
  exactly what ZeRO v1 and Muon's REDUCE_BROADCAST fallback use.
- `CP_MUON_RB=1` -> `mcfg.collective = MuonCollective::REDUCE_BROADCAST`.

Name corrections caught by checking the header instead of guessing: the enum is
`MuonCollective` (singular) and the field is `collective`, not `MuonCollectives` /
`collectives`.

**A mistake I made and fixed:** a conditional in my patch script ran
`s.replace('class DeviceMesh {','')` and DELETED the class declaration from
device_mesh.h. Caught immediately by grep (count 0) and restored. Lesson: never put
a speculative replace behind a condition in a bulk patch script.

### Translation of the team's command

    BLU_OPTIMIZER=muon BLU_ZERO_V2=1 BLU_SYMMETRIC=1        (zero)
    CP_OPTIMIZER=muon  CP_ZERO_V2=1  CP_ZERO_SYMMETRIC=1    (ours)

### Verification

- `bluscriptCP.cpp` syntax: rc=0
- `device_mesh.cpp` syntax: rc=0
- MuonConfig / MuonCollective / MuonAdjustLr names all grepped from ShardedMuon.h
  before use

## Next

1. Build on the instance: `make CP_FUSED_ROPE=1 CP_ATTN_SPLIT=1 bluscript-cp -j24`
2. Parity check FIRST: CP_OPTIMIZER unset must reproduce the previous AdamW
   numbers exactly (step-0 loss 11.148983 at NP=4 bf16+ZeRO).
3. Then CP_DIAG=1 for ~150 steps on the bf16 AdamW config to see whether the +2.8
   nats is one bad micro-batch per step or a uniform shift - that distinction picks
   the next move.
4. Muon arm: CP_OPTIMIZER=muon with muon_lr 1e-3.
