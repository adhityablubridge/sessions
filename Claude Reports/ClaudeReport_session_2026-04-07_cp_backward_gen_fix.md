# Session 2026-04-07 — CP Backward Graph Fix + Generation Shape Fix

## TensorParallelismBeta / gpt2_cp_test.cpp, ContextParallel.h

---

## Bug 1: `bwd: 0.0ms, norm: 0.0000` — No Backward Graph

### Root Cause
`loss = loss * rank_scale` used the raw `operator*` (from `TensorOps.h`) which performs element-wise multiply without registering any autograd node. This silently dropped `loss.grad_fn`, making loss a leaf tensor. `loss.backward()` returned immediately — zero timing, zero gradients.

### Fix
```cpp
// BEFORE (breaks backward graph):
loss = loss * rank_scale;

// AFTER (preserves grad_fn):
loss = autograd::mul(loss, rank_scale);
```

---

## Bug 2: `.contiguous()` Drops `grad_fn` in Forward Graph

### Root Cause
`Tensor::contiguous()` creates a new tensor via `Tensor(sizes, dtype, device, requires_grad)` — this copies `requires_grad` but does NOT copy `grad_fn`. Any call to `.contiguous()` on a tensor with a `grad_fn` (e.g. a shard from `make_shards_inplace_axis`) breaks the gradient chain.

Affected sites:
- `gpt2_cp_test.cpp` line 453: `x = x_chunks[rank_].contiguous()` after sharding embeddings output
- `ContextParallel.h` lines 111-113: `q_work = q.contiguous()` etc.

### Fix
Use `autograd::contiguous()` (from `ReshapeOps.h`) which wraps the op in a `ContiguousBackward` node:
```cpp
// gpt2_cp_test.cpp
x = autograd::contiguous(x_chunks[rank_]); // [B, T/n, C]

// ContextParallel.h
Tensor q_work = autograd::contiguous(q);
Tensor k_work = autograd::contiguous(k);
Tensor v_work = autograd::contiguous(v);
// pre_sharded=true path: local_q = q_work (already contiguous, no extra op)
local_q = q_work;
local_k = k_work;
local_v = v_work;
```

---

## Bug 3: `Shapes are not broadcastable` at Generation Step 100

### Root Cause
During generation, `GPT::forward` skips sharding (`is_in_generation_mode_=true`), so `x` is full `[B, T, C]`. But `CPAttention` always passed `pre_sharded=true` to `forward_cp`, meaning it treated the full-T q/k/v as already-sharded T/n chunks. With `unshard_=true`, `T_out = T * world_size_ = 2T`, so `proj` came out `[B, 2T, C]` while `x` was `[B, T, C]` — unbroadcastable residual.

### Fix
Added `generation_mode_` bool to `CPAttention`. In generation mode:
- `pre_sharded=false` — CP shards q/k/v internally (T → T/n per rank)
- `cp_unshard=true` — CP all-gathers back to full T after ring attention
- `T_out = T` — output matches input shape, residual works

```cpp
// CPAttention::forward
bool pre_sharded = !generation_mode_;
int64_t T_out = generation_mode_ ? T : (unshard_ ? T * world_size_ : T);
bool cp_unshard = generation_mode_ ? true : unshard_;
Tensor attn_out = cp_->forward_cp(q, k, v, cp_unshard, pre_sharded);
```

`GPT::set_generation_mode` now also calls `a->set_generation_mode(gen)` on each attention block.

---

## Result
- `bwd: 834ms`, `norm: ~0.8` — backward graph restored, gradients flowing
- Loss decreasing normally from ~10.9 to ~8.5 at step 100
- Generation at step 100 no longer crashes
