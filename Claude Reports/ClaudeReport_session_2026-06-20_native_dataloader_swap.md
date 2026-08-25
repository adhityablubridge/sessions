# Claude Report — Native DataLoader Swap + Custom-Impl Audit

- **Date/Time:** 2026-06-20 12:20
- **Work:** Replace hand-rolled `DataLoaderLite` with PyTorch's native `torch.utils.data.DataLoader`, and audit the script for any other non-native ("our own") implementations.
- **Workspace / File:** OldPush / `TensorParallelismBeta/DTensor/Pytorch/gpt2_cp_attnstyle_fp32_dl.py`

---

## Context

`gpt2_cp_attnstyle_fp32_dl.py` is a copy of `gpt2_cp_attnstyle_fp32.py`, the **PyTorch parity reference** used to validate the C++ Context-Parallel GPT-2 build (prior logs 2026-06-05/06-16: cosine 1.0 forward parity, step-0 grad parity). The user wants the PyTorch reference to use PyTorch's **native** data loader instead of the custom `DataLoaderLite`, since the comparison is "PyTorch vs our C++", and wants to know what other hand-rolled components exist.

## Change Made

Replaced `DataLoaderLite` + its usage with native `torch.utils.data` classes:

- `TokenBlockDataset(Dataset)` — exposes one shard's token stream as non-overlapping length-T blocks; item `k` -> `x = tokens[k*T : k*T+T]`, `y = tokens[k*T+1 : k*T+T+1]`.
- `NativeShardLoader` — drives one native `DataLoader(batch_size=B, shuffle=False, drop_last=True)` per shard and cycles shards forever. Keeps the **exact same external contract** (`next_batch() -> [B,T] CPU tensors`, `reset()`), so the training loop is unchanged apart from the two instantiation lines.

### Why batches stay byte-identical (parity preserved)

Old `DataLoaderLite` batch `n` used flat window `tokens[n*B*T : n*B*T + B*T + 1]`, reshaped `x=buf[:-1].view(B,T)`, `y=buf[1:].view(B,T)`. Row `r` of that batch = contiguous block at offset `(n*B + r)*T` with a +1-shifted target. The native version exposes exactly that block as Dataset item `k -> offset k*T`, and `DataLoader(shuffle=False)` regroups B consecutive blocks per batch in order. Identity `floor(floor((N-1)/T)/B) == floor((N-1)/(B*T))` guarantees the same full-batch count per shard; `drop_last=True` reproduces the old "abandon partial tail, advance to next shard" behavior; batches never straddle shards.

- Data: 5 `edufineweb_train_*.bin` + 1 `edufineweb_val_*.bin`, 100M uint16 tokens each. 161M run (~815M tokens) crosses shard boundaries, so per-shard cycling matters.
- `num_workers=0`, `pin_memory=False` by default (constructor args exist to bump them). Kept 0 to avoid changing behavior; the slice is from an in-RAM tensor so workers add no value.

### Verification

- `grep DataLoaderLite` -> only comment references remain.
- `python3 -m py_compile` -> OK (no GPU/torchrun run performed).

## Audit — remaining non-native ("own") implementations

| Component | Location | Native underneath? | Note |
|---|---|---|---|
| `DataLoaderLite`/`load_tokens` | data section | n/a | REPLACED with native DataLoader. `load_tokens` kept (plain numpy I/O). |
| `CPAttention` CP bracketing | ~L206-285 | Math = native `F.scaled_dot_product_attention`; CP wiring is hand-rolled (`_enable_cp_dispatcher` + `DTensor.from_local` + backward hooks) | The "native" CP path is `context_parallel()`, which this file's own docstring documents as BROKEN (dispatcher torn down before backward -> dK/dV dropped). NOT a from-scratch attention; do not swap without deciding. |
| Manual buffer sharding | `GPT.forward` ~L398-425 | Uses PyTorch **private** APIs `_context_parallel_buffers`, `_generate_round_robin_indices` | PyTorch code, but private/experimental. |
| `shard_for_cp` | ~L569-590 | wraps the private API | Currently UNUSED (dead code) — sharding happens inside `GPT.forward`. |
| `get_lr` | ~L706-713 | hand-written cosine+warmup schedule | Standard; must match C++ schedule. No torch native scheduler used. |
| `CudaTimer` | ~L156-165 | wraps native `torch.cuda.Event` | Measurement only; no effect on numerics/parity. |
| `configure_optimizers` | ~L512-523 | native `torch.optim.AdamW` | Only custom param grouping. |
| GPT/MLP/Block model | model section | native nn modules | Hand-written model (normal; no native GPT exists). |

**Most relevant for "compare with PyTorch, not our own":** the data loader (now native) and `CPAttention` (partially custom; cannot trivially become native `context_parallel()` because that path is documented broken).

## Follow-ups / Not Done

- No torchrun execution / loss-curve diff run to empirically confirm parity (syntax + algebraic argument only).
- Decision pending from user on whether `CPAttention` custom CP bracketing should also be considered "non-native" for the comparison.
