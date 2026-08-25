# ClaudeReport
2026-05-09 - Fix broken autograd graph in shard_sequence via CatBackward node - TensorParallelismBeta / CatBackward.h, CatBackward.cpp, ContextParallel.h

## Root Cause
`ContextParallel::shard_sequence` used `Tensor::cat` (non-autograd) and `.contiguous()` (non-autograd), breaking the gradient chain from transformer blocks back to wte/wpe embedding tables. Result: wte/wpe received no gradients, producing a persistent ~0.5 loss gap vs PyTorch CP reference.

Evidence: grad checksum diff showed wte/wpe "only in PT"; backward graph had 40 GradAccumulators (39 expected + 1 dangling for dropped x_local grad); no EmbeddingBackward nodes present.

## Fix

### New files
- `DTensor/Tensor-Implementations/include/autograd/backward/CatBackward.h` — declares `CatBackward : public Node` and `autograd::cat()`
- `DTensor/Tensor-Implementations/src/autograd/backward/CatBackward.cpp` — implements both; `apply()` narrows upstream grad into per-input slices via `Tensor::narrow(cat_dim_, offset, sz)`

### Modified
- `DTensor/gpt2_cp_test/context_parallel/ContextParallel.h` — `shard_sequence` LB path: `Tensor::cat` -> `autograd::cat`; non-LB path: `.contiguous()` -> `autograd::contiguous`

## Verification
New backward graph sequence (misc/GraphNodeBackwardPrint.md):
```
[100] CatBackward
[103] make_shards_inplace_axis_Backward
[104] AddBackward
[105] EmbeddingBackward   (wte)
[106] EmbeddingBackward   (wpe)
[107] GradAccumulator     (wte.weight)
[108] GradAccumulator     (wpe.weight)
```
Chain fully restored. wte/wpe now train.

## Pending
- Re-run DUMP_GRADS=1 to confirm grad checksum parity for wte/wpe
- Run 200-step training curve to confirm loss gap closed
