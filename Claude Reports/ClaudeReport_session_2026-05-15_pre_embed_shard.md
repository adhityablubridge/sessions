2026-05-15 - Pre-embedding sequence shard (contiguous + HeadTail) added for Context Parallel, matching PyTorch's `context_parallel()` buffer pre-sharding - Workspace: OldPush/TensorParallelismBeta/DTensor, Files: gpt2_cp_test/context_parallel/ContextParallel.h, gpt2_cp_test/gpt2_cp_test.cpp

## Summary
- Added `ShardedInputs` struct and inline `shard_sequence_pre_embed()` helper in `ContextParallel.h`. Helper returns this rank's `[B, T/n]` idx/y slices and `[1, T/n]` pos indices.
  - `load_balance=false` -> contiguous chunk via `make_shards_inplace_axis`.
  - `load_balance=true`  -> HeadTail permutation (`out[2k]=k, out[2k+1]=T-1-k`) then take this rank's slice, applied via `OwnTensor::gather` with an int64 index (sidesteps float-only HeadTail kernel).
- `GPT::forward` now shards `idx` BEFORE the embedding lookup; removed the post-embedding `make_shards_inplace_axis` block. Generation/`cp_unshard` paths unchanged.
- Validation and training-loop loss paths now call the same helper to produce `y_local`, replacing the inline `y_in.make_shards_inplace_axis` chunking. Ensures `y` permutation matches the model's sharded logits in both modes.
- Build verified: `make gpt2_cp_test` -> SUCCESS.

## Known limitation (documented in plan, not fixed here)
HeadTail mode runs end-to-end with correct shapes and gradient all-reduce, but the existing ring-attention causal mask in `ContextParallel.h` assumes contiguous per-rank chunks. With HeadTail each rank owns interleaved global positions, so causal-attention loss will be wrong until the ring loop is upgraded to the 3-mask sub-chunk scheme (SKIP / CAUSAL / NOT_CAUSAL over head-half and tail-half sub-chunks) that PyTorch uses. Contiguous mode (default, `config.load_balancing=false`) is fully correct.

## Files Modified
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/gpt2_cp_test/context_parallel/ContextParallel.h`
- `/home/blu-bridge25/TP/OldPush/TensorParallelismBeta/DTensor/gpt2_cp_test/gpt2_cp_test.cpp`
