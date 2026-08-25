2026-06-23 - Generalized sequence length T (block_size no longer hardcoded to 1024) - Workspace: OldPush, File: TensorParallelismBeta/DTensor/Pytorch/gpt2_cp_attnstyle_fp32.py

## Problem
T was capped at 1024 because GPTConfig.block_size=1024 fixed the wpe positional
embedding table to 1024 rows. Setting T=2048/4096 made pos=arange(0,T) index past
wpe -> device-side assert / index out of range at forward (wpe lookup).

## Change
- block_size now derived: _block_size = int(env BLOCK_SIZE default max(1024, T)).
- Passed block_size=_block_size into both GPTConfig constructors (44M and 161M).
- Added fast asserts with messages: T % cp_world_size == 0, and
  total_batch_size % (B*T) == 0 (grad_accum integer math), plus _block_size >= T.

## Caveats
- Changing block_size changes wpe shape -> different param set; NOT parity-compatible
  with init_weights_named_*.bin or the C++ reference (both assume 1024).
- Attention is O(T^2): 1024->4096 ~16x SDPA cost/mem; may need smaller B.
- For parity workflows keep T=1024 or BLOCK_SIZE=1024 with T<=1024.
