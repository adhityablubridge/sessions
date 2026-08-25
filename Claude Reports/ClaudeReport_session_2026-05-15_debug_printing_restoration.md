# Session Report: Debug Printing Restoration for CP Gradient Sync Analysis

**Date:** 2026-05-15  
**Workspace:** TensorParallelismBeta  
**Context:** Debugging PyTorch Context Parallel training - investigating why non-synchronized gradients yield better GPU saturation (4.0) than synchronized gradients (6.0)

## Objective

Restore and expand debug printing in `gpt2_cp_headtail_fp32.py` to print parameter and gradient tensor values for **all CP ranks at every training step**, enabling visualization of whether gradients are identical or diverging across ranks.

## Problem Statement

- User reported counterintuitive behavior: synchronized gradients (via AllReduce) → 6.0 GPU saturation vs. non-synchronized gradients → 4.0 saturation
- Previous session had debug printing that only printed rank 1 (not rank 0) and only at certain steps
- User explicitly requested full tensor printing (not checksums) for all ranks at all steps to diagnose gradient divergence

## Changes Made

**File:** `/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/Pytorch/gpt2_cp_headtail_fp32.py` (lines 1022-1057)

### Key Modifications

1. **Changed rank gate from single rank to all ranks:**
   - Old: `if cp_rank == 1:` (only rank 1 prints)
   - New: `if cp_rank == r:` where `r` ranges over `range(cp_world_size)` (all ranks print in sequence)

2. **Added parameter name to output:**
   - Added `print(f"\nParam: {name}")` before tensor display for clarity

3. **Updated parameter-less print message:**
   - Added parameter name to "NO GRADIENT" message for better identification

### Design

The loop structure with barrier ensures proper synchronization:
```python
for r in range(cp_world_size):          # Rank 0 prints, all barrier
    if cp_rank == r:                     # Rank 1 prints, all barrier
        print(...)                       # Rank 2 prints, all barrier, etc.
    torch.distributed.barrier(group=cp_group)
```

## Expected Output

When running the training script, each step will now show:
- Full parameter and gradient tensors for all CP ranks (rank 0, then rank 1, etc.)
- Parameter names for easier identification
- Synchronized output with barriers ensuring no interleaved prints

Example structure:
```
=== DEBUG: Parameter Gradients at Step 1 [Rank 0] ===
Param: transformer.wte.weight
Param size: 50257

Param: transformer.wpe.weight
...
[tensor values for rank 0]

=== DEBUG: Parameter Gradients at Step 1 [Rank 1] ===
Param: transformer.wte.weight
...
[tensor values for rank 1 - should match rank 0 if properly synced]
```

## Next Steps for User

1. Run training script with modified debug printing
2. Compare gradient values across ranks at each step to identify:
   - Whether replicated parameters (wte, wpe, ln_f, lm_head) have identical gradients
   - At which step/parameter divergence begins (if any)
   - Whether AllReduce synchronization is the actual problem or a red herring

3. Use insights to determine if:
   - Gradients naturally diverge due to ring-rotated attention computation
   - AllReduce synchronization is masking a deeper issue with loss calculation
   - Load balancing (when enabled) affects gradient synchronization

## Files Modified

- `DTensor/Pytorch/gpt2_cp_headtail_fp32.py` - Debug printing block (lines 1022-1057)

## Logs

- Added entry to Claude Logs.md: "2026-05-15 - Debug printing restored for all CP ranks at every step"
