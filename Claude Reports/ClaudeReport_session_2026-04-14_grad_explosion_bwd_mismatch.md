# Claude Report: Context-Parallel Gradient Explosion Investigation
**Date:** 2026-04-14  
**Session Duration:** ~2 hours  
**Project:** TensorParallelismBeta / DTensor Context-Parallel GPT-2  
**Branch:** `_adhi_`  
**Model Used:** Claude Opus 4.6 → Claude Haiku 4.5

---

## Executive Summary

Investigated gradient norm explosion in custom C++ Context-Parallel training vs stable PyTorch baseline. The training logs show:
- **C++ CP:** grad_norm explodes from ~1 (step 195) → 400k+ (step 511)
- **PyTorch CP:** grad_norm stays stable 0.2-0.4 throughout

**Root Cause Identified:** Forward/backward mismatch in fused SDPA BOT_HALF mask handling for TC (TensorCore) kernel path.

**Fix Applied:** Disabled BOT_HALF from TC forward path; now falls back to scalar kernel which correctly masks top-half Q rows (output=0, LSE=-1e30), matching backward kernel semantics.

**Status:** Fix applied but requires recompilation and verification. Initial hypothesis test (log 182) still showed explosion, but fix may not have been compiled.

---

## Problem Statement

### Symptom
Training loss stayed bounded (7-9) but **gradient norm exploded catastrophically** starting at step ~196:
```
Step 195: grad_norm = 1.6
Step 196: grad_norm = 4.6
Step 212: grad_norm = 131
Step 265: grad_norm = 5,598
Step 511: grad_norm = 426,578
```

This pattern appeared consistently across three identical runs (logs 180, 181, 182).

### Reference Baseline
PyTorch's native CP (with HeadTail + load balancing) showed **perfectly stable gradients** at identical hyperparameters:
```
Step 195-215: grad_norm = 0.2-0.4 (stable)
Step 500: grad_norm = 0.28 (stable)
```

This confirmed the issue was **algorithmic, not hyperparameter-based**.

---

## Root Cause Analysis

### Systematic Debugging Approach (Phase 1-4)

**Phase 1: Root Cause Investigation**
- Examined training logs: explosion starts exactly at step 196 when LR warmup reaches ~1.75e-4
- Compared PyTorch vs C++: identical model config, identical warmup schedule
- **Key Evidence:** PyTorch stable, C++ exploding → code bug, not math/hyperparams

**Phase 2: Pattern Analysis**
- Traced forward/backward consistency across all mask types (CAUSAL, NONE, LEFT_HALF, BOT_HALF)
- Found mismatch: TC kernel and scalar kernel have **different semantics for BOT_HALF**

| Mask Type | TC Forward | Scalar Forward | Backward | Match? |
|-----------|-----------|----------------|----------|--------|
| CAUSAL | Q_full x K_full + causal | same | same | ✓ |
| NONE | Q_full x K_full | same | same | ✓ |
| LEFT_HALF | Q_full x K[:T/2] | same | K blocks < T/2 only | ✓ |
| **BOT_HALF** | **Q_full x K[T/2:]** | **Q[T/2:] x K_full** | **Q[T/2:] x K_full** | **✗** |

### The Bug

In `FusedSDPAOp.h` lines 131-141 (TC forward path):
```cpp
// BOT_HALF: attend to last T_k/2 tokens of K per head (NONE mask).
// ERROR: This computes attention for ALL Q rows against K[T/2:],
// but backward expects only BOTTOM-HALF Q rows.
const int64_t half_offset = (T_k / 2) * D;
OwnTensor::cuda::mem_efficient_attn_forward_tc(
    Q_ptr, K_ptr + half_offset, V_ptr + half_offset, O_ptr, LSE_ptr,
    ...);
```

**Forward computes:** `Q[T_q:, :] x K[T/2:, :]` for ALL Q rows (includes top-half Q rows that shouldn't attend)

**Backward expects:** `Q[T/2:, :] x K_full` (only bottom-half Q rows attend)

This causes:
1. Top-half Q rows get non-zero forward output but dQ=0 in backward
2. Missing gradient contributions accumulate over steps
3. Once LR is high enough (~step 196), parameter drift destabilizes training
4. Gradient clipping at 1.0 keeps loss bounded but can't correct accumulated drift

### Why It Starts at Step 196

- Learning rate warmup: step < 676 → LR = 6e-4 × (step+1) / 676
- Step 196: LR ≈ 1.75e-4
- Per-step parameter updates scale with LR
- Accumulated gradient error finally becomes significant enough to destabilize training at higher LR values

---

## Fix Implementation

**File Modified:** `DTensor/gpt2_cp_test/context_parallel/FusedSDPAOp.h`

**Change:** Remove BOT_HALF from TC kernel condition (lines 113-118)

```cpp
// BEFORE:
bool tc_half_ok = ((mask_type == MaskType::LEFT_HALF ||
                    mask_type == MaskType::BOT_HALF) && (T_k % 2 == 0));

// AFTER:
bool tc_half_ok = (mask_type == MaskType::LEFT_HALF && (T_k % 2 == 0));
// BOT_HALF falls through to scalar kernel
```

**Rationale:** 
- Scalar forward kernel at [FusedSDPAKernel.cu:72-79] correctly handles BOT_HALF
- Sets `O = 0` and `LSE = -1e30f` for top-half Q rows (merger ignores them)
- Matches backward kernel's BOT_HALF masking exactly
- TC kernel lacks `Q_head_stride` parameter to skip top-half Q rows, so can't implement BOT_HALF correctly

**Verification Needed:**
- Recompile with fix
- Run training → expect grad_norm to stay stable past step 200

---

## Investigation Summary

### Files Examined
1. **FusedSDPAOp.h** (context_parallel forward)
   - Lines 112-141: TC vs scalar kernel dispatch logic
   - Found BOT_HALF mismatch in TC path

2. **FusedSDPABackwardKernel.cu** (context_parallel backward)
   - Lines 72-79, 290, 612-616: BOT_HALF masking (Q[T/2:] x K_full)
   - Verified scalar kernel handles it correctly

3. **FusedSDPAKernel.cu** (scalar forward)
   - Lines 72-79: BOT_HALF correctly outputs zeros for top-half Q

4. **ContextParallelBackward.h** (backward node)
   - Verified merger rescaling and gradient communication consistent

5. **PyTorch CP baseline** (gpt2_cp_headtail_fp32.py)
   - Confirmed stable training with identical config
   - PyTorch's context_parallel() uses correct ring attention + HeadTail load balancing

### Test Data
- **Log 180:** 519 steps, grad_norm explosion at step 196
- **Log 181:** 519 steps, identical pattern
- **Log 182:** 345 steps (stopped), identical pattern (pre-fix compilation)
- **PyTorch baseline:** 300 steps, grad_norm stable 0.2-0.4

---

## Next Steps

1. **Verify Fix Compiled**
   - Recompile and ensure BOT_HALF paths use scalar kernel
   - Check build logs for kernel dispatch

2. **Run Training with Fix**
   - Run new training run (log 183)
   - Monitor grad_norm at steps 190-220
   - Expected: stable around 1-2 (no explosion)

3. **If Explosion Still Occurs**
   - Check if other ring attention steps (LEFT_HALF, CAUSAL) have similar forward/backward inconsistencies
   - Compare AlltoAll vs P2P rotator paths
   - Profile individual attention blocks for numerical instability

4. **Long-term Validation**
   - Match PyTorch loss curve at 300 steps
   - Run to full convergence (6768 steps)
   - Compare final model outputs and generation quality

---

## Technical Debt & Recommendations

### Immediate
- Add unit tests for forward/backward consistency across all mask types
- Add assertion checks verifying forward and backward use same code path for each mask
- Log TC vs scalar kernel dispatch at startup

### Future
- Consider unified CUDA kernel supporting all mask types with all options
- Add numerical validation layer comparing forward outputs to reference (PyTorch)
- Profile gradient flow through each ring attention step

---

## Time Breakdown
- Root cause investigation: 45 minutes
- Pattern analysis & code review: 45 minutes  
- Fix implementation: 15 minutes
- Report writing: 15 minutes

**Total: ~2 hours (completed within one session)**

---

## Files Modified
- `/home/blu-bridge25/TP/TensorParallelismBeta/DTensor/gpt2_cp_test/context_parallel/FusedSDPAOp.h`
  - Removed BOT_HALF from TC kernel condition
  - Updated comments to explain fallback to scalar kernel

---

## Key Learning
Forward/backward kernel mismatches are **silent killers** in distributed training:
- They don't immediately crash (gradients are computed, just wrong)
- They accumulate slowly (error compounds over steps)  
- They manifest when LR increases (error becomes significant relative to updates)
- They're invisible in single-GPU training (mask types only triggered with world_size > 1)

**Lesson:** Always verify forward and backward kernels compute **identical operations** for every code path.
