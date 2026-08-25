# Claude Report - 2026-04-04 - PyTorch SDPA Ring Attn Benchmark Fix

**Workspace:** TensorParallelismBeta
**File:** DTensor/pytorch_attn_bench.py

---

## Problem

PyTorch backward timing regressed in cp_sdpa_compare_test:
- Old buggy: FWD=1.087ms, BWD=7.247ms (single-step, cache artifacts)

Two root causes:

### Bug 1: GPU L2 cache pollution between measurement sections

The forward-only measurement loop (20 iters, torch.no_grad()) evicted backward kernel data from L2 cache. When the fwd+bwd timed section ran next, backward kernels hit cold cache: 7ms instead of ~1.4ms.

Fix: Added a re-warmup pass (NWARM=5 fwd+bwd iters) between the forward-only measurement and the fwd+bwd measurement.

### Bug 2: Single-step benchmark vs 2-step ring

The benchmark was measuring single-GPU local SDPA (1 ring step), while Our C++ and Megatron both measure full 2-step ring attention. This made PyTorch appear faster than Megatron (1.1ms vs 2.9ms), which is wrong for an apples-to-apples comparison.

Fix: Rewrote benchmark to simulate 2 ring steps per iteration:
- Step 0: SDPA(q, k_local, v_local, causal=True)
- Step 1: SDPA(q, k_remote, v_remote, causal=False)
- Output = o0 + o1 (simulates merge, same compute cost)

No MPI communication in the Python benchmark — compute-only ring simulation.

## Final Result

```
                                  Our C++ (TF32)  PyTorch (TF32)  Megatron (BF16, cuDNN FA)
  Forward only                :    3.510 ms        2.038 ms  2.983 ms
  Backward only               :    6.542 ms        2.789 ms  2.116 ms
  Forward + Backward          :   10.051 ms        4.828 ms  5.099 ms
```

Sanity checks:
- PyTorch FWD (2.038ms) < Megatron FWD (2.983ms): correct, PyTorch has no AlltoAll comm overhead
- PyTorch BWD (2.789ms) > Megatron BWD (2.116ms): correct, FP32 math backward > BF16 cuDNN flash backward
- Our C++ > both: FP32 custom flash + AlltoAll comm, expected
