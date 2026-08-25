# Session Report: TC Kernel Wiring for CP Forward
**Date:** 2026-04-10
**Project:** TensorParallelismBeta
**Files:** FusedSDPAOp.h

## Change
Wired `mem_efficient_attn_forward_tc` (WMMA tensor core kernel) into `sdpa_fused_forward` for ring steps where T_q == T_k and mask_type is CAUSAL or NONE.

## Dispatch Logic
```
if T_q == T_k AND D <= 128 AND (CAUSAL or NONE):
    -> OwnTensor::cuda::mem_efficient_attn_forward_tc (tensor cores)
else:
    -> launch_flash_attn_fwd_f32 (scalar FP32)
```

## Ring Step Coverage (world_size=2, lb_active=true)
| Step | mask_type | Kernel |
|------|-----------|--------|
| Self (source==rank) | CAUSAL | TC |
| Past (source<rank) | LEFT_HALF | scalar |
| Future (source>rank) | BOT_HALF | scalar |

## Expected Impact
- attn_cp: 221ms -> ~165ms (25% improvement)
- tok/sec: 56k -> ~60k
- Remaining gap to PyTorch 64k: LEFT_HALF/BOT_HALF steps still scalar

## Namespace Fix
Added `OwnTensor::cuda::` qualifier for `mem_efficient_attn_forward_tc` call since the function is declared in `namespace OwnTensor { namespace cuda { ... } }`.

## Next Steps
To close remaining gap: modify TC kernel to accept T_q != T_k, or convert LEFT_HALF to sub-chunk pattern.
