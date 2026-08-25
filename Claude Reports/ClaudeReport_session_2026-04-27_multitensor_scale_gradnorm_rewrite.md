# Claude Report — 2026-04-27 — MultiTensor scale/grad_norm rewrite (chunked + float4)

**Workspace:** TensorParallelismBeta
**File:** `DTensor/Tensor-Implementations/src/Kernels/cuda/optimizer/MultiTensorKernels.cu`
**Backup:** `MultiTensorKernels.cu.bak_2026-04-27` (untouched copy of pre-change file)
**Target arch:** sm_86 (Ampere)

---

## Context

User asked which is better optimized for sm_86: the active `MultiTensorKernels.cu` or the sibling `MultiTensorKernels.cu_backup` (which contains only the `_sm89_` Ada-dispatch kernels). Note: `sm_86` = Ampere, NOT Ada — Ada is `sm_89`. On user's Ampere card, the Ada dispatch (`get_arch() == ArchFamily::Ada`) is never taken, so the kernels in the active `.cu` are what actually executed.

## Findings (before change)

| Op | Active `.cu` design | `_sm89` (in `_backup`) design |
|---|---|---|
| `multi_tensor_grad_norm` | global-work + per-element binary search over prefix-sum offsets, scalar loads | delegates to active (same kernel) |
| `multi_tensor_scale` | global-work + per-element binary search, **scalar `*= scale`** (no float4) | chunked block→tensor + float4 + `__launch_bounds__(256,2)` |
| `multi_tensor_adam` | chunked + float4 + `__launch_bounds__(256,2)`, **PyTorch math** `1/(sqrt(v/bc2)+eps)` | chunked + float4 + `__launch_bounds__(256,2)`, **APEX math** `rsqrtf(v/bc2 + eps)` |

Key issues with active scale/grad_norm: every thread did a `log2(N)` binary search per element to map global→tensor, and scale had no float4 vectorization at all. Both are major inefficiencies on Ampere.

Adam designs were structurally identical between files; only difference was numeric (eps placement). Active version matches PyTorch `torch.optim.Adam` reference.

## Decision

Apply the chunked + float4 + `__launch_bounds__` design from the `_sm89` kernels to the active `sm_86` path for `scale` and `grad_norm`. Keep `Adam` byte-identical (PyTorch eps-outside-sqrt math preserved for parity with the `comparison_curve.png` baseline).

The optimizations (`float4` vector loads/stores, chunked block→tensor mapping, `__launch_bounds__`) are all available and beneficial on Ampere — they were never Ada-specific despite the `_sm89` naming.

## Changes applied

1. **Backup** of pre-change file: `MultiTensorKernels.cu.bak_2026-04-27`.
2. Added `ScaleLaunchMetadata` and `NormLaunchMetadata` structs (mirrors existing `AdamLaunchMetadata`).
3. Rewrote `multi_tensor_grad_norm_kernel` and `multi_tensor_grad_norm_cuda`:
   - chunked block→tensor mapping (one binary-search-free lookup per block instead of per element)
   - float4 main loop with scalar head/tail
   - per-block shared-memory reduction → atomicAdd to global accumulator
   - `__launch_bounds__(256, 2)`
4. Rewrote `multi_tensor_scale_kernel` and `multi_tensor_scale_cuda` body (Ada dispatch preserved):
   - chunked block→tensor mapping
   - float4 main loop
   - `__launch_bounds__(256, 2)`
5. Removed dead helpers no longer used: `find_tensor_and_local_idx`, `build_offsets`, `compute_grid_size`, `get_num_sms`, `cached_num_sms`, `ensure_metadata_buffers`, persistent buffers `d_metadata_A..D` / `h_metadata_A..D` / `d_offsets` / `h_offsets`.
6. Adam (kernel + host) left unchanged.

## Expected impact on sm_86

- **scale:** ~4× memory throughput from float4 + elimination of per-element binary search. Hot during gradient clipping every step.
- **grad_norm:** float4 loads + no binary search. Hot during gradient clipping every step.
- **adam:** unchanged (was already optimized).

No change to optimizer numerics. No change to public API (`multi_tensor_grad_norm_cuda` / `multi_tensor_scale_cuda` / `multi_tensor_adam_cuda` signatures unchanged).

## Caveats / not yet verified

- Did **not** run a build. The user has not asked for a build invocation; any compile errors from the rewrite need to be caught on their next build. Likely-clean since the patterns are copied from the working `_sm89` versions.
- `_backup` (i.e. `MultiTensorKernels.cu_backup`) and the new active file both define `ScaleLaunchMetadata` and `AdamLaunchMetadata` in the same namespace — identical definitions, ODR-compatible across TUs. Confirmed no kernel/function symbol conflict (the backup's symbols are all `_sm89_` suffixed).
- No throughput measurement run yet — user should re-run training and compare against the `bfha → 64k throughput` baseline.

## Files

- Modified: `DTensor/Tensor-Implementations/src/Kernels/cuda/optimizer/MultiTensorKernels.cu`
- Created: `DTensor/Tensor-Implementations/src/Kernels/cuda/optimizer/MultiTensorKernels.cu.bak_2026-04-27`
- Untouched: `DTensor/Tensor-Implementations/src/Kernels/cuda/optimizer/MultiTensorKernels.cu_backup` (the sm89 dispatch file)
