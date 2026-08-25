2026-06-18 - 10:44 - Created an sm89/Ada drop-in copy of FusedSDPAOp.h that calls the Ada PTX-MMA attention kernels directly (forward + backward) instead of the sm86/CP kernels - Workspace: OldPush / TensorParallelismBeta / gpt2_cp_test/context_parallel/FusedSDPAOp_sm89.h

## Request
Make a copy of `FusedSDPAOp.h` dedicated to the sm89 kernels (`AttentionForward_sm89.cu`, `AttentionBackward_sm89.cu`) that can be dropped onto an Ada machine directly, with NO runtime arch gate, so the offset/arch selection logic does not run twice.

## Findings
- The original `FusedSDPAOp.h` calls the CP-namespaced strided kernels (`OwnTensor::cp::cuda::mem_efficient_attn_forward_tc_strided`, `mem_efficient_attn_backward_strided`) — the sm86/CP family.
- The CP forward (`gpt2_cp_test/context_parallel/AttentionForward.cu:990`) ALREADY dispatches to `fused_attn_forward_tc_sm89_cuda` when `T_q==T_k && q_offset==0 && k_offset==0 && get_arch()==Ada`. The CP backward has NO sm89 dispatch.
- The TI-library forward (`Tensor-Implementations/.../AttentionForward.cu:1050`) gates on arch only, because its signature has a single T and no offsets.
- The sm89 arch kernels (`fused_attn_forward_tc_sm89_cuda`, `mem_efficient_attn_backward_sm89_cuda`, both `namespace OwnTensor`) take a single T and have NO q_offset/k_offset params -> only valid for T_q==T_k, offsets==0. No public header declares them; call sites forward-declare them.
- `get_arch()` / `ArchFamily::Ada` live in `ops/helpers/KernelDispatch.h`.

## What was done
- New file `FusedSDPAOp_sm89.h` (rename to `FusedSDPAOp.h` on the Ada build).
- Same public interface (`sdpa_fused_forward`, `sdpa_fused_backward`) as the original.
- Forward path calls `OwnTensor::fused_attn_forward_tc_sm89_cuda` directly; backward calls `OwnTensor::mem_efficient_attn_backward_sm89_cuda` directly.
- No arch gate (file is Ada-dedicated). Forward-declares both sm89 kernels in `namespace OwnTensor`.
- Correctness guard (NOT an arch gate): throws if `T_q != T_k || q_offset != 0 || k_offset != 0`, since the sm89 kernels cannot express CP ring-attention sub-chunking; this fails loudly instead of producing wrong gradients.
- Forward scale hard-derived as `1/sqrt(D)` to match CP/sm86 numerics (CP path hardcodes the same).
- Backward: `D_buf` strides set as contiguous [B*H, T_q] (`d_sB=H*T_q`, `d_sH=T_q`); `skip_grad_zero=false` so the kernel zeroes dQ (atomicAdd accumulation) while dK/dV are written by direct assignment.

## Not done / notes
- Not compiled here (dev box is RTX 3060 / sm86; sm89 kernels cannot run). Needs a build + numerical parity check on Ada hardware.
- Original `FusedSDPAOp.h` left untouched.
