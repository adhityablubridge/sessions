2026-06-18 - 10:55 - Ported the Ada (sm89) PTX-MMA attention forward/backward kernels into CP-aware versions (T_q/T_k + q_offset/k_offset, global-position causal masking) and wired them into the sm89 drop-in FusedSDPAOp - Workspace: OldPush / TensorParallelismBeta / gpt2_cp_test/context_parallel/

## Request
Apply the same TI->CP transformation the user did for the generic kernels
(Tensor-Implementations/.../AttentionForward.cu/AttentionBackward.cu ->
gpt2_cp_test/context_parallel/AttentionForward.cu/AttentionBackward.cu) to the
arch sm89 kernels (arch/AttentionForward_sm89.cu, arch/AttentionBackward_sm89.cu),
so CP ring-attention steps (offset != 0, T_q != T_k) can run on the Ada PTX-MMA
path instead of the generic WMMA fallback.

Decisions (user-confirmed):
- New CP files in namespace OwnTensor::cp with new entry points (no ODR clash).
- FusedSDPAOp_sm89.h calls the new CP-sm89 kernels directly.

## Files created
- gpt2_cp_test/context_parallel/AttentionForward_sm89.cu
  - struct CPFwdParamsSm89 (adds T_q, T_k, q_offset, k_offset).
  - kernel fused_attn_forward_kernel_tc_sm89_cp<HeadDim,BQ,BK,MB>: MMA/ldmatrix/
    scatter-gather core identical to arch kernel; only bounds (T->T_q/T_k) and
    causal masking (global positions q_global=q_offset+qi, k_global=k_offset+kj)
    changed; max_kj computed in global coords; LSE indexed by T_q.
  - cp::cuda::mem_efficient_attn_forward_tc_sm89_strided(...) launcher (same
    signature/dispatch table as the generic CP forward; scale=1/sqrt(hd) internal).
    Dropped the arch kernel's L2-persist window (unsafe under ring rotation).
- gpt2_cp_test/context_parallel/AttentionBackward_sm89.cu
  - struct CPBwdParamsSm89; precompute_D rows bounded by T_q, D flat [BH,T_q].
  - kernel mem_efficient_bwd_unified_kernel_exp12_cp<HeadDim,Causal>: arch exp12
    core (BM=32, BN=16, dK/dV persistent register accumulators direct-assigned,
    dQ atomicAdd) with CP bounds + global causal q_loop_start =
    max(0, k_offset+kv_base-q_offset) and per-element global mask.
  - cp::cuda::mem_efficient_attn_backward_sm89_strided(...) launcher: zeroes dQ
    (atomicAdd target), dK/dV direct-assigned; per-call grads accumulated across
    ring steps by the driver (matches generic CP contract).
- gpt2_cp_test/context_parallel/AttentionForward_sm89.h / AttentionBackward_sm89.h
  - declarations for the two cp::cuda entry points.

## Files modified
- gpt2_cp_test/context_parallel/FusedSDPAOp_sm89.h
  - forward -> mem_efficient_attn_forward_tc_sm89_strided
  - backward -> mem_efficient_attn_backward_sm89_strided
  - Removed the throw-on-offset guards (CP kernels handle offsets now).
  - Kept ATTN_FP32=1 escape routing forward to the CP scalar fp32 path.

## Build wiring (NOT applied here; do on Ada box)
Add to Makefile alongside the existing CP attention objects:
  CP_ATTN_FWD_SM89_OBJ = gpt2_cp_test/context_parallel/AttentionForward_sm89.o
  CP_ATTN_BWD_SM89_OBJ = gpt2_cp_test/context_parallel/AttentionBackward_sm89.o
  $(CP_ATTN_FWD_SM89_OBJ): gpt2_cp_test/context_parallel/AttentionForward_sm89.cu
  	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@
  $(CP_ATTN_BWD_SM89_OBJ): gpt2_cp_test/context_parallel/AttentionBackward_sm89.cu
  	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@
Then append both objects to GPT2_CP_TEST_OBJS, and rename FusedSDPAOp_sm89.h ->
FusedSDPAOp.h on the Ada build. NVCC_FLAGS must target sm_89 (-arch=sm_89 /
-gencode arch=compute_89,code=sm_89).

## Not done / risks
- NOT compiled or run: dev box is RTX 3060 (sm86); ldmatrix/mma.sync.m16n8k8 +
  the sm89 kernels target Ada. Needs build + numerical parity check on sm89.
- The underlying arch sm89 MMA core was itself never validated on hardware
  (per prior logs sm89 cannot run on the 3060); the CP port preserves that core
  verbatim, so forward/backward parity must be validated on Ada before training.
- ContextParallel forward/backward still call the generic CP strided functions;
  only the FusedSDPAOp_sm89.h path uses these new kernels. Original CP files and
  FusedSDPAOp.h left untouched.
