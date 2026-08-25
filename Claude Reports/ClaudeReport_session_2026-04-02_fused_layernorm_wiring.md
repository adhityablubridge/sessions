# ClaudeReport_session_2026-04-02_fused_layernorm_wiring

**Date:** 2026-04-02
**Workspace:** TensorParallelismBeta
**Files:** DTensor/gpt2_cp_test/gpt2_cp_test.cpp, dnn/FusedLayerNormOp.h, dnn/FusedLayerNormOp.cpp, Tensor-Implementations/include/nn/optimizer/Optim.h

---

## Work Done

### Goal
Wire component optimizations (FusedAdamW, FusedLayerNorm) from previous session into the actual gpt2_cp_test training loop.

### FusedLayerNorm Wiring
- Added `dnn/FusedLayerNormOp.h` include and `using namespace OwnTensor::dnn`
- Replaced `ln.forward(x)` with `dnn::fused_layer_norm(...)` at 3 sites:
  - CPAttention::forward() — `dims[2]` for cols
  - MLP::forward() — `dims.back()` for cols
  - GPT::forward() final ln_f — `config.n_embd` for cols

### FusedAdamW Wiring (attempted, reverted)
- Added inline accessors to AdamW in Optim.h: `get_m()`, `get_v()`, `get_beta1/2/eps/weight_decay()`, `ensure_initialized()`, `increment_step_count()`
- Replaced `optimizer.step()` with fused AdamW loop using `cuda::fused_adamw_with_unscale_cuda(...)`
- **Reverted**: caused 543MB memory regression (log118=4692MB → log120=5236MB)

### Memory Regression Investigation
- Added `cudaMemGetInfo` checkpoints at 4 stages (cudaSetDevice, model init, optimizer init, fwd+bwd)
- Confirmed regression was entirely in the optimizer step, not in LN or fwd+bwd
- Root cause: `optimizer.step()` in libtensor.a detaches parameters from autograd graph after update, freeing backward graph nodes. FusedAdamW writes directly to data pointers, leaving graph intact
- Fix: revert to `optimizer.step()`; FusedAdamW available via standalone test_fused_adamw

### Final State
- FusedLayerNorm: wired in (1.89x fwd, 1.75x bwd speedup, no memory regression)
- FusedAdamW: standalone test only (test_fused_adamw)
- optimizer.step() restored in training loop

### Comparison (log118 baseline vs expected after rebuild)
| | log118 | log120 (with FusedAdamW) |
|---|---|---|
| mem_gpu_mb | 4692 MB | 5236 MB |
| timer_optim | 37.9 ms | 3.9 ms |

After rebuild with FusedLayerNorm only: expected ~4692MB, faster LN timing.
