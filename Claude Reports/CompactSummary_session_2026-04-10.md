# Compact Summary: Context Parallel Round-Robin TDD Session
**Date:** 2026-04-10
**Project:** TensorParallelismBeta
**Branch:** _adhi_

---

## Session Overview
TDD-driven implementation of two CP ring attention optimizations: (1) Tiling increase (BLOCK_K 32→64), (2) Round-robin sub-chunking for future-chunk steps. NaN/Inf crash at step 302 was traced to BLOCK_K=64 with BLOCK_Q=32 threads (cooperative tile load bug). After reverting BLOCK_K to 32, crash resolved but throughput shows NO improvement (~55.8k tok/sec vs PyTorch baseline 64k, attn_cp ~224ms vs baseline 221ms).

---

## Key Technical Findings

### BLOCK_K=64 Critical Bug (FIXED)
- **Problem:** Cooperative tile load assigns one K/V row per thread. With 32 threads but BLOCK_K=64, rows 32-63 never initialized → garbage data
- **Symptom:** norm:inf at step 275 → NaN at step 301 → crash at step 302
- **Secondary effect:** smem doubled (32KB vs 16KB) → halved SM occupancy → masked speedup benefits
- **Fix:** Reverted `BLOCK_K = 64` → `BLOCK_K = 32` with guard comment
- **Status:** RESOLVED

### No Throughput Improvement (UNRESOLVED)
- **Observation:** After BLOCK_K fix, attn_cp still ~224ms (no change from baseline 221ms)
- **Expected:** Round-robin sub-chunks use T/2 × T/2 FLOPs vs BOT_HALF's T/2 × T = 2× cheaper
- **Actual:** tok/sec ~55.5-55.8k (vs PyTorch 64k), no timer improvement
- **Verified working:** 
  - `rr_active=true`, `load_balancing=true`
  - Tensor splits correct via `make_shards_inplace_axis(2,2)`
  - LSE shape `[B,H,T_q,1]` compatible with dim-2 split
  - Backward sub-chunk correctly reduces FLOPs by 4×
- **Root cause:** Not yet found — investigated function signatures, tensor shapes, strides
- **Hypotheses remaining:** (1) ring communication latency dominates; (2) merger overhead ≈ FLOP savings at world_size=2; (3) sub-chunk path somehow not executing

---

## Files Modified

| File | Changes |
|------|---------|
| **FusedSDPAKernel.cu** | Reverted `BLOCK_K = 64` → `BLOCK_K = 32` + guard comment |
| **ContextParallel.h** | Pre-split local_q into front/back halves; dual mergers; ring loop mask dispatch (rr_active logic); sub-chunk SDPA path |
| **ContextParallelBackward.h** | Sub-chunk backward with half-size tensor slicing; zero-padding and cat for full dQ/dK/dV |
| **gpt2_cp_test.cpp** | `config.load_balancing = true` at line 542; CPAttention instantiated with load_balancing parameter |
| **test_roundrobin_verify.py** | TDD verification: loss equivalence, throughput improvement ≥10%, timer distribution, no NaN/Inf |

---

## Architecture Summary

### Round-Robin Sub-Chunking Logic
- **When active:** `rr_active = use_roundrobin_ && lb_active`
- **Q split:** `q_front = q[:T/2]`, `q_back = q[T/2:]`
- **Dual mergers:** `merger_front` for q_front tokens, `merger_back` for q_back tokens
- **Ring mask dispatch (lb_active=true):**
  - `source_rank == rank_` → CAUSAL
  - `source_rank < rank_` → LEFT_HALF
  - `source_rank > rank_` → if rr_active: `use_sub_chunk=true, NONE`; else `BOT_HALF`
- **Sub-chunk computation:** `sdpa_fused_forward(q_back, k_front, v_front, NONE, ...)`
- **Final output:** `cat({out_front, out_back}, dim=2)`

### Load-Balanced CP Context
- **Permutation:** HeadTail interleaves tokens across ranks → q/k/v become [B,H,T/n,D] instead of [B,H,T,D]
- **T_local_fwd = T/world_size = 1024/2 = 512 per rank**
- **Round-robin applies only to future-chunk steps** (source_rank > rank_) to reduce FLOPs from T²/2 → T²/4

---

## Test Coverage (TDD Verification)

**File:** `DTensor/test_roundrobin_verify.py`

| Test | Status | Details |
|------|--------|---------|
| TestSchema | Implemented | Baseline CSV exists, columns present, ≥150 rows |
| TestLossEquivalence | Implemented | step0 loss match (atol=0.01), max diff ≤0.1 over 150 steps, no NaN/Inf, monotone decrease |
| TestThroughput | Implemented | attn_cp improved ≥10%, tok_per_sec not regressed >-2%, dt_ms change ≤2% |
| TestTimerDistribution | Implemented | No outlier >3× median, CV < 0.20 |
| New-run CSV | Pending | Skips if `DTensor/CP_Training_logs/CP_Training_log_rr.csv` absent |

---

## Next Steps (Iterations: 2)

1. **Debug Iteration 1:** Find why attn_cp timer shows no improvement despite BLOCK_K fix and rr_active=true
   - Trace execution path: verify `use_sub_chunk=true` is actually called
   - Check tensor shapes in sub-chunk path: confirm q_back is [B,H,T/2,D], k_front is [B,H,T/2,D]
   - Verify `autograd::contiguous` on half-tensor produces correct shape, not full-tensor copy
   - Hypothesis: pre_sharded path incorrectly sets T_local_fwd or sub-chunk path is skipped

2. **Debug Iteration 2:** Measure per-rank computation overhead vs communication latency
   - Profile ring all-reduce cost
   - Measure mergers CPU overhead
   - Compare: T²/4 FLOP savings vs total overhead at world_size=2

3. **After debug:** Run TDD verification once new-run CSV saved as `CP_Training_log_rr.csv`
   ```bash
   pytest DTensor/test_roundrobin_verify.py -v
   ```

4. **Log work** to `/home/blu-bridge25/Documents/obsidian Vaults/Adhitya's Vault/Claude Logs.md`

5. **Write full session report** to `/home/blu-bridge25/Documents/obsidian Vaults/Adhitya's Vault/Claude Reports/`

---

## Stats
- **Total files read:** ~12 (kernel, header, test, CPP files)
- **Total commits:** None (git commands disabled per CLAUDE.md rule 1)
- **Build status:** Compiled successfully after BLOCK_K revert
- **Current throughput:** 55.5-55.8k tok/sec (vs PyTorch baseline 64k)
- **attn_cp timer:** ~224ms (vs baseline 221ms = no improvement)

