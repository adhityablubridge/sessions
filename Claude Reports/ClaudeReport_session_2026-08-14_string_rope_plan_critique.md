# ClaudeReport_session_2026-08-14_string_rope_plan_critique

2026-08-14 - Critique of the STRING (Shifted RoPE) plan: block-quantised two-pass design, verified against the fused-RoPE SDPA signature - CP / .mdfiles/polished-cuddling-graham.md (plan 1 of 5 in the file)

## Verified sound
- The paper's Eq. 3 correction is right: their worked example (L=9, S=3, last row [5,4,3,2,1,0,2,1,0])
  implies `m - n >= S`, with W=0 in that example.
- Two-pass derivation: (m - S + W) - n = d - S + W. Correct.
- `sdpa_fused_forward_rope` (context_parallel/FusedRoPESDPA.h:87) takes independent T_q/T_k and returns
  lse [B,Nq,T_q,1] keepdim, so the K-slicing and SDPAMerger reuse work as written.
- Block decomposition: far pairs have d > (k-1)B, near pairs d < kB, so the threshold is fuzzy over
  exactly one block width - as the plan states. FLOPs claim holds (the slice union is the causal
  triangle). lambda-linearity composition and the n_hat==0 gate are correct.
- G=1 gives exactly one causal call, so "bit-identical" is achievable there (out is bf16 internally,
  but with one call there is no merge to perturb it).

## Findings
1. BLOCKING - both identity tests are excluded by the plan's own asserts (`0 < W < S`, `k <= G`,
   S=(k-1)B). Test 2 (G=1) forces k=1 -> S=0 -> assert fails. Test 3 (W=S), called "the key test",
   is excluded by the strict `W < S`. Both cases are safe (shift=0 keeps rows in [0,T)). Relax to
   `0 < W <= S` and skip the W<S check when the far region is empty (k >= G).
2. The pre-registered criterion can produce a false negative. With G=32,k=11: at T=16k the far region
   starts at d>5120 (below the 8,174 reach, so STRING bites), but at T=32k it starts at d>10240,
   ABOVE the 8,183 reach - the whole retrieval-relevant range is in the untouched near band. STRING
   could then create retrieval only beyond a still-broken 8.2k-10.2k gap, which scores as "reach
   unchanged" under a first-failure-point metric. Fix: tie k to the measured reach per T (T=32k wants
   k~9), and judge on the full NLL-vs-distance curve ("retrieval present beyond the baseline reach").
3. Criterion is written against run100's numbers (8,174/8,183) while the table names run1 as the
   primary target and run1's baseline reach does not exist yet. Restate per-model; make the run1
   baseline a prerequisite.
4. `--blocks 3` contradicts the sibling needle plan's pre-registered B=5 ("B=3 would hang a
   significance call on 2 dof") while STRING's criterion is itself a 2 s.e. test.
5. Silent no-op under CP>1: dispatch is `string_on && world_size == 1`, so CP_STRING=1 on a CP=2 launch
   yields baseline numbers labelled STRING. Should die - same class as the theta and n_hat gates.
6. "One STRING row = one baseline row" rests on FLOPs parity, but the plan's own stated overhead is
   ~96 kernel launches/layer vs 1, and at T=4k/G=32 the tiles are 128 wide (launch-bound). Measure one
   cell before propagating cost parity.
7. Minor: `S` has two meanings - code S=(k-1)B (T/3.2 at k=11) vs the documented midpoint (k-0.5)B
   (T/3.05). The asserts and Test 3 reference the ambiguous symbol. Name them S_shift / S_nominal, the
   same argument the plan already made for renaming blocks -> GRID.

## Files touched
None - review only.

## Round 2 findings
All six round-1 items closed. Notable self-addition: the "What STRING can and cannot do" section
deriving reach_STRING = R + S - W, the 2R - W ceiling, and validating it against the paper's own 1.56x
numbers (Llama3.1-70B / Qwen2-72B 64K->100K) - that closed form is what makes a null interpretable.

Remaining:
1. The OLD success criterion is still in the file (lines ~328-330), stating the criterion against
   run100's 8,174/8,183 - verbatim the text the new per-model criterion section says it replaces. Two
   mutually exclusive "pre-registered" criteria in one plan. Delete the stale block.
2. All four worked S values violate the stated S-selection rule (largest multiple of B <= R). With
   B=1024: 4k R=3045 -> 2048 not 3072; 8k R=4078 -> 3072 not 4096; 16k R=8174 -> 7168 not 8192;
   32k R=8183 -> 7168 not 8192. Every one is rounded UP. The direction is load-bearing: the merge
   argument [0,R) U [S, R+S-W] -> one interval needs S <= R, and rounding up reopens the gap [R,S) that
   the criterion section identifies as the false-negative mechanism.
3. The T=4k cell silently degrades to baseline: with the plan's own S=3072, B=1024, T=4096 -> G=4,
   k = S/B+1 = 4, so far_exists = (k<G) = false, no far pass, output identical to baseline, no warning.
   Use the rule-correct S=2048, and warn/die when far_exists is false while CP_STRING=1 (same
   silent-wrong-answer class as the world_size>1 die).
4. Stale justification text: delta-arithmetic paragraph still cites `W < S` / "that is what 0 < W < S
   guarantees" after the assert became 0 < W <= S (the bound still holds at W=S, only the citation is
   stale); and the caveat's "fuzz is +/-5% of S" is from the retired G=32,k=11 parameterisation - at
   B=1024 it is +/-512, i.e. +/-6% at S~8k but +/-17% at S=3072.
5. Minor: the S ablation {0.5R, R, 1.5R} must be snapped to multiples of B or `assert S % B == 0`
   kills the run (0.5*8174 = 4087 is not).

## Round 3 findings (final)
All round-2 items closed. Verified the new per-cell (B, S) table arithmetic end to end - every value is
correct: S = largest multiple of B <= R (2560/3584/7168/7168), G, k, k<G, fuzz = B/2 over S
(10.0%/7.1%/7.1%/7.1%), and predicted R+S-W (5477/7534/15214/15223). Ablation snapping is right too
(3072/7168/11264 at B=1024). CP_STRING_ALLOW_NOOP keeps the identity tests runnable while blocking a
silent baseline-labelled-STRING cell.

Remaining, both minor:
1. The Cost paragraph did not follow the per-cell B change - still says "B=1024 ... G=4 at T=4k,
   G=32 at T=32k", but the grid uses B=512 at 4k/8k, giving G=8 and G=16 (24 and 48 calls/layer).
   Conclusion unaffected, numbers stale.
2. The S = 1.5R ablation is described as "should degrade reach", but the plan's own model predicts no
   degradation: the near band still covers [0,R) untouched so first-failure stays at R; what appears is
   a DISCONNECTED retrievable island at [S, S+R-W) with a dead gap [R,S). That is the sharper,
   falsifiable prediction and the direct test of the merge argument that justifies rounding S down.

Verdict: design, asserts, identity tests, S-selection rule, closed-form criterion and caveats all hang
together. Start with the run1 baseline reach curve - every S in the grid depends on it.
