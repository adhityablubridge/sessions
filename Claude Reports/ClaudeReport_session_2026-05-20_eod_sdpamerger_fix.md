2026-05-20 - EOD writeup on the loss deviation / norm explosion at step 33 debug - TensorParallelismBeta / SDPAMerger.h, ContextParallelBackward.h, cp_bwd_isolated_test

# EOD Report - 2026-05-20

Spent the day chasing down the loss graph deviation and the norm explosion that kept showing up around step 33. Ended up actually finding it, which was nice.

Started by writing a test file that mirrors the live scripts on both sides (ours and PyTorch) but isolates just the SDPA backward intermediates and outputs, comparing with cosine similarity. First useful signal came right away - the ranks weren't symmetric, which immediately pointed at the LB-specific accumulation paths.

The numbers looked like this:

- rank 0 dQ: max=4.28e-03, cos=0.9920
- rank 0 dK: max=2.95e-03, cos=0.9965
- rank 0 dV: max=1.63e-03, cos=0.9981
- rank 1 dQ: max=1.98e-05, cos=0.99999971
- rank 1 dK: max=1.71e-03, cos=0.9895
- rank 1 dV: max=1.76e-03, cos=0.9924

Mapping that back to the code, rank 0 dQ is the only one using the i > rank partial-Q accumulation path (cat-clone-add into the tail half), and that was the worst at 0.992. rank 1 dQ uses only full Q add and was basically perfect. rank 1 dK/dV are the ones doing the i <= rank head-half pad-then-add - also bad. So the pattern was super clear: anything touching the LB-specific accumulation path was off by ~1-10%, anything on the plain full-add path was within 0.1-0.3%.

First instinct was that the bug was in cat/narrow_view/clone themselves under some weird stride condition. Wrote a tiny unit test - took a contiguous [1,2,64,64] tensor, ran the exact narrow_view + clone + cat pattern, checked it against torch.cat(t.chunk(2), dim=2). Bit-exact, max_abs_diff = 0. So the ops in isolation are fine.

That was confusing because the live pipeline still diverges. Two things stood out - rank 1 dQ being bit-exact to ~7 sigfigs whenever no LB-specific accumulation is invoked, and rank 0 dQ being worst exactly when the partial-dQ path kicks in. Started suspecting the live grad_q tensor had different strides or storage than my synthetic arange tensor, or maybe a grad_fn attached doing something funny.

Then I ran an LB-OFF vs LB-ON comparison and that confirmed it 100% - with LB off, every tensor matches PT to ~1e-5 (basically TF32 WMMA noise floor). With LB on, errors blow up by 100x on exactly the LB-specific paths. The per-T profile on rank 0 dQ was the giveaway: head=4.04e-6, tail=2.5e-4 with LB on, ratio 62.77x. With LB off the same head/tail were both noise.

For a while I thought the SDPA backward kernel itself was over-amplifying dQ at heads other than head 0 in the Q tail region - argmax was sitting at H=1, T=44, D=51. But that turned out to be wrong. The same kernel with the same T_q=T_k=64 step-0 shape produces clean output when called from the non-LB path. So the kernel was getting blamed for someone else's mess.

To actually pin it down I added a DUMP_CP_DEEP=1 path in ContextParallelBackward.h that dumps full-tensor .bin snapshots at every meaningful point in the ring loop - grad_q before the step, the grad_q_step output, grad_q after accumulation, same triple for grad_k/grad_v. Mirrored the same dumps in cp_bwd_isolated_test.py at the matching points in the copied _templated_ring_attention_backward. Then a python diff script to walk through pairs and report max_abs_diff.

The first divergence showed up at step 0 grad_q_step on rank 0 - 16.5% max relative divergence. That was already weird because step 0 is the first SDPA backward call, before any partial accumulation has even run. Inputs to that kernel: q_bwd, k_bwd, v_bwd, grad_out_bwd were all bit-exact between PT and C++. The only inputs that could possibly differ were out_bwd (merged_out_) and lse_bwd (merged_lse_), since those come from the forward pass.

That shifted the search to forward. Rank 0 forward does step 0 SDPA full, then step 1 partial merge into the tail half of the accumulator. Rank 1 doesn't hit the partial-merge path at all. So if the partial merger drifts in C++, rank 0's merged_out tail half is wrong, which then feeds the backward kernel and gets amplified into the chunk_3 tail dQ blow-up. Consistent with everything we'd seen.

Dumped merged_out/merged_lse after each merger.step(...) call on both sides:

- step 0 rank 0: noise (full merge init)
- step 0 rank 1: noise (full merge init)
- step 1 rank 0: 14.3% relative drift (partial merge tail-half)
- step 1 rank 1: noise (full merge)

Bug located. It's in SDPAMerger::step in the partial=true branch, specifically the narrow_view(seq_dim, half_T, half_T) + correction update arithmetic.

Then the actual root cause - the element-wise binary op in our tensor lib was flat-fast-pathing a narrow_view on a non-leading axis. For lse_.narrow_view(2, half_T, half_T) on a [1,2,64,1] tensor, the correct stride-aware read at position (h=1, t=0) should hit physical offset 96. The op was treating the view as flat[32:32+64], so the same position read physical 64 - which is head 1's FIRST half. Head 0 by sheer coincidence reads correctly because head 0's second half lives at flat 32..63. Head 1 reads head 1's first half. That's exactly the 14.3% drift, and explains why it only shows up at H>=1.

Fix was three lines - clone accum_out and accum_lse in SDPAMerger.h so the arithmetic sees contiguous tensors instead of a strided narrow_view.

After the fix:

- Rank 0 merged_out rel_max: 14.3% -> 0.063%
- Rank 0 merged_lse rel_max: 8.9% -> 0.029%
- Backward dQ/dK/dV cos_sim ~ 0.9999997 on both ranks, rel_max <= 3e-3

Norm doesn't explode at step 33 anymore. Forward fix propagates cleanly through backward.

Bigger picture takeaway - the tensor library's binary op flat-fast-paths whenever it can, and a narrow_view on a non-leading axis silently breaks that assumption. Worth auditing other places that do narrow_view on dim>=2 and feed it into elementwise ops. Sanity test passing in isolation didn't catch this because the synthetic test happened to land in a layout where the fast path was still correct - the real merger state had a different shape that exposed the bug.
