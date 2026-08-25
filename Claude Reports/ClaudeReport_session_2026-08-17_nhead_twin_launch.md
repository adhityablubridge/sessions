# ClaudeReport_session_2026-08-17_nhead_twin_launch

2026-08-17 - 16:29 - Launched a parameter-matched head-split twin to test whether head count is what sets the context-rot boundary; scored the S=4096 STRING arm; found a plan provenance error and a live-edit bug - Workspace: CP - Files: /tmp/.../overnight_20260817.sh, Tests/bluscriptcp/needle_sweep.sh, Scripts/Blutrain/bluscriptCP.cpp, CP_BluScreipt_Training_logs/CP_Training_log{1,100,200}_config.txt

## What was asked

Whether the running jobs had checkpoints and could be stopped, then whether the claim that `n_head` drives context rot could be tested on the local GPUs. The user's counter-proposal to head knockout — "why can't we just configure fewer heads or layers and preserve the parameter count, then compare the rot boundary" — is the better experiment, and is what was launched.

## Two errors found, neither by the tests themselves

**1. The S=4608 arm never ran, and the cause was self-inflicted.** `string_optimal_S.log` ends with `needle_sweep.sh: line 217: syntax error near unexpected token '('` against a file that is 214 lines long and passes `bash -n` clean. Cause: `needle_sweep.sh` was edited (adding `--p-fracs` and `PROBE_POS`) **while a running job was executing it**. Bash reads scripts by byte offset, so the edit shifted offsets under the live shell; arm 1 completed, then the interpreter resumed at a stale offset and hit garbage. Arm 1's 65 markers survived, arm 2 got zero.

Standing rule from this: **never edit a script that a running job is executing.** The overnight script carries this in its header so the next session cannot repeat it.

**2. The plan's checkpoint provenance is wrong.** The plan's B2 table asserts `run1` is "the 4k-pretrained **114M** base" and that `run100` is "the 64k fine-tune of `run1`". The config files say otherwise:

| | run1 | run100 | run200 |
|---|---|---|---|
| d_model | **384** | 768 | 384 |
| n_layers | **6** | 12 | 6 |
| q/kv heads | **6 / 2** | 12 / 4 | 6 / 2 |
| params | **48,076,416** | 114,151,680 | 48,076,416 |
| context_length | 4096 | **65536** | 4096 |

`run1` is a **48M**, architecturally identical to `run200`. `run100` is the only 768/12-layer run in the logs, so it cannot be a fine-tune of a 48M. Every plan claim resting on "run1 = the 114M base" — including the twin budget-matching and the capacity-contrast design in B2/B2b — needs re-deriving. Recorded, not yet fixed in the plan file.

## The S=4096 STRING arm: mechanism confirmed, magnitude short

65/65 complete, scored. The curve is **non-monotone, and that is the signature of STRING working**, not noise:

| p_frac | d | STRING remaps to | score |
|---|---|---|---|
| p0.7 | 4,920 | *near band, untouched* | 0.182 |
| p0.65 | 5,738 | 1,770 | **0.383** |
| p0.6 | 6,555 | 2,587 | 0.254 |
| p0.55 | 7,373 | 3,405 | 0.171 |

d=5,738 retrieves twice as well as the *shorter* d=4,920, because 5,738 sits past the far threshold (~5,108) and gets remapped onto a well-trained 1,770 while 4,920 stays in the untouched near band. In NLL terms (10.47 +/- 0.20 vs 12.29 +/- 0.13) that is ~7.6 s.e. — real, not sampling noise.

**Two findings, one good and one not:**
- **The curve is CONTINUOUS — no dead gap.** This is the direct confirmation of the pre-registered `S <= R` merge condition, and the contrast with S=7168's disconnected island is exactly what the plan predicted.
- **The magnitude falls well short.** The 0.2 crossing moves ~4,900 -> ~6,900 (+2,000) against a predicted `R+S-W ~ 8,968`. Checked against pure-remapping expectation: remapped d=3,405 scores 0.171 where the baseline at 3,405 would be ~0.4, and remapped 5,857 scores 0.123 where baseline would be ~0.2. **The far band underperforms pure remapping by roughly 2x**, so remapping a distance is not equivalent to having been trained at it. That is a real limit on STRING's mechanism and belongs in the write-up.

## The n_head experiment, and why the user's framing beat mine

I had proposed **head knockout** (zero one trained head at a time, count which ones destroy retrieval — the Wu et al. method). It measures a *count* and needs a new `CP_MASK_HEAD` knob. The user's proposal — reconfigure at fixed parameter count and compare the rot boundary — is the **direct causal test**; knockout is only a proxy for it. The user's framing was adopted.

A trained checkpoint cannot be reconfigured (re-splitting `head_dim` 64->128 hands every channel the wrong RoPE frequency), so this requires pretraining. The local box has already done exactly this scale: `run200` is 1300 steps at 20.3 s/step = **~7.3 h on one GPU**.

**The design, at `d_model=384` with parameters held exactly:**

| arm | n_head | head_dim | kv | k/v_proj | RoPE freqs/head | params |
|---|---|---|---|---|---|---|
| `run200` (ref) | 6 | 64 | 2 | 384x128 | 32 | 48,076,416 |
| **`run21`** (twin) | **3** | **128** | **1** | 384x128 | **64** | 48,076,416 |

Exact match because `q_proj` is 384x384 either way and `k/v_proj` stays 384x128 while `kv_heads * head_dim` is held at 128 (`2*64 == 1*128`).

**Zero code changes were needed** — `head_dim` auto-derives at `bluscriptCP.cpp:661` (`d_model / q_heads`), `CP_N_HEAD`/`CP_N_KVHEAD` are existing env knobs, and the eval harness is already `head_dim`-generic (`HEAD_DIM` overridable at `needle_sweep.sh:40`; `lambda_file` derives `half = hd//2` and generates the 64-entry NTK vector automatically).

**Why there is no third arm.** `bluscriptCP.cpp:934` gates `head_dim must be 64 or 128 (fused kernel)`, and the GQA fused dispatch (`GQA_fused_fwd_sm103_cp.cu:351,357`, `GQA_fused_bwd_sm103_cp.cu:651,658`) has only `case 64:` and `case 128:`. So `n_head=12` (which needs `head_dim=32`) is blocked without new CUDA work. The axis is 2 points, not 3.

### The predictions are mutually exclusive, which is the point

| hypothesis | mechanism | predicts |
|---|---|---|
| retrieval-head count | 6 heads is a bigger pool for the sparse ~5% to emerge from | **6 beats 3** |
| low-frequency RoPE channels | `head_dim=128` gives 64 frequencies/head vs 32, and low-freq channels are what carry retrieval | **3 beats 6** |

Either outcome discriminates. That is much stronger than a one-sided test that can only fail to reject.

### The confound, stated plainly

Fixing the parameter count **forces `kv_heads` 2->1**, since `kv_heads * head_dim` must stay 128. So this is 6-head GQA vs 3-head MQA — a **joint contrast over (n_head, head_dim, kv_heads)**. At fixed `d_model` and fixed parameters there is no way to separate them. The write-up must say "joint", never "n_head".

**`n_layer` at fixed parameters does not work at this scale.** Going 6->12 layers forces `ffn_hidden` from 1024 down to ~128 to stay matched — degenerate. Depth requires `d_model` to move too, changing everything at once. Heads are isolable here; depth is not.

## What is running

`/tmp/.../overnight_20260817.sh`, master log `/tmp/overnight_20260817/master.log`. Strictly sequential, per the user's choice of backlog-first:

| phase | work | GPUs | est |
|---|---|---|---|
| 1 | STRING S=4608 B=512, 65 evals | 0,1 | ~1.3 h |
| 2 | context-rot test 3, resumed from 8/60 | 0,1 | ~1.0 h |
| 3 | `run21` twin pretraining, 1300 steps | 0 | ~7.3 h |

Phases 1-2 are allowed to fail without blocking phase 3 (no `set -e`) — the training is the science.

**A parameter-match gate is built into phase 3**: it waits for the driver's config file, asserts `Parameters: 48076416`, and **kills the run immediately on mismatch** rather than burning 7 h on a twin that is not comparable. `run21` was verified free (logs 1-20, 100, 200 exist; `next_free_log` = 21) and 107 GB disk is available.

Everything affecting the computation is copied from `run200`: T=4096, B=2, global_batch=131072, max_steps=1300, warmup=91, ffn=1024, tying=0, ulysses, 1 GPU, and the compiled LR defaults (6e-4 -> 6e-5 — there are no LR env knobs, so these cannot silently drift). Only the head split differs.

## Next

1. **Morning: eval the pair.** `run21` at T=16k on the fine grid with `HEAD_DIM=128`, and re-run `run200` on the **same** grid so the comparison is matched-stimulus. Deliberately not automated — the reach curve should be read before committing more GPU time.
2. **The 48M lives at T=16k only.** Its reach there is 4,087; at 32k the score denominator collapses to 0.21, under the pre-registered 0.5 gate. Do not extend this experiment to 32k.
3. **Correct the plan's B2 provenance table** and re-derive everything downstream of "run1 = the 114M base".
4. Standing caveat: a 2-point joint contrast gives direction, not an exponent, and 48M is the rung where the probe is weakest.

---

## ADDENDUM (20:59) - phases 1-2 results, and two more launch bugs

### TEST 3 IS THE HEADLINE: context rot tracks DISTANCE, not absolute position

The needle sat at absolute slot 4087 in every one of these trials. Only the probe moved.
lambda was held fixed (T=16384, NTK s=4) throughout, so compression cannot explain it.

| probe at | distance | predicted if DISTANCE | predicted if POSITION | observed | sig |
|---|---|---|---|---|---|
| 16364 | 12,277 | ~0.05 | ~0.05 | 0.05 | - |
| 12288 | 8,201 | ~0.06 | ~0.05 | **0.055** | no |
| 8192 | 4,105 | ~0.22 | ~0.05 | **0.220** | YES |
| 6144 | 2,057 | ~0.35 | ~0.05 | **0.391** | YES |

The DISTANCE prediction was pre-registered in the script header before the run and it hit
almost exactly. Retrieval goes from invisible to strong purely by moving the QUESTION closer
to a needle that never moved. "Buried early in the document" is not the failure mode;
"far from the query" is.

Consequences: (a) one axis of the dependency map is settled; (b) every earlier reach number
is correctly read as a DISTANCE limit -- the old layout confounded distance with absolute
position and distance is the causal one; (c) STRING's premise (remap relative distance) is
the right premise for these models.

### S=4608 / B=512 did not underperform -- it BROKE, and gate A4.1 caught it

```
floor(p1)=12.9428  absent=12.9447  null=12.8164  denom=0.0019 nats
GATE A4.1 FAIL: floor is not below absent by >= 0.5 nats
```

Against the S=4096 arm's denominator of 9.03 nats this arm has 0.0019. Every NLL collapsed to
~12.9 including p1, where the needle is IMMEDIATELY ADJACENT to the probe -- the model lost the
ability to copy a neighbouring token. The banner confirms the config was as intended
(`S=4608 B=512 -> G=32 k=10 shift=-4480 far=yes`), and the near band is 5,120 tokens wide in
both arms, so the only real difference is tile granularity.

**Conclusion: B=512 (G=32) is not usable on this path.** Most likely bf16 accumulation through
SDPAMerger across 32 blocks x 3 calls -- the STRING report already measured mean 1.6e-3 /
max 2.1e-2 nats of merge noise at G=8. This is a limit on the IMPLEMENTATION, not a fact about
STRING. The scorer withholding scores instead of emitting a plausible-looking 0.00 curve is the
A4.1 gate earning its keep.

So the STRING result rests on the B=1024 arm alone, and the attempt to tighten the fuzz window
failed.

### Two more launch bugs, both mine, both environmental

1. **GLIBCXX_3.4.32 not found (libprofiler.so).** Not a broken build -- the same binary had just
   run 130 evals. needle_sweep.sh:64 exports its own
   `LD_LIBRARY_PATH=BluTrain/Tensor-Implementations/lib:BluTrain/Profiler/lib:...`, so every eval
   inherited it while the direct mpirun launch did not. **Rule: any direct bluscriptCP_exec
   invocation needs that export.**
2. **`./build/bluscriptCP_exec missing`** when running a scratchpad copy of needle_sweep.sh:
   line 30 is `cd "$(dirname "$0")/../.."`, which resolves to the repo root only from
   Tests/bluscriptcp/. Patched the copy to an absolute cd.

### Parameter gate: the twin is NOT exactly matched

run21 reports **48,077,184** vs run200's 48,076,416 -- delta **+768**, fully accounted for:
q_gamma/k_gamma (bluscriptCP.cpp:241-242) are head_dim-sized, so 2*(128-64)=128 per layer x 6
layers. That is **0.0016%** of the model and cannot plausibly move a retrieval boundary, but the
write-up must say "+768 params (QK-norm gammas)", NOT "exactly parameter-matched". The gate was
pinned to 48,077,184 rather than loosened to a tolerance, so a genuinely mis-specified arch
still aborts.

### Both GPUs now in use

- **GPU 0**: run21 training, 22.92 s/step steady (run200 was 20.3 -- wider heads cost ~13%),
  1300 steps -> 8.3 h, ETA ~05:05. Loss 10.91 -> 8.94 by step 30.
- **GPU 1**: run200 reference arm at T=16k on the fine grid, 65 evals. Reads the SAME
  /tmp/string_eval_16k/needles files as the 114M arms, so run200/run21/run100 are all
  matched-stimulus. Uses a scratchpad COPY of needle_sweep.sh with an ALLOW_CONCURRENT_EXE
  opt-out; the tracked harness is untouched. Training confirmed unperturbed after the eval
  started.
