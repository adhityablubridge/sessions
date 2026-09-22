# 2026-09-21 - 64k -> 128k re-run on the rebuilt mix: the contiguity fix, with tables and plots

Workspace: CP (`/home/blu-bridge25/CP`), training on 8x H200 (torn down), artifacts on
`hf.co/unparallelled/BluScriptCP`.
Deliverables: `results_2026-09-21/{data.py, mk_tables.py, mk_t128b.py, TABLES.md,
t128b_results.png, loss_run1_oldmix.csv, loss_run2_newmix.csv}`.

---

## 1. What was run

`t128b` = a second attempt at the 64k -> 128k rung, started from the `t64` checkpoint
(step 1200) and stopped at step 500, before annealing, so more data can be added later
and the anneal deferred. Config: `T=131072`, `NP=8`, `CP_SIZE=8`, `DP=1`,
`YARN_SCALE=32`, `YARN_ORIG_MAXPOS=4096` (invariant `32 x 4096 == 131072` asserted in the
script), `rope_theta=500000`, `grad_accum=4`, global batch 524,288.

Exactly one variable was changed against the first attempt: the bucket weights.

| mix | b1 4-8k | b3 16-32k | b4 32-64k | b5 64-131k | b6 >=131k | contiguity |
|---|---:|---:|---:|---:|---:|---:|
| mix64 (t64, window 65,536) | 10.0% | 15.0% | 25.0% | 50.0% | 0% | **50.0%** |
| mix128 (t128 first attempt) | 10.5% | 15.5% | 25.5% | 39.5% | 8.9% | **8.9%** |
| mix128_b6_35 (t128b) | 10.0% | 15.0% | 15.0% | 25.0% | 35.0% | **35.0%** |

Contiguity = share of tokens from documents at least as long as the training window, i.e.
the share of windows one document can fill on its own. The first 128k attempt collapsed
to 8.9% for a boring reason: the window doubled to 131,072 and the bucket weights did not
move, so most windows became several unrelated documents concatenated.

## 2. The result

`niah_single`, T=131,072, YaRN x32, ORIG=4,096, N=100, identical haystack
(199,754,694 B) as every prior measurement in this project.

| prompt length | virtual pos | t128b s300 | t128b s500 | t64 (parent) | t128 first attempt |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 128 | 100 | 95 | 100 | 98 |
| 16,360 | 511 | 99 | 97 | 99 | 98 |
| 32,744 | 1,023 | **89** | 77 | 93 | **69** |
| 65,512 | 2,047 | - | - | 63 | 31 |

The hypothesis held. At 32,744 the first attempt had lost 24 points against its own
parent (69 vs 93); the rebuilt mix scores 89 at step 300, recovering 20 of them.

## 3. The loss inverts the usual reading

50-step trailing mean:

| step | first attempt (b6 8.9%) | t128b (b6 35%) |
|---:|---:|---:|
| 300 | 2.55 | 2.52 |
| 400 | 2.08 | 2.45 |
| 500 | **1.52** | **2.33** |
| 874 | 1.02 | - |

The run with the *lower* loss is the broken one. Predicting the next token inside one
short document, inside a 131,072-token window, is much easier than predicting across the
window. The loss drop beginning around step 350 in the first attempt is the model finding
that shortcut, and it maps onto the retrieval collapse. This is worth keeping in mind for
future rungs: at long context, loss falling faster than expected is a reason to check the
data, not to celebrate.

## 4. The learning rate was 6x the planned value

`t128b_train.sh:63` sets `CP_REWARMUP_PEAK=0.3`. `max_lr` is hardcoded to `6e-4` at
`bluscriptCP.cpp:151` and `CP_MAX_LR` is a no-op (confirmed again: zero occurrences in
the source). The run therefore trained at a flat **1.8e-4** throughout - verified in the
log, peak LR reached is `1.8000e-04`.

The ladder note written the same day specifies **3e-5** for the 128k rung, i.e.
`CP_REWARMUP_PEAK=0.05`. The run is 6x hot. This is my own inconsistency: I flagged the
setting in the morning and then did not apply it in the launch script.

Consequences:

- It is the leading explanation for section 5's step-300 -> step-500 decline.
- It does **not** undermine the contiguity result. Both 128k runs used the identical
  1.8e-4 schedule, so 69 -> 89 at 32,744 remains a clean one-variable test of the mix.
- It is cheap to test: resume from the step-500 resumable set at 3e-5 and see whether
  32,744 recovers toward 89.

## 5. Retrieval fell between step 300 and step 500

| length | s300 | s500 | delta |
|---:|---:|---:|---:|
| 4,096 | 100 | 95 | -5 |
| 16,360 | 99 | 97 | -2 |
| 32,744 | 89 | 77 | -12 |

Only the -12 clears the +/-9-13pp Wilson interval at N=100. But the sign is the same at
all three lengths, which is what makes it worth watching rather than dismissing. Given
section 4, the most likely cause is the hot LR. Practical consequence: evaluate every 100
steps on the next rung, not every 500 - the checkpoints are already being kept at that
cadence, the evals were not.

## 6. RULER coverage (t64, measured 2026-09-17)

13 tasks, scale chosen per length so the needle lands near virtual position 1,023
(4,096 T=8192 x4 | 8,192 T=16384 x8 | 16,360 T=16384 x16 | 32,744 T=32768 x32).

| task | 4,096 | 8,192 | 16,360 | 32,744 |
|---|---:|---:|---:|---:|
| niah_single_1 | 100 | 99 | 98 | 99 |
| niah_single | 87 | 91 | 92 | 93 |
| niah_multikey | 43 | 53 | 44 | 38 |
| niah_single_3 | 28 | 30 | 23 | 28 |
| niah_multikey_2 | 27 | 25 | 35 | 28 |
| qa_2 | 16 | 12 | 17 | 15 |
| qa_1 | 3 | 7 | 8 | 15 |
| niah_multikey_3 | 10 | 4 | 13 | 5 |
| vt | 3 | 3 | 2 | 6 |
| niah_multivalue | 0 | 0 | 0 | 0 |
| niah_multiquery | 0 | 0 | 0 | 0 |
| cwe | 0 | 0 | 0 | 0 |
| fwe | 0 | 0 | 0 | 0 |
| **mean (13)** | **24.4** | **24.9** | **25.5** | **25.2** |

One task of thirteen clears the 85.6% RULER "effective length" bar. The mean is flat in
length (24.4 -> 25.2), which says the ceiling is capability, not context window. Four
tasks read exactly 0.0 at every length; a flat zero across four lengths is as consistent
with a scoring bug as with a real floor, and the Qwen3-0.6B-Base control that would
separate the two was staged but never ran. The 65,512 column started at 14:41 and was
lost to teardown.

## 7. Provenance, including one gap

Everything in `results_2026-09-21/` is read from a published artifact, not retyped from
memory. `data.py` carries the source per table; `mk_tables.py` generates `TABLES.md` from
`data.py` so the tables and the figure cannot diverge.

| table | source |
|---|---|
| x32 sweep, 4,096 / 16,360 | `evals/2026-09-21/e8_{4096,16360}_summary.csv` |
| x32 sweep, 32,744 | **run stdout only - CSV never uploaded** |
| x32 sweep, t64 / t128 columns | 2026-09-20 x32 sweep, run stdout |
| x64 sweep, 6 lengths | `evals/2026-09-21/ruler_len*_summary.csv`, config from `logs/128k_master.log` |
| RULER 13 tasks | `evals/2026-09-17/ruler_t64_*.csv`, `logs/2026-09-17_rulerfull_logs_master.log` |
| mixes | `data/mix*_MANIFEST.txt` |
| loss | `logs/128k_train.log`, `logs/2026-09-21_t128b_train.log` |
| checkpoints | `HfApi.repo_info(files_metadata=True)` |

The gap is worth stating plainly: the 32,744 cell is the headline number of the whole
session and it is the one cell with no published CSV behind it. The eval directory was on
the box when it auto-closed. The number is reproducible from the step-300 checkpoint,
which is on HF.

## 8. Checkpoints held (680M long-context lineage)

| run | steps kept | files | GB | fully resumable |
|---|---|---:|---:|---|
| `p1_680m` (4k->16k) | 200,400,600,800,1000,1200 | 12 | 42.98 | 600, 1200 |
| `t32_680m` (16k->32k) | 1200 | 4 | 11.21 | 1200 |
| `t64_680m` (32k->64k) | 1000,1200 | 2 | 4.84 | none (rank_0 only) |
| `t128_run3` (first 128k attempt) | 600,800 | 33 | 37.30 | 600, 800 |
| `t128b` (rebuilt mix) | 100,200,300,400,500 | 20 | 26.07 | 500 |
| **total** | | **71** | **122.40** | |

A fully resumable set is 16 files: `rank{0..7}.ckpt` plus `rank{0..7}.zero_g1` (ZeRO-1
optimizer shards). Everything else is rank_0 weights - enough to evaluate, not to resume.
`t128b` resumes from step 500 at LR 1.8e-4, flat, pre-anneal.

## 9. Open items

1. **Resume 500 -> 1000 at 3e-5** (`CP_REWARMUP_PEAK=0.05`) once the extra data lands,
   then anneal. This doubles as the test of section 4.
2. **Evaluate t128b at 65,512** from the step-300 and step-500 checkpoints.
3. **Re-run the 32,744 eval and upload the CSV**, closing the provenance gap.
4. **The four RULER zeros** - run the Qwen3-0.6B-Base control through our harness.
5. **`prep128.sh` is still broken**: `--include ... || true` swallows HTTP 404s, so a
   missing shard silently becomes a short mix. It produced the b6=0.11 target that came
   out at 0.0893.
6. **Evaluate every 100 steps** on the next rung.
7. Optional: HF export to `LlamaForCausalLM` for the official RULER harness and for a KV
   cache (`CP_GENERATE` has none today - full forward per token).
