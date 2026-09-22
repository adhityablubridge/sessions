# 2026-09-22 - 128k run #3 (t128c): the batch fix, and localising the collapse to steps 300-400

Workspace: CP, training on 8x H100 80GB (`/root/Adhi/BluTrain/dist/Context_Parallelism`).
Artifacts: `hf.co/unparallelled/BluScriptCP` under `checkpoints/t128c_step*`.

---

## 1. What was run and why

Run #2 (`t128b`) degraded between step 300 and 500 (100/99/89 -> 95/97/77 at
4k/16k/32k). I attributed that to gradient noise per optimizer step, on the strength
of one control: `t64` trained at the **identical** LR (1.8e-4 flat, `CP_REWARMUP_PEAK=0.3`
against the hardcoded 6e-4) and *improved* (90 -> 95 -> 99 at 32,744), while its grad
clip fired on 0.2% of steps against run #2's 12.4%.

The measured difference was sequences per optimizer step:

    t64      524,288 / 65,536  = 8 seqs/step
    t128b    524,288 / 131,072 = 4 seqs/step

So run #3 changed **one variable**: `CP_GLOBAL_BATCH` 524,288 -> 1,048,576, giving
8 seqs/step at the 128k window. Same parent (`t64`), same mix (`mix128_b6_35_640M`),
same LR, same YaRN (x32, ORIG=4096, invariant `32 x 4096 == 131072` asserted).

H100 needed `CP_SAVE_KV_BF16=1 CP_RING_BF16=1` - the user's probe showed baseline OOMs
and the two flags bring peak to 71 GB. Confirmed in the run at 73,975 MB of 81,559.

Steps 600 (609 usable at the new batch), ckpt every 50, `KEEP=4`.

## 2. Result: the batch fix was real but partial

`niah_single`, T=131,072, YaRN x32, ORIG=4,096, N=100, haystack 199,754,694 B
(byte-identical to every prior eval in this project).

| step | tokens | 4,096 | 32,744 |
|---:|---:|---:|---:|
| 300 | 315M | 99 | **92** |
| 400 | 419M | 97 | **75** |
| 450 | 472M | 93 | **69** |
| 500 | 524M | 94 | **76** |

Reference: t64 100/93 | run2 s300 100/89 | run2 s500 95/77 | old-t128 98/69

- **Step 300 is the best 128k checkpoint produced so far**: 92 at 32,744 against the
  t64 parent's 93, and above run #2's best of 89. The larger batch raised the peak.
- **Then it collapses**: 92 -> 75 -> 69 within 150 steps. By step 450 it is exactly
  the old broken-mix score. The -17 and -23 moves are far outside the +/-9-13pp Wilson
  band at N=100.
- **4,096 degrades too** (99 -> 93). Retrieval at 4,096 cannot be a long-context data
  problem, so something is damaging the model globally.

**My hypothesis was half right.** Bigger batches halved the clip rate and delayed the
collapse by roughly 100 steps from a higher peak, but did not prevent it.

| run | seqs/step | clip fires (norm>1) | max norm | best 32,744 |
|---|---:|---:|---:|---:|
| t64 | 8 | 0.2% | 2.20 | 93 (improving) |
| t128b | 4 | 12.4% | 7.38 | 89 |
| **t128c** | **8** | **6.8%** | 6.34 | **92** |

## 3. Loss and retrieval move together

50-step trailing mean:

| step | loss | 32,744 |
|---:|---:|---:|
| 300 | 2.41 | 92 |
| 350 | 2.53 | - |
| 400 | 2.13 | 75 |
| 450 | 2.01 | 69 |

Loss is flat at ~2.4-2.5 through step 350, then falls exactly as retrieval falls. That
is the shortcut signature from run #1, not learning.

**Token-matched, run #3 is still the healthiest of the three** - at 262M tokens it reads
2.53 where run #1 had already collapsed to 1.52 and run #2 was 2.33. The collapse is
later and shallower, but it is the same collapse.

Note: comparing loss-vs-*step* across runs with different `CP_GLOBAL_BATCH` is an axis
artifact and inverts the conclusion. Compare on tokens consumed (`step x global_batch`).

## 4. What is now the prime suspect

Not batch size. Remaining candidates, in order:

1. **Learning rate.** 1.8e-4 flat with no decay, held long enough for the model to find
   the degenerate solution. The ladder note specifies 3e-5 for this rung. t64 survived
   1.8e-4 but only ran 1200 steps at a 65,536 window on diverse data.
2. **Domain skew.** 35% of every batch comes from b6 = 597 documents across four
   institutional sources (USGPO, Biodiversity Heritage Library, Library of Congress,
   USGS), while the eval haystack is CommonCrawl-derived web text. The mismatch
   compounds with training.
3. **Diversity exhaustion.** The collapse begins near where b6's small document pool
   would start repeating patterns.

## 5. Checkpoints

25 files, 36.68 GB on HF. Steps 50-500 rank_0 every 50; **step 300 as a full 16-file
resumable set** (8 `.ckpt` + 8 `.zero_g1`).

This mattered: `KEEP=4` rotated step 300 off local disk around step 500. Without the
shipper the best checkpoint of the run would have been destroyed - the same mechanism
that cost steps 200/400 of run #1.

Gap: step 500's full resumable set is local only (rank_0 on HF). Not worth uploading -
step 500 is the degraded checkpoint.

## 6. Own errors this session

- **Shipped a broken shipper.** `pgrep -x bluscriptCP_exec` never matches because Linux
  truncates `comm` to 15 chars (`bluscriptCP_exe`), so the exit condition was permanently
  true and it would have quit ~105 s in. `pgrep -f` is also wrong here - three orphaned
  watcher loops from a prior session carry the literal string in their own command lines.
  Correct predicate is argv[0]-as-path: `ps -eo args | grep -qE '^[^ ]*build/bluscriptCP_exec'`.
  **Fourth occurrence of this trap in this project.**
- **Three bugs in the eval script at once**: `PY` defaults to `python3` which has no
  tiktoken/numpy (venv required); `CKPT_EXTRA` takes a bare filename because the harness
  prepends `rank_0/`; and `local S=$1 ... O="ec_s${S}_..."` expands `$S` before `local`
  assigns it, which aborts under `set -u`.
- **ETA wrong by 2x.** I derived 37 s/step from `probe128.sh`, which runs `CP_MAX_STEPS=3`
  - all three steps are cold. Real steady state was 16.0 s/step. Probes measure fit, not
  speed.
- **Checked completion with a grep that matched the previous failed run's "EVAL DONE".**

## 7. Next

1. Resume from `t128c_step300` (full resumable set on HF) at `CP_REWARMUP_PEAK=0.1`
   (6e-5), real decay tail, hard cap ~300-400 steps. This tests the LR hypothesis.
2. Do not raise b6 %. t64's 50% contiguity was 50% *diverse* CommonCrawl long data;
   ours is 35% from 597 institutional documents. Raising it deepens the only skew we have.
3. SPLiCe-style similarity-grouped packing - builds long windows from *related* shorter
   documents, unlocking arxiv/Libgen/openalex/CommonCrawl at 128k. No new corpus needed.
4. Code-by-repo corpus. Absent from `longdoc.md` entirely; ProLong's largest single
   weight (0.30). Longest lead time, so start acquisition independently.
5. `prep128.sh` still has `--include ... || true` swallowing HTTP 404s.
