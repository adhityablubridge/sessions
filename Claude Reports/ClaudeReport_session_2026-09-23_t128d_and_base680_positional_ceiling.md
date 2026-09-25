# 2026-09-23 - t128d (b6 60%) failed, and the base model's positional ceiling was found

Workspace: CP, 4x H200 (torn down). Checkpoints on `hf.co/unparallelled/BluScriptCP`.
Local record: `results_2026-09-23/data.py`.

**Teardown status: checkpoints safe, eval CSVs LOST.** 22 t128d files / 39.39 GB
are on HF with full resumable sets at steps 300 and 360. The train log, every eval
CSV from today, and all scripts existed only on the box. The numbers below are
transcribed from run output and are the only surviving record.

---

## 1. t128d: the b6 60% mix made things worse

Hypothesis was that b6 is the only bucket with 128k-range structure (3.01x vs
~1.0x for every other bucket), so raising it 35% -> 60% would hold the curve flat.

Matched steps, same batch / LR / parent, mix the only difference, @32,744 x32:

| step | t128c (b6 35%) | t128d (b6 60%) |
|---:|---:|---:|
| 100 | 84 | 75 |
| 200 | **92** | 61 |
| 300 | **92** | 62 |

t128c rose 84 -> 92 and held. t128d fell 75 -> 61 and plateaued. Worse at every
matched step. The hypothesis is falsified; 35% is near the optimum of an
inverted U (8.9% -> 69, 35% -> 92, 60% -> 62).

Also falsified: my "flat loss = healthy" claim. t128d's loss WAS flat
(2.53 -> 2.36) and it was the worst run.

## 2. The real result: effective context is 32k, not 128k

Native config (T=131,072, YaRN x32, ORIG=4,096, fixed across lengths -- which is
what RULER does via vLLM static YaRN):

| length | virtual | t128c s300 | t128d s360 | >= 85.6? |
|---:|---:|---:|---:|:--:|
| 4,096 | 128 | 99 | 99 | yes |
| 8,192 | 256 | 97 | 96 | yes |
| 16,360 | 511 | 99 | 93 | yes |
| 32,744 | 1,023 | **92** | 71 | yes |
| 65,512 | 2,047 | **45** | 41 | **no** |
| 131,000 | 4,094 | **28** | 32 | **no** |

**Effective context length = 32,744 by RULER's 85.6% criterion -- a quarter of
the trained window.** This is the first time either 128k model was measured at
65,512 or 131,000; every prior headline was at 32,744 or below.

## 3. base680's positional ceiling

Pulled the true base from `navingv/varybatch` (1,362,674,564 B = 681,328,128 x 2,
bf16 weights). 14-cell grid of (scale, real length):

Rows = virtual position, cols = real length:

    virtual ~1,024 :  96  96  89  88     (real 1k, 2k, 4k, 8k)
    virtual ~2,048 :  95  83  84  67     (real 2k, 4k, 8k, 16k)
    virtual ~4,090 :  70  60  40  34     (real 4k, 8k, 16k, 33k)

Pure scale=1 diagonal (virtual == real): 512->97, 1024->96, 2048->95,
3072->88, **4096->70**.

Two findings:

1. **The base cannot use its own pretrain window.** Pretrained at 4,096, scores
   70 there vs 95 at 2,048. Usable range is ~2,048-3,072.
2. **Virtual position dominates, real length is secondary.** At fixed virtual
   ~1,024 the score barely moves across an 8x span of real length (96/96/89/88).
   At fixed real length, moving virtual 1,024 -> 4,096 costs 19-28 points every
   time. That is a RoPE signature, not a data or attention-span effect.

This is consistent with `rope_theta = 500,000` at a 4,096 pretrain leaving 32 of
64 RoPE dims without a full rotation. Llama 2 used 10,000 at 4,096; Llama 3 uses
500,000 but pretrains at 8,192 with 800B long-context tokens.

**Caveat I have to state**: base680 (96/95/70) and t128c (92/45/28) do NOT have
the same shape at matched virtual position, and the comparison is confounded --
base at virtual 2,048 is 2,048 real tokens, t128c at virtual 2,047 is 65,512.
The clean test is base680 at x32 on the same real lengths. It was launched and
the box went away before it finished. **That test is still outstanding and is the
single thing that would confirm or kill the theta diagnosis.**

## 4. A ladder bug found while checking

| rung | T | scale | ORIG | scale x ORIG | |
|---|---:|---:|---:|---:|---|
| p1 | 16,384 | 4 | 4096 | 16,384 | ok |
| **t32** | **32,768** | **32** | **4096** | **131,072** | **4x over-compressed** |
| t64 | 65,536 | 16 | 4096 | 65,536 | ok |
| t128* | 131,072 | 32 | 4096 | 131,072 | ok |

`low=14, high=32` in every log is the ORIG=4096 signature. t32 should have used
scale=8. For all 1,200 steps of the 16k->32k rung the model saw only virtual
0-1,023 -- and t64 descends from t32, t128* from t64.

This cannot explain base680's own 95 -> 70 decay (base predates t32) but may
compound it. Re-running the ladder correctly is days; re-pretraining is weeks.

## 5. MFU across the lineage (680M)

| run | T | GPU | dt ms | tok/s/GPU | GF/tok | MFU |
|---|---:|---:|---:|---:|---:|---:|
| p1 4k->16k | 16,384 | 2 | 7,843 | 33,423 | 7.7 | 26.1% |
| t32 16k->32k | 32,768 | 2 | 10,402 | 25,202 | 11.3 | 28.9% |
| t64 32k->64k | 65,536 | 8 | 5,267 | 12,442 | 18.6 | 23.4% |
| t128 old | 131,072 | 8 | 8,010 | 8,182 | 33.1 | 27.4% |
| t128b | 131,072 | 8 | 8,019 | 8,173 | 33.1 | 27.3% |
| t128c | 131,072 | 8 | 16,083 | 8,150 | 33.1 | 27.2% |
| t128d | 131,072 | 4 | 31,825 | 8,237 | 33.1 | 27.5% |

MFU is flat at 23-29% across every context length and both model sizes -- the CP
implementation does not degrade with length. At 131,072, attention is 87% of all
FLOPs. 4x H200 matches 8x H100 per GPU (8,237 vs 8,150 tok/s): compute-bound,
H200 bandwidth buys nothing.

## 6. BM25 coherence is not a valid proxy - retracted

| | far/cross |
|---|---:|
| b6 institutional | 3.01x |
| b5 CommonCrawl | 1.26x |
| b4 / b3 / b1 | 0.97 / 1.01 / 0.99 |
| mix b6_60 | 2.58x |
| mix b6_35 | 2.20x |
| mix splice30 | 1.91x |
| SPLiCe k=1 / k=6 | 1.51 / 1.66 |

It predicted the wrong direction twice: SPLiCe (3.45x whole-doc, useless) and
b6 60% (2.58x, worst run). Do not use it to drive mix decisions.

## 7. Errors I made today

- Proposed the LR test twice after the 2026-09-18/19 ablation had already shown
  LR schedule shape is a NULL on retrieval (+1/+1/+2pp). Should have read it.
- Compared t128c at x32 against t64 at x64 and called it an 18-point regression.
  Different configs. Same class of error as 2026-09-16 and 2026-09-20.
- Claimed the 128k models "inherit exactly this shape" from base680. They do not,
  and the comparison was confounded by real length.
- Claimed "flat loss = healthy". t128d falsified it.
- Reintroduced the `local S=$1 ... O="${S}"` bug a third time (expands before
  local assigns, aborts under set -u).
- Ran evals at 1 job/GPU using 4 GB of 143 GB. The harness documents
  JOBS_PER_GPU oversubscription at run_longeval.sh:163.

## 8. Next

1. **base680 at x32** on 4,096-131,000. Confirms or kills the theta diagnosis.
   Nothing else should be decided first.
2. If theta is confirmed: new pretrain at theta 10,000-50,000 for 4,096, or
   pretrain at 8,192+. Cannot be retrofitted.
3. If exonerated: re-run the ladder from p1 with t32 corrected to scale=8.
4. Upload eval CSVs BEFORE teardown next time. Today cost ~2 GPU-hours of
   measurements that now exist only as transcribed numbers.
