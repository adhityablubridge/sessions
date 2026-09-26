2026-09-25 - 21:40 - 1.59B 4k->16k YaRN extension; found and fixed a defect in the NIAH haystack that had been depressing every score in the project; re-scored the 680M base runs and the whole extension ladder clean, each checkpoint at its own native YaRN scale - CP / bluscript_zero_ckpt.cpp, stage16k/run16k.sh, evalwork/niah_one.sh, results_2026-09-25/

# Session report

## 1. The NIAH haystack defect

`run_longeval.sh` built its haystacks by slicing raw corpus shards. Those shards are
document-separated with `<|endoftext|>` (token 50256). Depending on the window length,
85.9-97.2% of the sampled windows contained at least one EOT, which makes any needle
placed before it unreachable. Every NIAH number produced in this project before today
was measured through that bias.

Stripping EOT from the haystack took the 1.59B base model from 70 to 100 at T=4096.
The fix is at source in `niah_one.sh`: it now filters token 50256 and caches the result
as `<name>.noeot.bin`. Training data is untouched - it MUST keep EOT as the document
separator. Only the haystack strips it.

What this invalidates: every absolute NIAH reading before today. What it does NOT
invalidate: A/B differences, because both arms ate the same bias. The 2026-09-24
long-doc finding survives - it was +8 points contaminated and is +9 points clean.

Claims retracted as a result:
- "the 1.59B fails content-addressed retrieval" - it does not, it scores 100
- "70% at T=4096 is a gate failure" - it was the haystack
- "long-doc data does not help the 1.59B" - unmeasurable, the model is at ceiling
- "reach reverts after 600 more long-doc steps" - clean it is 91 -> 90, noise

![[fig1_1b59_contaminated.png]]

## 2. The 680M ladder, re-scored at native scales

Each extension checkpoint was previously read at whatever YaRN scale happened to be
convenient, which produced cross-config comparisons. Re-run with every checkpoint at
the scale it was trained with, the pattern is uniform: each rung reaches exactly 25%
of its nominal window.

  t64        x16, trained 65,536  -> effective 16,360
  t128c s300 x32, trained 131,072 -> effective 32,744
  t128d s360 x32, trained 131,072 -> effective 32,744

t64's headline "98" came from evaluating it at x64, four times beyond its training
scale. At its native x16 it scores 45 at 65,512.

![[fig2_680m_corrected.png]]

## 3. The 1.59B 4k -> 16k extension

Ran on 2x H200. Rather than wait on `bluscriptCP.cpp`, YaRN was ported into
`bluscript_zero_ckpt.cpp` (+104 lines, `yarn_rope_cache`, guarded so scale<=1 falls
through to the stock builder bit-identically). Patch saved as
`ext16k_2026-09-25/source/yarn_in_zero_trainer.patch`.

Config: YARN_SCALE=4, YARN_ORIG_MAXPOS=4096, BLU_T=16384, 190 steps, constant
lr 1.2e-4 (no warmup, no decay), c2 mix ratios, data length-filtered to >=16384
tokens per domain with sha1 dedup and straddling documents dropped.

Grad norm without YaRN was 1.3248; with YaRN 0.2996, a 4.4x reduction - the RoPE
cache, not the data, was the source of the gradient spike.

Result: 91.0% at 16,360. This is the first rung in the project to hold RULER's 85.6%
criterion across 100% of its nominal window rather than 25%. The depth-0.00 cell -
the needle furthest from the query, historically the first to fail - is 20/20.
Val loss improved in every domain over the 190 steps.

![[fig3_1b59_ext16k.png]]

## 4. Errors made this session

- The 21.9% yield figure: I used "fraction of documents >=16k among those >=4096" as
  if it were the raw yield. True raw yield for commoncrawl is 6.0%, so 69.1M filtered
  tokens not 263M. Forced 190 steps instead of the planned 200.
- Document-count interleaving: `pick_domain` tracked error on document counts, so
  commoncrawl landed at 0.290 against a 0.330 target because domains have different
  mean document lengths. Fixed by keying the error term on tokens.
- Wall-clock estimates wrong twice in the same direction (9.5h -> 25min, 2.7h -> 1h40m).
  In the first case I inherited an 18.3k tok/s figure, which is 1.2% MFU, without
  sanity-checking it.
- Reported wall clock at the minimum GPU count that fits rather than a realistic node.
- Over-corrected on the d=4096 cell: I said it invalidated results. It does not. It is
  a fair between-model comparison; only the absolute reading is confounded.

## 5. Operational notes worth keeping

- `BLU_CKPT_KEEP=3` pruned the parent checkpoint mid-run. `BLU_CKPT_WEIGHTS_ONLY`
  needs the parent inside `BLU_CKPT_DIR`, and the keep-policy prunes that same
  directory. Original survived only because a copy sat in `/root/ckpt8b`.
- Missing `tiktoken` makes `longeval_gen.py` fail, yet `run_longeval.sh` still prints
  `DONE -> summary.csv` over an empty file. A green log with zero evaluation.
- `GEN_DATA_ROOT` must be relative; an absolute path gets cwd prepended.
- `PROMPT_LEN` must be `T - 24`, else the generate path pads past CP_T.
- CP_SIZE does NOT shard the fp32 logits tensor - verified. The 131,000 rung OOMs on a
  single 26.4 GB allocation on an otherwise empty 80 GB card.
- `pgrep -f 'build/bluscriptCP_exec'` matches mpirun, not the trainer; and
  `pkill -f '<script>.sh'` kills the calling shell. Both bit me again.

## 6. Artifacts

  results_2026-09-25/data.py      all numbers, generated from 100 eval CSVs on HF
  results_2026-09-25/mk_plots.py  figure generator
  results_2026-09-25/fig1_1b59_contaminated.png
  results_2026-09-25/fig2_680m_corrected.png
  results_2026-09-25/fig3_1b59_ext16k.png
  results_2026-09-25/_hf/         the 100 source CSVs

Checkpoints and eval CSVs are on hf.co/unparallelled/blutrain-680m-reach.
