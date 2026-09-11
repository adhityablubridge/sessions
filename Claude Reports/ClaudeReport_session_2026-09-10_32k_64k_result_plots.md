# 2026-09-10 - Plots for the 32k and 64k runs - CP

Request: plots of both runs done today (32k and 64k), separately then combined.

## Deliverables

| file | what |
|---|---|
| `Tests/bluscriptcp/mk_today_notebook.py` | generator, 15 KB; builds and executes the notebook |
| `today_results_2026-09-10.ipynb` | 13 cells, executed, 5 figures |
| `today_results_2026-09-10.html` | 376 KB self-contained, figures embedded as base64 |

Style matches `l3long_results_2026-09-10.html` (max-width 1000px, `#fcfcfb`, `#e4e4e1`).

## The five figures

1. 32k run, accuracy vs context length, one curve per checkpoint (750/1100/1650/2227)
2. 64k run vs its 32k parent, with 32,768 marked as the parent's hard ceiling
3. Combined, both runs on one axis
4. Slope in pp per doubling
5. Depth profile at each run's longest length

## Data

All read from the uploaded `summary.csv` files, never from prose. An earlier notebook in
this project hardcoded values transcribed from a markdown report and three were wrong.

```
32k run (b4), niah_single:
  step  750: 4096=75.0  8192=60.0  16360=45.0  32744=27.0   decayed (ext32k stage 1)
  step 1100:            8192=55.0  16360=40.0  32744=24.0   undecayed
  step 1650:            8192=64.0  16360=55.0  32744=33.0   undecayed
  step 2227:            8192=70.0  16360=58.0  32744=43.0   decayed (full epoch)
64k run (b5):
  step  695: 8192=67.0  16360=65.0  32744=52.0  65512=29.0  plateau
```

The invalid `l2` column in the `ext32k_len_*` dirs was filtered out: that script passed
`YARN_ORIG=16384 YARN=2` to both models, so l2 was read under a cache it never trained on.

## Findings

- 32k run gains grow with length: +10.0 / +13.0 / +16.0 pp at 8k / 16k / 32k. Combined
  Wilson bar is +-12.8 to 13.5, so only the 32,744 cell is individually resolvable.
- 64k extension reaches 29.0% +-8.8 at 65,512. The parent cannot be evaluated there at
  all -- its RoPE cache is 32,768 -- so that cell is absent, not zero.
- 64k also improves 16k (+7.0) and 32k (+9.0); 8k moves -3.0, inside noise.
- The 64k model's 8k->16k slope is -2.0 vs the parent's -12.0, the flattest in the
  project. Its 32k->64k slope of -23.0 is the new frontier edge.
- Both depth profiles are monotone rising, 0% at depth 0.00 and 90-100% at 1.00. Pure
  recency, not lost-in-the-middle (that would need a bump at depth 0 too).

## Caveats carried onto the figures

- Steps 1100/1650 and the 64k checkpoint are undecayed/plateau; drawn hollow. Step 1100
  reads below 750 for that reason, not from a regression.
- The 64k run is 42% of a b5 epoch, stopped on the 23:30 IST deadline. Not converged.
- The claim that decay improves retrieval is NOT made: the b4 decay tail moved loss only
  0.024 nats and the undecayed 1100->1650 interval improved 5x faster per step
  (+2.73 vs +0.52 pp/100 steps at 16k). An earlier ~5pp figure was retracted as confounded.
- `niah_single` is the easy RULER task; `val_backward` remains at chance in every arm.
- The dominant deficit is unchanged: 1.049 B params at 0.99 tokens/param.

## Environment note

matplotlib, nbformat, nbclient, ipykernel and huggingface_hub were all absent locally and
were installed into the pyenv 3.12.13 interpreter.
