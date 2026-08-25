# Claude Report - automatic checkpoint sync to Hugging Face

2026-08-25 - Built an automatic pre-teardown snapshot for the 600M checkpoint plus
the non-reproducible run output, after establishing that Git LFS on GitHub cannot
physically hold the file - CP (/home/blu-bridge25/CP) / Scripts/sync/ckpt-sync.sh,
.gitignore

## The blocker that redirected the request

The ask was "automatic setup using git LFS for the latest checkpoint". Git LFS on
GitHub cannot do it:

| limit | value | our file |
|---|---|---|
| LFS per-file, Free/Pro | 2 GB | **8.18 GB** |
| LFS storage, free tier | 1 GB | 8.18 GB per snapshot |
| LFS bandwidth, free tier | 1 GB/month | 8.18 GB up + 8.18 GB down per session |

The per-file limit is a hard pre-receive reject, not a quota that can be topped up,
so no amount of paid data packs makes it work. Data packs are $5/mo per 50 GB;
even with one, 8.2 GB up + 8.2 GB down is ~3 sessions/month of bandwidth, and LFS
objects are never garbage-collected, so storage grows without bound.

Slimming was checked and rejected: 8,175,962,282 B / 600M params = 13.6 B/param,
i.e. fp32 weights + Adam m + v. Weights-only fp32 is still ~2.4 GB (over the
limit); bf16 weights-only would fit but destroys exact resume, which is the whole
point when pausing mid-cosine.

**The user proposed Hugging Face, which is the right answer** and better than the
four alternatives offered (R2 / S3 / split-LFS / B2). HF model repos ARE git-lfs
repos, so it is literally the requested workflow, with a 50 GB per-file ceiling,
free egress, and a resumable chunked uploader.

## What already existed

`Scripts/sync/ckpt-sync.sh` was written in an earlier session with `lfs` and
`release` backends. It could never have worked here, for two reasons found by
testing it against a reproduction of the instance's actual layout:

1. **The glob never matched.** It looked for `${PREFIX}_step_*.ckpt` =
   `blumodelcp_step_*.ckpt`, but CheckpointManager writes the run into the
   filename: `blumodelcp_run4_step_800.ckpt`. On the instance it would have
   printed "found no ... .ckpt" and exited.
2. **"Latest per run" grouped by DIRECTORY.** All runs share one flat dir, so
   run100_step_1387 outranked run4_step_800 on step number alone and **run4 would
   never have been published** - the 30 GPU-hours would have been silently dropped
   in favour of the old 120M.

Both fixed: the glob is now `${PREFIX}*_step_*.ckpt`, and runs are keyed by
`<dir>|<run-token>` via an associative array.

## What was added

- **`hf` backend, now the DEFAULT.** Uses `hf upload-large-folder` (resumable -
  an interrupted 8 GB upload otherwise restarts from zero) to a private model
  repo. Auth is checked with `hf auth whoami` BEFORE hashing 8 GB.
- **`--pull`**, placed deliberately before the CKPT_DIR existence check, since a
  freshly rented box has no checkpoint dir and that is exactly when you restore.
  Extracts with `tar -xzk` so a half-finished local run is never clobbered.
- **Sidecars and artefacts.** The old script published only `.ckpt`. A checkpoint
  without its 66-byte `<prefix>_run<N>.meta` is not resumable - losing the meta
  costs the same as losing the 8.2 GB file. Now also publishes every `.meta` plus
  a tarball of `$CKPT_ARTEFACTS` (training-log CSVs, probe_run4, in2_logs,
  in2_arms_state, mspoe dirs) - all gitignored output that exists nowhere else.
- **`MANIFEST.txt`** so the Hub page says what the snapshot is without downloading
  8 GB.
- **Per-backend size ceilings** (hf 50000 / release 2000 / lfs 1900 MB) replacing
  the flat 1900 that refused every model above 120M.

## Mistakes made and fixed during the session

- Put the per-backend `MAX_MB` resolution BEFORE the arg-parsing loop, so
  `--backend lfs` inherited the hf default and the 2 GB limit silently stopped
  being enforced. Caught by a test asserting lfs must REFUSE. Moved after the loop.
- Offered four object-storage backends without considering HF; the user's
  counter-suggestion was better than all of them.

## The .gitignore trap that was closed

The file carried `!checkpoints_latest/` + `!checkpoints_latest/**`. Since `*.ckpt`
is a file pattern, the negation re-included checkpoints there, so a plain
`git add .` would stage 8 GB straight into a push GitHub rejects. Replaced with a
plain `checkpoints_latest/` ignore; the lfs/release backends are unaffected
because they already reach the dir with `git add -f`.

Also added earlier this session (instance side): `in2_data/` and
`.pretrain600.pid`, after a push was rejected for two 104 MB / 103 MB JSON shards.

## Verification

Five behaviours asserted against a reproduction of the instance layout (two runs,
flat dir, rotating steps, sparse 8.2 GB files):

1. lfs refuses the 8.2 GB file (1900 MB ceiling) - PASS
2. release refuses it (2000 MB ceiling) - PASS
3. hf accepts, totals 8.9 G across run4 + run100 - PASS
4. `CKPT_BACKEND=lfs` env selection honours its own ceiling - PASS
5. `--pull` fails fast with no repo id - PASS

Newest-per-run selection verified correct: run4_step_800 AND run100_step_1387 both
picked, step 775/750 correctly skipped.

## Usage

```bash
export CKPT_HF_REPO=<user>/bluscript-checkpoints
hf auth login                       # WRITE-scoped token

Scripts/sync/ckpt-sync.sh --dry-run # always look first
Scripts/sync/ckpt-sync.sh -m "600M run4 @ step 800"
Scripts/sync/ckpt-sync.sh --pull    # on the next instance
```

## Not done

Not run against the real instance - this session is on the laptop, and the
instance holds the actual 8.2 GB file. The script needs to be pulled there and
executed before teardown.
