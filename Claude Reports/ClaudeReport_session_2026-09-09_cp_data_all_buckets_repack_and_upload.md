# Claude Report - 2026-09-09 - All 7 bucket roots repacked for DataLoaderLite, b1-b6 on HF

Date - time: 2026-09-08 19:00 - 2026-09-09 14:45 (local)
One line: Completed the per-length CP data roots for every bucket (b0 through b6, 100.01 B tokens, 1,003 shards), uploaded b1-b6 to the private HF dataset repo, and diagnosed an upload failure caused by another user's disk benchmark filling the root filesystem.
Workspace: CP / server 10.101.0.203
Files: repack_by_length.py, verify_by_length.py, hf_upload_bylen.py, hf_upload.py, hfwatch.sh, build_combined_manifest.py, patch_upload.py

Continues `ClaudeReport_session_2026-09-07_gpt2_bucket_tokenization.md`, which covers
the tokenization itself and the first three bucket roots.

---

## Result

Every bucket is now a self-contained, flat `CP_DATA_ROOT` with its own train and val
shards, and all of them verify against DataLoaderLite's real constraints.

| bucket | token range | train | val | tokens | max shard | on HF |
|---|---|---|---|---|---|---|
| b0_lt_4096 | [0, 4096) | 736 | 1 | 73,663,368,206 | 99,953,973 | no |
| b1_4096_8192 | [4096, 8192) | 139 | 1 | 13,980,346,665 | 99,866,141 | yes |
| b2_8192_16384 | [8192, 16384) | 69 | 1 | 6,991,372,853 | 99,889,424 | yes |
| b3_16384_32768 | [16384, 32768) | 30 | 1 | 3,028,006,031 | 97,701,502 | yes |
| b4_32768_65536 | [32768, 65536) | 12 | 1 | 1,269,502,884 | 97,692,527 | yes |
| b5_65536_131072 | [65536, 131072) | 9 | 1 | 962,072,457 | 96,334,986 | yes |
| b6_gte_131072 | [131072, inf) | 1 | 1 | 114,467,449 | 57,334,448 | yes |
| **total** | | **996** | **7** | **100,009,136,545** | | |

`verify_by_length.py`: **RESULT PASS** on all 7 roots. 100.0% length purity in every
sampled shard, no shard over the 100,000,000 cap, EOT present and terminal, all token
ids < 50257, correct train/val counts, and no train filename that also matches "val".

The token total is 100,009,136,545, which is the 99,934,492,753 raw GPT-2 tokens plus
exactly 74,643,792 appended EOTs - one per document across the whole corpus. The two
figures reconcile to the token.

Usage:

```bash
export CP_DATA_ROOT=/mnt/volgrp04/3rd_floor/adhitya/cp_data_by_length/b5_65536_131072
```

## Hugging Face

Private dataset repo `unparallelled/BluScriptCP-data`, 1,133 files total:

- `b1_4096_8192/part_XX.bin` + `.idx.npy` + `.meta.json` etc - the per-bucket
  archival format from the previous session (866 files, 48.0 GB)
- `cp_data_by_length/<bucket>/train_shard_XXXXX.bin` + `val_shard_00000.bin` -
  loader-ready, b1 through b6 (267 files, 52.7 GB)

Verified against the remote rather than trusting the uploader's own report:
`private: True`, per-bucket train/val counts and sizes match local exactly, and
sha256 spot-checks on 10 shards across both uploads (including every val shard of
b4/b5/b6) matched local with **0 mismatches**.

b0 is server-local only, by instruction - it is 147 GB and would have been a 2-3 hour
upload.

Pulling one category:

```bash
hf download unparallelled/BluScriptCP-data --repo-type dataset \
  --include "cp_data_by_length/b3_16384_32768/*" --local-dir ./data
export CP_DATA_ROOT=./data/cp_data_by_length/b3_16384_32768
```

## Incident: upload failed on a full root filesystem

The `cp_data_by_length` upload for b1/b2/b3 transferred all 48 GB successfully, then
failed in a retry loop during the **commit** phase:

```
Failed to commit: I/O error: No space left on device (os error 28) at path
"/home/blubridge/.cache/huggingface/xet/.../staging/shard-session/.tmpXXXXXX"
```

Root cause was **not** our data and **not** the reboot. Another user was running a
disk benchmark: `/benchpool` held 32 x 8 GiB files (`p.0.0` .. `p.31.0`, 256 GiB),
root-owned, created 19:19. That took `/` from 254 GB free to 0 bytes. HF's Xet
uploader stages commit metadata under `~/.cache/huggingface`, which lives on `/`, so
every commit attempt hit ENOSPC.

Fix: set `HF_HOME` and `TMPDIR` to the large data volume, before the
`huggingface_hub` import so it actually takes effect. Verified the Xet cache then
resolves to `/mnt/volgrp04/3rd_floor/adhitya/hfcache/xet`. `/benchpool` was left
untouched - root-owned, not ours, and no sudo available. The relaunch showed
`pre-uploaded: 241/241 (48.0G/48.0G)` and only had to commit.

This is worth keeping in the scripts permanently: the root filesystem on this box is
879 GB and shared, so any HF operation that stages tens of GB must not default its
cache there.

## Server outage

The box was powered off for ~30 minutes mid-session. Notes for next time:

- On return, `sshd` accepted TCP on port 22 while `pam_nologin` still rejected every
  unprivileged login ("System is booting up"). **A port check is not a readiness
  check** - wait for an actual successful login.
- At 4 minutes into boot, `/mnt/volgrp04` was not yet mounted and `df` silently fell
  back to reporting `/`. Reading a data directory at that point makes intact data
  look **missing**. Check the mount before concluding anything about the contents.
- All data survived. Re-ran the full verification rather than assuming: identical
  token totals, zero `.partial` files.
- `upload_large_folder` resume state lives in `<folder_path>/.cache/huggingface`,
  i.e. on the data volume, so it survived the power cycle. The resume recovered
  113/241 files (22.5 GB) that had already transferred.

## b6 is thin - worth knowing before training on it

`b6_gte_131072` holds only 790 documents in total, so it produces just **2 shards:
1 train + 1 val**, ~57.2 M tokens each. That is a valid CP_DATA_ROOT and far above
any `B*T`, but with a single train shard the loader cycles the same shard
repeatedly, so there is no shard-level variety in that category. Its documents are
also extreme: median 139,237 tokens, max 704,226.

`b5` is similar in kind but less severe: 9 train shards from 10,594 documents.

## Combined manifest

`repack_by_length.py` rewrites `repack_manifest.json` on every invocation, so after
the b4/b5/b6 and b0 runs it described only the last bucket set - actively misleading
in a repo holding six. `build_combined_manifest.py` scans the shards actually on disk
and writes one manifest covering all 7 roots, with a `published_to_hf` flag per
bucket so b0's absence from the repo is explicit. It writes in place
(truncate-and-write) so the hardlink into the HF staging tree is preserved; verified
by inode before and after. The corrected manifest was pushed to HF.

## Estimation errors of my own

Recorded because the pattern repeats: I extrapolate from a sample taken during a
transition, or assume a phase is free.

1. **b0 repack: estimated ~10 minutes, took 128.** I scaled from b1-b3 (36 s for
   3.35 M documents) by document count, giving ~13 min. The real cost is the
   per-document Python loop over 71.25 M documents whose median length is only 804
   tokens (1.6 KB), so the work is seek- and interpreter-bound rather than
   throughput-bound, and the box was at load 50-69 from other users. Roughly 10x
   my estimate.
2. **HF commit phase treated as negligible.** Told the user "~15 min" from a 29 MB/s
   transfer measurement; the b1/b2/b3 run took 47 min because committing 576 large
   files dominated the tail. Transfer rate is not the whole cost of an HF upload.
3. **First throughput sample read 4 MB/s** and I projected 3.3 hours; that window
   straddled the hashing-to-upload transition. Sustained rate was 15, then a clean
   120 s window gave 29 MB/s.
4. **Claimed the interrupted upload had transferred nothing.** It had transferred
   22.5 GB of 48; I read an early status line and generalised.

Operational mistakes in the same period:

- `pkill -f hf_upload_bylen.py` matched **my own SSH command line** and killed the
  session (exit 255). Fixed by using a `[h]f_upload_bylen` pattern that cannot
  self-match.
- A log monitor's regex contained a bare `429` intended as the HTTP rate-limit code;
  it matched `429kB/s` in progress bars and fired false events. Later, a monitor
  filtering on `No space left` produced dozens of duplicate notifications from the
  retry loop - replaced with a single-notification waiter.
- `hfwatch.sh` had three defects that each produced a wrong conclusion: it printed
  "UPLOAD PROCESS NOT RUNNING" on **successful** completion (only checked liveness,
  not the success marker); `pgrep -cf hf_upload.py` does not match
  `hf_upload_bylen.py`, so it declared a healthy upload dead; and it derived the
  rate from a counter that only refreshes every 60 s, showing 0.0 MB/s on a 15 s
  loop. It now checks the `done:` marker, matches both script names, and reads the
  NIC's `tx_bytes`.

## Tooling

- `repack_by_length.py --buckets <b> ... [--val-shards N] [--target-tokens N]` -
  per-bucket repack. Interleaves documents within a bucket by fractional token
  position across its 96 source shards, so every output shard (the val shard in
  particular) samples the whole bucket.
- `verify_by_length.py [--decode]` - re-implements `list_shards()` and
  `UInt16ShardView::open()` in Python and checks each bucket root independently.
- `hf_upload_bylen.py --buckets <b> ...` - now takes a bucket list; the allow-list is
  derived from it, so an unstaged or unwanted bucket cannot be picked up implicitly.
  Uploads under a `cp_data_by_length/` prefix via a hardlink staging tree, because
  `upload_large_folder` has no `path_in_repo` and would otherwise dump shards into
  the existing per-bucket directories.
- `hfwatch.sh [log-glob]` - live counters, NIC-measured rate, ETA, and a correct
  success-vs-failure verdict.
- `build_combined_manifest.py` - regenerates the all-bucket manifest.

## Open items

- b0 is not on HF (147 GB). `hf_upload_bylen.py --buckets b0_lt_4096` after staging
  it would do it; budget 2-3 hours.
- The dataset card asserts `license: other` and still needs review, since the data is
  CommonCrawl-derived.
- Credentials in plaintext on disk: `/home/blubridge/.hf_token` (mode 600) on the
  server and `~/.ssh/.pw_203` locally. Adding the SSH public key to the server's
  `authorized_keys` removes the need for the latter.
- `hfstage/` is a hardlink tree, so it costs no real space, but it can be removed
  once no further uploads are planned.
- `/` on the server is shared and was filled to 100% by another user's benchmark.
  Worth watching if further large uploads are run.
