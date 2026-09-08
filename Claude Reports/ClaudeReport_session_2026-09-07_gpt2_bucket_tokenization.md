# Claude Report - 2026-09-07 - GPT-2 streaming tokenization of all 7 token buckets

Date - time: 2026-09-07, 12:41 - 14:54 (local)
One line: Streamed GPT-2 (tiktoken) tokenization of the complete token_buckets corpus (all 7 buckets, 99.93 B tokens) into flat uint16 token bins on 10.101.0.203, with exact-count verification against the pre-computed per-document token counts.
Workspace: CP (local driver) / server 10.101.0.203
Files: tokenize_buckets_stream.py, verify_tokenized.py, progress.py (all new, on server at /mnt/volgrp04/3rd_floor/adhitya/)

---

## Result

The entire corpus is tokenized: all 7 buckets, 672 shards, in two runs totalling
11.1 minutes of wall clock. Every bucket reproduces the expected document and
token totals from `summary.tsv` exactly.

| bucket | shards | documents | GPT-2 tokens | bin size | count mismatches |
|---|---|---|---|---|---|
| b0_lt_4096 | 96 | 71,254,943 | 73,592,113,263 | 147.2 GB | 0 |
| b1_4096_8192 | 96 | 2,571,614 | 13,977,775,051 | 28.0 GB | 0 |
| b2_8192_16384 | 96 | 634,512 | 6,990,738,341 | 14.0 GB | 0 |
| b3_16384_32768 | 96 | 142,657 | 3,027,863,374 | 6.1 GB | 0 |
| b4_32768_65536 | 96 | 28,682 | 1,269,474,202 | 2.5 GB | 0 |
| b5_65536_131072 | 96 | 10,594 | 962,061,863 | 1.9 GB | 0 |
| b6_gte_131072 | 96 | 790 | 114,466,659 | 0.2 GB | 0 |
| **total** | **672** | **74,643,792** | **99,934,492,753** | **187 GB** | **0** |

Every bucket is an exact MATCH against `summary.tsv` on both documents and tokens.
The document total, 74,643,792, is exactly the source corpus count stated in the
bucket README - so nothing was lost anywhere across the full pipeline.

Runs:
- run 1, 14:27:37, buckets b1/b2/b3, 288 shards, 48 workers, 2.7 min
- run 2, 14:44:03, buckets b0/b4/b5/b6, 384 shards, 64 workers, 8.4 min at 147.2 M tok/s

Output root: `/mnt/volgrp04/3rd_floor/adhitya/token_buckets_tokenized/<bucket>/`
Run logs: `/mnt/volgrp04/3rd_floor/adhitya/run_20260907_142737.log`, `run_20260907_144403.log`
Manifest: `/mnt/volgrp04/3rd_floor/adhitya/token_buckets_tokenized/manifest.json`
(the manifest is rewritten per run, so it currently describes run 2's four buckets)

## Blockers found and resolved before any tokenization ran

**1. The tagged script would have produced nothing.**
`/mnt/volgrp04/Sanjay/Flux_edu_classifier/tokenize_gpt2.py` globs
`os.path.join(INPUT_DIR, "*.jsonl")`, but every shard in these buckets is
`part_XX.jsonl.zst` (zstd level 9). The glob returns empty, so the script prints
"No files found!" and exits 0. It also opens files with plain `open()` and has no
decompression path at all. Not a partial failure - a silent no-op.

**2. `zstandard` Python module is not installed on the server.**
The `zstd` CLI is present at `/usr/bin/zstd`. I stream through
`subprocess.Popen(["zstd", "-dc", "-T1", src])` and read the pipe line by line,
which also moves decompression into its own process rather than competing with
the tokenizer for the GIL.

**3. Disk: the input volume is nearly full.**
`/mnt/lv_132tb` (where the buckets live) is 98% used with 1.4 TB free. The tagged
script writes *uncompressed* JSONL with a `tokens` list appended per record;
at 24 B tokens that is roughly 260-300 GB written onto an almost-full volume.
Output was placed on `/mnt/volgrp04` (19 TB free) per instruction.

**4. `gpt2_tokens` in the source records is a COUNT, not the token ids.**
The bucket README says each record "gains one field: gpt2_tokens", which reads
ambiguously. Inspected a record directly: `gpt2_tokens: int = 4111`. So the
tokenization work is genuinely required - the buckets carry only lengths.
This turned out to be the single most useful fact in the session, because it
gives a free per-document correctness oracle (see Verification).

## Design

Format chosen: flat `uint16` token stream plus an int64 offset index. GPT-2 vocab
is 50257, which fits uint16 (max 65535) losslessly. This is ~6x smaller than the
JSONL-with-tokens format and is directly memmap-able.

Per input shard, three files in `<out>/<bucket>/`:

- `part_XX.bin` - uint16 little-endian GPT-2 ids, documents concatenated
- `part_XX.idx.npy` - int64 cumulative offsets, length ndocs+1; document `i` is
  `bin[idx[i]:idx[i+1]]`
- `part_XX.meta.json` - per-shard counts, written LAST and used as the resume marker

Streaming and safety properties:

- Peak memory per worker is one document plus a bounded 8 M-token (16 MB) flush
  buffer, held in an `array('H')` rather than a Python int list. Shards are never
  materialised. 48 workers stayed well inside RAM on a 660 GB box.
- Outputs are staged as `.partial` and renamed only on shard success; a shard that
  raises deletes its temps and is reported as failed without a meta file. Relaunching
  the identical command skips completed shards and retries only failures.
- Jobs are sorted largest-shard-first so the tail of the run is not one straggler.
- `RAYON_NUM_THREADS=1` is set before importing tiktoken. Without it, tiktoken's
  Rust core spins a rayon pool inside every one of the 48 workers, which multiplies
  into hundreds of threads on a 512-core machine. This is the specific mechanism
  behind the "CPU crash" risk that motivated the streaming requirement.
- Workers run at `nice 10`; the box had 8 other active users. Load peaked at 76 of
  512 cores.
- `encode_ordinary` is used, which is equivalent to the original script's
  `encode(text, disallowed_special=())` - both treat `<|endoftext|>`-like text as
  ordinary tokens - and is faster.
- No EOT separator is inserted by default (a `--eot` flag exists). Document
  boundaries are preserved exactly by the index, so a separator can be chosen at
  pack time; omitting it also keeps slice lengths directly comparable to
  `gpt2_tokens` for verification.

## Verification

Three independent layers, all passing:

1. **Per-document count oracle.** Every document's `len(encode_ordinary(text))` was
   compared against the record's own pre-computed `gpt2_tokens`. Across all
   **74,643,792 documents: 0 mismatches**. This confirms the encoding matches the
   one used to build the buckets, over the entire corpus rather than a sample.
2. **Index integrity** (`verify_tokenized.py`): `len(bin) == idx[-1]`, `idx[0] == 0`,
   `idx` strictly increasing, dtype int64, `max(token_id) < 50257`. Pass on all
   shards checked.
3. **Exact text roundtrip.** `enc.decode(bin[idx[i]:idx[i+1]])` compared byte-for-byte
   against the source `content` field: 115/115 exact matches across `part_00` of all
   seven buckets. Pass on every bucket.

Filesystem check: 96 `.bin` + 96 `.idx.npy` + 96 `.meta.json` in each of the 7
buckets (672 of each), 0 leftover `.partial` files.

## Error of my own, corrected before the run

My first version computed the running offset as `offsets[-1] + len(tokens_buf)`.
That is correct only until the first buffer flush - after a flush clears
`tokens_buf` while `offsets[-1]` still holds the cumulative total, every subsequent
offset is inflated by the pre-flush total, so the index would have been silently
wrong for any shard exceeding the 8 M-token flush threshold (i.e. all of them).
Replaced with an explicit monotonic `n_tokens` counter. The exact-text roundtrip
check in Verification is what would have caught this had it shipped, which is why
that check exists rather than only the count check.

Two smaller fixes in the same pass: `np.save` appends `.npy` to any path lacking it,
which would have produced `part_XX.idx.npy.partial.npy` (now writes through an open
file object); and `numpy` was imported inside the worker function rather than at
module scope.

A third, in `progress.py`: the rate and ETA divided the *cumulative* token total by
the *current* run's elapsed time, so launching run 2 over an output tree that already
held b1/b2/b3 credited it with those 24 B tokens and displayed "1242 M tok/s, ETA
1.0 min" seconds after start. Now only shards whose `.meta.json` mtime is at or after
the run's start timestamp count toward the rate, while the bucket bars still show
cumulative state.

I also mis-stated the b4+b5+b6 estimate once by 10x (used 23.45 B tokens instead of
2.35 B), reporting 2.6 min for what is 16 s of compute.

## Measured throughput

Per-worker rates differ by bucket because per-document JSON parsing cost is amortised
over fewer tokens in the short-document buckets:

- b1/b2/b3: 3.13 M tok/s per worker (avg 7,166 tokens/doc)
- b0: 2.75 M tok/s per worker, 2,660 docs/s (avg 1,033 tokens/doc)

Run 2 sustained 147.2 M tok/s aggregate on 64 workers. Load peaked at 76 of 512 cores
across both runs, with 8 other users active throughout. Disk use on volgrp04 went from
19 TB to 18 TB free.

## Live monitoring

`progress.py` reports run state at any time, during or after:

```bash
cd /mnt/volgrp04/3rd_floor/adhitya
python3 progress.py                # one snapshot
python3 progress.py --watch 5      # refresh every 5s
```

It counts completed shards from their `.meta.json` markers, sums documents and tokens
against `summary.tsv`, and shows a per-bucket bar, the running mismatch count, in-flight
shard count, current-run M tok/s and an ETA while workers are alive. The raw per-shard
stream is `tail -f /mnt/volgrp04/3rd_floor/adhitya/run_*.log`.

## Access note

SSH key auth to `blubridge@10.101.0.203` is rejected (host offers publickey and
password only; neither loaded key is authorized). `sshpass` is not installed
locally. Used OpenSSH 8.9's `SSH_ASKPASS` + `SSH_ASKPASS_REQUIRE=force` with
`setsid -w` instead, reading the password from a 0600 file so it never enters argv
or shell history:

- `~/.ssh/askpass_203.sh` - password provider (mode 700)
- `~/.ssh/.pw_203` - password file (mode 600)
- `~/.ssh/s203` - ssh wrapper with a shared ControlMaster (8 h persist)
- `~/.ssh/scp203` - scp wrapper reusing that ControlMaster

Adding the public key to the server's `authorized_keys` would make all of this
unnecessary.

## Reproducing / extending

```bash
# what was actually run (resumable - rerun to retry only failed shards)
cd /mnt/volgrp04/3rd_floor/adhitya
python3 -u tokenize_buckets_stream.py --workers 48
python3 -u tokenize_buckets_stream.py --workers 64 \
  --buckets b0_lt_4096 b4_32768_65536 b5_65536_131072 b6_gte_131072

# smoke test without writing a full bucket
python3 -u tokenize_buckets_stream.py --output-root ./_smoketest \
  --buckets b3_16384_32768 --workers 4 --max-docs-per-shard 200

# verify any shard
python3 verify_tokenized.py <src.jsonl.zst> <out_stem_without_suffix> 25
```

Reading a document back:

```python
import numpy as np
idx = np.load("token_buckets_tokenized/b1_4096_8192/part_00.idx.npy")
tok = np.memmap("token_buckets_tokenized/b1_4096_8192/part_00.bin",
                dtype=np.uint16, mode="r")
doc_i = tok[idx[i]:idx[i + 1]]
```

## Remaining decision for the training side

No EOT separator was inserted (the `--eot` flag exists but was not used). Document
boundaries are preserved exactly by the index, so the separator choice can be made at
pack time. This is the one open item: whichever packing step consumes these bins must
insert its own document delimiter, or sequences will run across document boundaries
without a break token.

---

## Addendum - Hugging Face upload of b1/b2/b3

Uploaded the tokenized bins for the three originally-requested buckets to a **new
private dataset repo**, `unparallelled/BluScriptCP-data`.

- 15:08:32 - 15:55:37, **47 min** wall clock, 48.0 GB, 864 data files
- https://huggingface.co/datasets/unparallelled/BluScriptCP-data

| bucket | files | size |
|---|---|---|
| b1_4096_8192 | 96 bin + 96 idx.npy + 96 meta.json | 28.0 GB |
| b2_8192_16384 | 96 bin + 96 idx.npy + 96 meta.json | 14.0 GB |
| b3_16384_32768 | 96 bin + 96 idx.npy + 96 meta.json | 6.1 GB |
| root | README.md, .gitattributes | - |
| **total** | **866 files** | **48.0 GB** |

### Decisions taken

The obvious target, `unparallelled/BluScriptCP`, is an existing **public, ungated
model repo** (2,338 files: checkpoints plus project docs, including Claude Logs and
an EOD report PDF). Pushing 48 GB of CommonCrawl-derived corpus there would have
published it irreversibly - HF retains git history, so a later delete does not undo
distribution. Raised this rather than proceeding; the call was a new private dataset
repo, which is also the HF convention for data. Note there was no dataset repo named
`BluScriptCP` (the 200 was on the *models* API; the datasets API returned 401).

Only the three requested buckets were uploaded. `b0_lt_4096`, `b4_32768_65536`,
`b5_65536_131072`, `b6_gte_131072` are excluded by an explicit `allow_patterns`
whitelist rather than by omission, so a future rerun cannot pick them up by accident.
`manifest.json` is also excluded: it is rewritten per run and currently describes the
b0/b4/b5/b6 run, so it would have been actively misleading in a b1/b2/b3 repo.

### Verification of the upload

- Remote reports `private: True`, 866 files.
- Exactly 96 `.bin` + 96 `.idx.npy` + 96 `.meta.json` per bucket, all three buckets.
- Remote sizes 28.0 + 14.0 + 6.1 = 48.0 GB, matching local exactly.
- **Content integrity**: remote LFS `sha256` compared against locally computed
  sha256 for 4 files (the three `part_00.bin`, 324/166/76 MB, plus one `idx.npy`) -
  **4 matched, 0 mismatched**. The bytes on HF are the bytes produced locally.

### Tooling

- `hf_upload.py` - `upload_large_folder` (multi-threaded, batched commits,
  resumable: rerunning re-hashes locally, skips what is remote, continues).
  Token read from a 0600 file, never in argv. `--dry-run` lists the selection first.
- `hf_transfer` 0.1.9 installed (`pip install --user`) for the Rust uploader.
- `hfwatch.sh` - 15 s refresh of counters, computed MB/s, ETA; reports if the
  process dies.
- A dataset card was written covering the format spec, a memmap usage snippet, the
  verification results, the absent-EOT caveat, and the excluded buckets. It asserts
  `license: other` - **this needs your review**, as the data is CommonCrawl-derived
  and I did not pick a license on your behalf.

### Estimation errors of my own

Three, worth recording because the pattern is the same each time - extrapolating
from a sample taken during a transition:

1. First throughput sample read **4 MB/s** and I projected 3.3 hours. That window
   straddled the hashing-to-upload transition and was not representative.
2. Second reading gave 15 MB/s, then a clean 120 s window gave **29 MB/s**.
3. From the 29 MB/s figure I projected "~15 min" remaining. Actual was longer: the
   **commit phase for the 576 large files took far more than the "few minutes" I
   allowed**. Total run was 47 min. Data transfer is not the whole cost of an HF
   upload of many large files, and I should have measured the commit rate separately
   instead of assuming it was negligible.

Also armed a log monitor whose regex contained a bare `429` (intended as the HTTP
rate-limit code); it matched `429kB/s` in the progress bars and fired false events
until I tightened it.

### Credentials note

The HF write token now also sits at `/home/blubridge/.hf_token` on the server
(mode 600), alongside the SSH password at `~/.ssh/.pw_203` locally. Both are
plain-text on disk and should be removed once the key-based SSH and any further
uploads are settled.

---

## Addendum 2 - Repack into FLUX layout for DataLoaderLite

Question asked: can `CP_DATA_ROOT` in `Scripts/Blutrain/bluscriptCP.cpp` just be
pointed at the tokenized bucket output? **No** - it would have thrown at startup,
and one of the five problems would have been silent.

### Why it would not load

Read from `BluTrain/Tensor-Implementations/include/DataLoader.h`:

1. **Filename filter.** `list_shards` (DataLoader.h:40) keeps a file only if
   `name.find(split) != std::string::npos`, i.e. the filename must contain "train"
   or "val". The bucket output is `part_00.bin`, so the loader throws
   `no .bin shards found for split train` at DataLoader.h:212.
2. **Non-recursive scan.** DataLoader.h:36 uses `fs::directory_iterator`, not
   `recursive_directory_iterator`. The bins live one level down in per-bucket
   subdirectories, so pointing at the output root finds nothing at all.
3. **No train/val split existed.** bluscriptCP.cpp:1740-1743 constructs both a
   `train_loader` and a `val_loader`; neither split had been produced.
4. **No EOT separators.** The loader slices flat contiguous `B*T` windows with no
   notion of document boundaries. Measured the existing corpus for comparison:
   `Data_Loader/BluWERP_data/train_shard_00001.bin` carries EOT (50256) every
   ~1,347 tokens. The bucket bins had **zero** - boundaries lived only in the
   `.idx.npy` sidecar, which this loader never opens. Training on them unrepacked
   would have taught the model to predict the first token of an unrelated document
   from the last token of the previous one, with no boundary signal.
5. **Silent 31% loss on b1.** bluscriptCP.cpp:1741 passes
   `max_tokens_per_shard=100000000` and DataLoader.h:78 does
   `tokens_ = std::min(total_tokens, max_tokens)` - truncation with no warning.
   **All 96 b1 shards exceed 100M tokens** (118,994,723 - 173,708,635), so roughly
   a third of b1 would have been dropped without any message. b2 (57.6-90.8M) and
   b3 (21.2-41.9M) were already under. The existing FLUX shards are 99,996,769 -
   99,999,585 tokens, i.e. deliberately sized just under that cap.

What *was* already compatible: the binary format itself. Raw `uint16`, no header,
mmap from offset 0, size divisible by 2 (DataLoader.h:71-81). This was a naming,
layout and separator problem, not a format problem.

### FLUX layout, measured

`Data_Loader/BluWERP_data/`: flat directory, 5 x `train_shard_%05d.bin` +
1 x `val_shard_00000.bin`, each ~200 MB = ~100M tokens. Per instruction the repack
keeps **1 val shard** (not the 1-in-6 ratio, which at 243 shards would have meant
~4B validation tokens).

### Result

`repack_for_cp.py`, 243 shards written in **20 seconds** on 32 workers.

| | |
|---|---|
| output root | `/mnt/volgrp04/3rd_floor/adhitya/cp_train_data` |
| shards | 242 `train_shard_%05d.bin` + 1 `val_shard_00000.bin` |
| documents | 3,348,783 (MATCH) |
| tokens | 23,999,725,549 (MATCH, includes one EOT per document) |
| tokens/shard | ~98.76M, max 98,770,441 - under the 100M cap |
| size | 48 GB |

### Interleaving

Documents are keyed by their fractional token position within their own bucket and
merged in key order, so any prefix of the stream holds all three buckets in their
true token proportions. Verified on four shards spanning the run:

| shard | b1 | b2 | b3 |
|---|---|---|---|
| train_00000 | 77.0% | 18.8% | 4.3% |
| train_00120 | 76.5% | 19.4% | 4.2% |
| train_00241 | 76.5% | 19.3% | 4.2% |
| val_00000 | 76.7% | 19.0% | 4.2% |
| corpus-wide | 76.8% | 18.9% | 4.3% |

Every shard is within 0.5 pp of corpus proportions. Without this the shards would
have been b1-only early and b3-only late, since the loader cycles shards in sorted
order - a length curriculum nobody asked for.

### Verification

`verify_cp_data.py` re-implements `list_shards()` and `UInt16ShardView::open()` in
Python and asserts what bluscriptCP.cpp actually hits at startup:

- `list_shards` finds 242 train and 1 val, and no filename matches both splits.
- Every shard size divisible by 2; **no shard exceeds the 100M cap**.
- EOT present in every sampled shard, mean gap ~7,200 tokens (consistent with the
  7,166-token mean document length of b1+b2+b3); every shard ends on an EOT.
- `max_id = 50256 < 50257`.
- Manifest token total equals bytes on disk.
- Decoded around a boundary to confirm the separator sits between two genuinely
  unrelated documents rather than mid-text.

RESULT: PASS.

### Usage

```bash
export CP_DATA_ROOT=/mnt/volgrp04/3rd_floor/adhitya/cp_train_data
# then run bluscriptCP as usual
```

Re-run the repack with a different split or shard size:

```bash
python3 repack_for_cp.py --val-shards 1 --target-tokens 99000000 --workers 32
python3 verify_cp_data.py --decode
```

### Open items

- This root covers b1+b2+b3 only (24.0B tokens). `b0_lt_4096` (73.59B) is not
  included; adding it means rerunning the repack with b0 in `BUCKETS`.
- The HF dataset repo holds the **per-bucket** format (bin + idx + meta), not this
  repacked layout. If the training instance should pull ready-to-train shards
  directly, the repacked root needs uploading too.

---

## Addendum 3 - Per-length CP data roots (corrects Addendum 2)

Addendum 2 built the wrong thing. I chose to **interleave** the three buckets into
one mixed root, on the reasoning that a length-ordered shard sequence would act as
an unintended curriculum. The actual requirement was the opposite: shards
**categorized by document length**, each category loadable on its own, each with its
own val shard, at FLUX's tokens-per-shard. Rebuilt accordingly.

### Layout

DataLoaderLite constrains the layout in two ways that decide this: it scans with
`fs::directory_iterator` (no recursion) and selects shards by the filename
containing "train"/"val". So a single flat directory cannot hold separated
categories - `list_shards` would match every category's train shards at once. The
answer is one flat, self-contained directory per bucket, selected by CP_DATA_ROOT:

```
/mnt/volgrp04/3rd_floor/adhitya/cp_data_by_length/
  b1_4096_8192/   train_shard_00000..00138.bin + val_shard_00000.bin   27 GB
  b2_8192_16384/  train_shard_00000..00068.bin + val_shard_00000.bin   14 GB
  b3_16384_32768/ train_shard_00000..00029.bin + val_shard_00000.bin  5.7 GB
```

### Result

`repack_by_length.py`, 241 shards in **36 seconds** on 32 workers.

| bucket | shards | train | val | documents | tokens | tokens/shard | max shard |
|---|---|---|---|---|---|---|---|
| b1_4096_8192 | 140 | 139 | 1 | 2,571,614 | 13,980,346,665 | 99,859,619 | 99,866,141 |
| b2_8192_16384 | 70 | 69 | 1 | 634,512 | 6,991,372,853 | 99,876,755 | 99,889,424 |
| b3_16384_32768 | 31 | 30 | 1 | 142,657 | 3,028,006,031 | 97,677,613 | 97,701,502 |
| **total** | **241** | **238** | **3** | **3,348,783** | **23,999,725,549** | | |

All three MATCH on documents and tokens. FLUX reference is 99,996,769 - 99,999,585
tokens/shard; every shard here sits just under that and well under the 100,000,000
cap, so nothing is silently truncated.

### Within-bucket interleaving

Documents are keyed by fractional token position within their **source shard** and
merged in key order, so each output shard draws proportionally from all 96 source
shards of its bucket. This matters mainly for the val shard: without it, val would
be a slice of two or three source shards rather than a sample of the whole bucket.

### Verification

`verify_by_length.py` re-implements `list_shards()` and checks each bucket
directory **independently**, since CP_DATA_ROOT points at one at a time:

- `list_shards('train')` / `list_shards('val')` return the expected counts in all
  three roots; no train filename also matches "val".
- Every shard size divisible by 2; **no shard exceeds the cap** in any bucket.
- EOT present, every shard ends on EOT, `max_id = 50256 < 50257`.
- **Length purity: 100.0% of documents fall inside the bucket's declared
  [lo, hi) range**, in every sampled shard of every bucket - b1 min 4096 / max 8191,
  b2 min 8192 / max 16382, b3 min 16387 / max 32753. This is the check that the
  categorization actually holds.
- Decoded across a boundary in each bucket to confirm EOT separates unrelated
  documents.

RESULT: PASS - each bucket loads independently.

### Usage

```bash
export CP_DATA_ROOT=/mnt/volgrp04/3rd_floor/adhitya/cp_data_by_length/b3_16384_32768
# or b1_4096_8192 / b2_8192_16384
```

Rebuild with different sizing or more val shards per category:

```bash
python3 repack_by_length.py --val-shards 1 --target-tokens 99999000 --workers 32
python3 verify_by_length.py --decode
```

### Housekeeping

`cp_train_data/` (the mixed root from Addendum 2, 45 GB) is superseded and can be
deleted. Left in place pending confirmation.
