# Portable Claude sessions

Park Claude Code sessions in git so they survive an instance being destroyed, and
restore them onto **any** new box with one command — no per-box renaming.

---

## Why the naive version breaks

Three failures, all of which this setup removes:

1. **The project-dir name is box-specific.** Claude Code stores a session at
   `~/.claude/projects/<launch-dir-with-slashes-as-dashes>/<uuid>.jsonl`.
   `/home/me/CP` -> `-home-me-CP`, `/root/BluTrain` -> `-root-BluTrain`. Commit the
   *encoded* name and it is wrong on the next box, so `--resume` cannot find it.
   **Fix:** the repo stores one canonical `projects/CP/`. `pull.sh` derives the
   encoded name from the directory you will launch from and materializes it there.

2. **GitHub hard-rejects any file over 100 MB.** Session logs are append-only and
   grow fast; measured here at 72 MB, 71 MB, 57 MB — already past GitHub's 50 MB
   warning. **Fix:** the store holds `*.jsonl.gz` (72 MB -> 22 MB), plus a 95 MB
   guard in `push.sh` that fails loudly *before* GitHub does.

   **What gzip does NOT do: shrink the repo.** Measured on a 38.2 MB session —
   `gzip -9` gives 8.8 MB and git's own zlib gives 8.8 MB on the *raw* file. Git
   already compresses blobs, so a pre-gzipped file gains nothing in the packfile.
   Gzip buys **per-file headroom against the 100 MB limit and nothing else**. The
   repo-size win comes from pruning (point 3), not compression.

3. **Unbounded bloat — this is where the real saving is.** 51 sessions / 537 MB
   of logs had produced an 845 MB `.git`,
   plus a 604 MB `projects/.git-inner-backup` — a git repo committed into a git
   repo, the classic artifact of cloning into an already-populated `~/.claude`.
   **Fix:** `KEEP` prunes to the N newest sessions, `KEEP_SESSIONS` pins the ones
   you actually resume, `.gitignore` hard-blocks nested `.git`, and `--squash`
   keeps history from accumulating dead blobs. 6 newest sessions = **47 MB**.

---

## Layout

```
<repo>/
  sessions.conf          CANON / MODE / KEEP / KEEP_SESSIONS  (policy, shared)
  scripts/
    lib-sessions.sh      shared helpers (path encoding, guards, listing)
    pull.sh / push.sh    sessions: restore onto this box / save back
    logs-pull.sh         logs branch -> vault
    logs-push.sh         vault -> logs branch
  projects/CP/           canonical store: <uuid>.jsonl.gz, subagent sidecars,
                         .manifest.tsv, .owner
  rules/                 your CLAUDE.md rules (symlinked into ~/.claude/rules)

~/.claude-sync.local.conf   per-box paths (VAULT_DIR). Outside the repo,
                            so it can never be committed.
~/.claude-logs/             the 'logs' branch checkout (separate working tree)
```

`projects/-*` (the encoded, box-specific dirs) are gitignored — they are created
at pull time and never committed.

---

## On a fresh instance

```bash
git clone https://github.com/adhityablubridge/sessions.git ~/.claude-sessions
cd /root/BluTrain                      # the dir you will launch Claude from
~/.claude-sessions/scripts/pull.sh     # keys sessions to THIS dir, prints UUIDs
claude --resume <UUID>
```

That is the whole bootstrap. `pull.sh` self-locates its repo, so clone it anywhere.
Launch dir defaults to `$PWD`; pass `--dir /path` to be explicit.

**Pick the launch dir once and stay consistent** — VS Code's extension uses the
workspace root (`/root/BluTrain`), a terminal uses your `cd`. Different dirs mean
different encoded names, and `--resume` only sees the one it was keyed to. If you
switch, just re-run `pull.sh --dir <the-other-dir>`.

## Before destroying the instance

```bash
~/.claude-sessions/scripts/push.sh -m "stage 3 hopper fork"
```

Nothing else protects the history. Run it while Claude is **closed** (an open
session file gets snapshotted mid-write; the script warns if it detects one).

## Other commands

```bash
scripts/pull.sh --list            # store contents: uuid, date, size, preview
scripts/pull.sh --adopt           # absorb an existing live project dir (migration)
scripts/pull.sh --no-fetch        # skip git pull (offline)
scripts/push.sh --keep 4          # prune to 4 newest this run
scripts/push.sh --squash          # single-commit history; stops remote growth
scripts/*.sh --dry-run            # print every action, touch nothing
```

---

## Safety properties

- **Refuses to clobber.** If the target project dir already holds real sessions,
  `pull.sh` aborts and tells you to use `--adopt`. It never silently overwrites.
- **Never overwrites newer local work.** In compress mode a live `.jsonl` newer
  than its stored `.gz` is kept and flagged — push before pulling.
- **Single-writer warning.** `.owner` records host + timestamp on every push;
  `pull.sh` warns when the store was last written by a different box. This is a
  manual handoff, not live sync — do not work two boxes against one store.
- **Churn control.** `.manifest.tsv` records each session's size, so unchanged
  sessions are not re-compressed and re-committed (a rewritten 70 MB blob would
  add ~17 MB of dead git objects per push).
- **No secrets.** `.credentials.json`, `ide/`, `session-env/`, `shell-snapshots/`,
  `backups/`, `file-history/` are hard-excluded regardless of the whitelist.

## Modes

**You must run `push.sh` before destroying a box in EITHER mode** — data does not
leave the machine by itself. What the modes change is how many copies exist.

`MODE=compress` (default) — **two copies.** Claude writes the live `.jsonl`; the
repo holds a `.gz` snapshot from your last push. The repo copy goes stale the
moment you work, so `push.sh` has to re-compress live -> store before committing.
Forget it and the work is lost. Required at your current sizes.

`MODE=link` — **one copy.** `~/.claude/projects/<encoded>` is a symlink to the
store, so Claude writes straight into the repo working tree; nothing can go stale
and `push.sh` only commits (no copying). But it stores raw `.jsonl`, so it walks
into the 100 MB wall. Only for sessions well under that.

The real fix for "I might forget to push" is not link mode, it is a Stop hook that
runs `push.sh` when a session ends. See `.claude/settings.json`.

## One-time reset of an already-bloated repo

History has no value here (these are snapshots), so collapse it and reclaim space:

```bash
cd <repo>
git rm -r --cached projects/.git-inner-backup     # stop tracking the nested repo
rm -rf projects/.git-inner-backup
scripts/push.sh --dir /root/BluTrain --squash -m "reset: compressed session store"
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

## Failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `--resume` does not list the session | launched from a different dir than pulled to | `pull.sh --dir <that dir>` |
| push rejected, "file too large" | a session crossed 100 MB | drop it from the store, or `--keep` fewer |
| `pull.sh` aborts, "REAL project dir exists" | live sessions present | `--adopt`, or move them aside |
| "keeping NEWER live copy" warnings | unpushed local work | `push.sh` first, then pull |
| store warns about another host | another box pushed last | confirm that box is done |

---

# The `logs` branch (work log + session reports)

The Claude work log and session reports live on a **separate branch** of the same
remote, checked out into **its own directory** (`~/.claude-logs`). That separation
is deliberate: a branch switch inside the session repo would replace the whole
working tree, so sharing one checkout between sessions and logs would wipe your
sessions every time you pushed a log.

```
<repo>@logs/
  Claude Logs.md          one line per session (Rule 2)
  Claude Reports/         ClaudeReport_session_<date>_<context>.md (Rule 3)
  .gitattributes          *.md merge=union
```

## Per-box setup (once per machine)

Vault paths differ per box, so they go in an override that lives **outside** the
repo and therefore cannot be committed:

```bash
cat > ~/.claude-sync.local.conf <<'CONF'
VAULT_DIR="$HOME/Documents/obsidian Vaults/Adhitya's Vault"   # laptop
# VAULT_DIR="$HOME/claude-vault"                              # gpu box, no vault
CONF
```

## First time only - create the branch

```bash
scripts/logs-pull.sh --init      # stage the orphan branch locally
scripts/logs-push.sh -m "init logs branch"
```

## Every box after that

```bash
scripts/logs-pull.sh                       # logs branch -> vault
# ... work, log per Rules 2/3 ...
scripts/logs-push.sh -m "session log"      # vault -> logs branch
scripts/logs-pull.sh --list                # entry count + recent reports
```

## Updating logs from your laptop

The scripts are versioned on `main`, so the laptop needs them before it can sync
logs. **Order matters the first time:**

```bash
# --- on the box that has the scripts (once) ---
cd ~/.claude && git add -A && git commit -m "add session+log sync scripts" && git push
scripts/logs-pull.sh --init && scripts/logs-push.sh -m "init logs branch"

# --- on the laptop ---
cd ~/.claude && git pull                      # get the scripts
cat > ~/.claude-sync.local.conf <<'CONF'
VAULT_DIR="$HOME/Documents/obsidian Vaults/Adhitya's Vault"
CONF
scripts/logs-pull.sh                          # merges remote entries INTO your vault
```

Your existing vault log is **never overwritten** — see below. From then on it is
just `logs-pull.sh` when you sit down and `logs-push.sh` when you finish, on
whichever box you are using.

## Why a union merge

`Claude Logs.md` is append-only: each box adds a line. Without a merge driver two
boxes both appending produces a hard conflict on every sync. `*.md merge=union`
concatenates both sides instead. Lines can land out of order (each is dated, so
that is cosmetic) but a sync never fails and no entry is lost.

## Safety

**The log file is merged, never copied.** An mtime comparison is worthless here:
a fresh `git clone` stamps every file with the checkout time, so the repo's copy
always looks "newer" and a copy-based sync would replace a 200-line vault log with
a 1-line one. Instead `merge_log` unions the two — the destination keeps its own
order and gains only the lines it was missing. Verified: 12 vault lines + 1 repo
line -> 13, and re-running yields 13 again (idempotent, no duplication).

**Reports are `cp -n` (no-clobber).** Filenames are per-session unique, so an
existing report is always authoritative and is never overwritten.

Logs are kilobytes of markdown, so unlike sessions there is no compression, no
pruning and no size guard - push them as often as you like.
