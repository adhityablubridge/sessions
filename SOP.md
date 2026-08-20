# SOP - portable Claude sessions + logs

Three phases: a one-time setup on the box that has the scripts, a one-time setup
per new box, then a short daily routine. Commands assume the sessions repo is at
`~/.claude` (adjust if you cloned elsewhere - the scripts self-locate).

---

## Phase 0 - ALREADY DONE. DO NOT RUN AGAIN.

> Executed 2026-08-19 on `plain-machine-roars-fin-02`. Kept below for the record
> only. Running it on any other box **destroys that box's live sessions**:
> `rm -rf projects/-home-blu-bridge25-CP` is "stale" only on the box that
> authored this SOP; on the laptop it is the real, live CP session dir.
>
> `--squash` is likewise spent. It force-pushes an orphan commit, and on
> 2026-08-19 it silently discarded a commit another box had just pushed. With
> `KEEP` pruning active, squashing is what turns a prune into permanent loss.
> Every box after the first is a **Phase 1** box.

<details><summary>historical Phase 0 (do not run)</summary>

Run in order. Steps 1-2 are local cleanup, 3 publishes, 4 creates the logs branch.

```bash
cd ~/.claude

# 1. drop the accidental nested repo (604MB) and stale per-box project dirs
git rm -r --cached projects/.git-inner-backup 2>/dev/null || true
rm -rf projects/.git-inner-backup
rm -rf projects/-home-blu-bridge25 projects/-home-blu-bridge25-CP

# 2. populate the canonical store from the live sessions (takes a minute:
#    26 sessions / 538MB being gzipped). --adopt is required because a real
#    project dir already exists; the script refuses to touch it otherwise.
scripts/pull.sh --dir /root/BluTrain --adopt --no-fetch

# 3. prune to KEEP=6, then publish scripts + store on a single fresh commit.
#    --squash resets the 845MB history; without it the old blobs stay forever.
scripts/push.sh --dir /root/BluTrain --squash -m "reset: compressed session store + sync scripts"
git reflog expire --expire=now --all && git gc --prune=now --aggressive

# 4. create the logs branch and publish today's log + report
scripts/logs-pull.sh --init
scripts/logs-push.sh -m "init logs branch"
```

Verify: `scripts/pull.sh --list` shows 6 sessions, `du -sh .git` is tens of MB
not hundreds, and the `logs` branch exists on GitHub.

</details>

---

## Phase 1 - ONE TIME per new box

```bash
git clone https://github.com/adhityablubridge/sessions.git ~/.claude-sessions

# per-box paths. Lives OUTSIDE the repo so it can never be committed.
cat > ~/.claude-sync.local.conf <<'CONF'
VAULT_DIR="$HOME/Documents/obsidian Vaults/Adhitya's Vault"   # laptop
# VAULT_DIR="$HOME/claude-vault"                              # gpu box, no vault
CONF
```

If the box already has real sessions in the target project dir, add `--adopt` to
the first `pull.sh` so they are absorbed rather than refused.

---

## Phase 2 - DAILY

### Sitting down

```bash
cd ~/.claude-sessions
scripts/pull.sh --dir /root/BluTrain     # sessions -> this box
scripts/logs-pull.sh                     # log + reports -> vault (merges, never overwrites)

cd /root/BluTrain                        # MUST match --dir above
claude --resume <UUID>                   # or: claude --resume
```

### Before destroying the box - MANDATORY

```bash
# close Claude first: push.sh now REFUSES an open transcript (--force overrides)
cd ~/.claude-sessions
scripts/logs-push.sh -m "session log"
scripts/push.sh --dir /root/BluTrain -m "what you did"
```

Nothing else preserves the work. `push.sh` is required in **both** sync modes -
data does not leave the machine by itself.

**Automate it.** Relying on memory here loses everything the one time you forget,
so wire a `SessionEnd` hook in that box's `~/.claude/settings.json`:

```json
{ "hooks": { "SessionEnd": [ { "matcher": "*", "hooks": [
  { "type": "command",
    "command": "$HOME/.claude-sessions/scripts/autopush.sh --dir <LAUNCH_DIR>",
    "timeout": 300 } ] } ] } }
```

`autopush.sh` is the unattended wrapper: it waits for the transcript handle to
close (SessionEnd fires while Claude is still exiting), skips silently if
another box owns the store, skips if no `GH_TOKEN`/`SESSIONS_SSH_KEY` is
configured, and never fails a shutdown. It logs to
`~/.claude-sessions/.autopush.log` - **check that file** if a session ever
appears to have gone missing. It is a safety net, not a licence to skip the
manual push before teardown.

### Ownership

`push.sh` hard-fails when `.owner` names a different host. That is deliberate:
before this, ownership was advisory (a `warn` in `pull.sh` only) and two boxes
pushing meant the second silently overwrote the first. Confirm the other box is
finished, then `--force` to take the store.

---

## Conflict handling

Sessions are append-only JSONL, so `classify_session` decides every live-vs-store
pair by CONTENT, never by mtime (a fresh clone stamps the whole store with the
checkout time, which silently defeated the old `-nt` test in both directions):

| verdict | meaning | action |
|---|---|---|
| `take` | store is a strict continuation of live | overwrite live |
| `keep` | live is equal to, or ahead of, store | leave live alone |
| `fork` | each side has lines the other lacks | **refuse**, keep both |

On `fork`, `pull.sh` writes the store copy beside the live one as
`<uuid>.jsonl.store` and leaves the live file untouched. Compare and keep one:

```bash
wc -l <uuid>.jsonl <uuid>.jsonl.store
```

`push.sh` and `pull.sh --adopt` apply the same test in the other direction and
refuse to overwrite a store copy that is ahead of, or diverged from, live.
A fork means two boxes wrote the same session concurrently - the owner lock
exists to stop that happening in the first place.

## The one rule that breaks everything

**The launch dir must match between pull and resume.** Sessions are keyed to the
directory you launch Claude from (`/root/BluTrain` -> `-root-BluTrain`). Pull to
one dir and launch from another and `--resume` shows nothing.

- VS Code extension -> the **workspace root**
- terminal -> wherever you `cd`

Pick one and keep it. To switch: `scripts/pull.sh --dir <the-other-dir>`.

---

## Quick reference

| Task | Command |
|---|---|
| what's in the store | `scripts/pull.sh --list` |
| restore sessions | `scripts/pull.sh --dir <launch-dir>` |
| save sessions | `scripts/push.sh --dir <launch-dir> -m "msg"` |
| absorb existing live sessions | add `--adopt` to pull |
| log/reports down | `scripts/logs-pull.sh` |
| log/reports up | `scripts/logs-push.sh -m "msg"` |
| log entry count + recent reports | `scripts/logs-pull.sh --list` |
| preview any command | add `--dry-run` |
| take a store owned by another box | `scripts/push.sh --force` (confirm it is idle first) |
| why did autopush not run | `cat ~/.claude-sessions/.autopush.log` |
| ~~shrink remote history~~ | ~~`--squash`~~ - spent, see Phase 0. Do not use. |

## When something goes wrong

| Symptom | Fix |
|---|---|
| `--resume` lists nothing | pulled to a different dir than you launched from |
| "REAL project dir already exists" | add `--adopt` |
| push rejected, file too large | a session passed 100MB; `--keep 4`, or `rm` that `.gz` |
| "store was last pushed from <other host>" | confirm that box is finished first |
| "VAULT_DIR is not set" | write `~/.claude-sync.local.conf` (Phase 1) |
| "no logs checkout" | `scripts/logs-pull.sh` (or `--init` the first ever time) |
