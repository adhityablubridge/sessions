2026-08-20 - 12:xx - Set up and debugged a git repo for syncing ~/.claude session JSONs + rules across machines - Workspace: .claude / .gitignore

# Session Report: .claude sessions git sync

Repo: https://github.com/adhityablubridge/sessions

## Problem chain (in order of discovery)

1. **Nested repo swallowing the folder.** `~/.claude/projects/` was its own git repo
   (branch `master`, `git remote -v` empty). Two consequences:
   - `git add projects/*` from `~/.claude` was a **silent no-op** - git will not absorb a
     nested repo's contents into the outer index.
   - `git push origin main` from inside `projects/` failed with
     `src refspec main does not match any` - branch was `master` and no `origin` existed.

2. **Fix applied.** `mv .git .git-inner-backup` (not `rm -rf`, to keep inner history
   recoverable) + a whitelist `.gitignore` at `~/.claude` tracking only `projects/` and
   `rules/`, ignoring everything else including `.credentials.json`, `cache/`, `ide/`,
   `debug/`, `shell-snapshots/`, `session-env/`.

3. **My .gitignore bug.** The rule `!projects/**` also un-ignored
   `projects/.git-inner-backup/`, so the entire inner object store was committed.
   Result: 843.82 MiB push, and every large blob appears TWICE in GitHub's >50MB
   warnings (once as the live `.jsonl`, once as the backup's loose object).
   Corrected: added explicit `projects/.git-inner-backup/` + `**/.git-inner-backup/`
   excludes. Cleanup pending: `git rm -r --cached projects/.git-inner-backup` then
   `rm -rf` it.

4. **Force-push collision.** The push succeeded (`8b6f977..0755765 main -> main`), but a
   subsequent `git fetch` reported `+ 0755765...57784f0 main -> origin/main (forced update)`.
   The `+`/`...` mean remote `main` no longer contains commit 0755765 - another machine
   overwrote it. This, not any local error, is the cause of the
   `fatal: Need to specify how to reconcile divergent branches` on `git pull`.
   A new `origin/logs` branch also appeared.

## Key finding: path-encoding mismatch

Claude Code names each project dir from the **encoded absolute path** of the workspace.
Locally that is `projects/-home-blu-bridge25-CP`. The remote (pushed from a different
machine) has `projects/CP`.

Implications:
- Sessions sitting under `projects/CP/` are **invisible** to the local Claude Code -
  only the encoded-path folder is scanned by `/resume`.
- Because the two paths are unrelated to git, the merge is **purely additive**: it adds
  `projects/CP/` alongside the existing folder and deletes nothing.
- To make remote sessions resumable locally, the `.jsonl` files must be copied into the
  encoded folder: `cp -n projects/CP/*.jsonl projects/-home-blu-bridge25-CP/`
  (`-n` = no-clobber, so local files are never overwritten).

## Data-loss analysis (question the user asked)

- Files matched by `.gitignore` (`cache/`, `ide/`, `settings.json`, `.credentials.json`):
  never touched by any git operation. Safe.
- `projects/` and `rules/` are now **tracked**, so a merge *can* modify or delete them.
  Mitigated by the path mismatch above, but a full copy backup was prescribed regardless:
  `cp -a ~/.claude/projects ~/claude-projects-backup-2026-08-20`.

## Security notes raised

- `.credentials.json` lives in `~/.claude` and was untracked; explicitly ignored before
  any commit. Verified absent from `git status` output.
- Session `.jsonl` transcripts contain full conversation text, including anything pasted
  into chats. Repo must stay **private**.

## Open decision (not yet resolved)

Another machine pushes a different folder layout to the same branch, so every sync will
keep producing duplicate-content/different-name folders. Two options presented:
- **Branch per machine** (`main`, `plain-machine-02`), merge selectively - keeps layouts
  separate.
- **Normalize on copy** - push raw `.jsonl` under a machine-neutral folder, rename into
  the encoded path on receive. Less long-term work; a sync script was offered.

Also unaddressed: GitHub's 50MB soft limit is already exceeded by 5 live CP transcripts
(51-71 MB each). Git LFS or transcript trimming will be needed as these grow.

## Operational caveat

Session `.jsonl` files are rewritten continuously while Claude Code runs, so committing
from an active machine produces constant dirty-tree noise and merge conflicts. Commit
only when a machine is idle.

## Files changed this session

- `/home/blu-bridge25/.claude/.gitignore` (created, then corrected)
- `/home/blu-bridge25/Documents/obsidian Vaults/Adhitya's Vault/Claude Logs.md` (appended)

No git commands were executed by Claude (rule 1); all git operations were run by the user.

---

## ADDENDUM - remote is a designed sync system, not a dump

Remote `main` (post-force-push) contains `SOP.md`, `.gitattributes`, `sessions.conf`,
`scripts/{pull,push,logs-pull,logs-push,lib-sessions}.sh`, and a canonical store at
`projects/CP/*.jsonl.gz` + `.manifest.tsv` + `.owner`.

`projects/CP` is NOT an encoded path - it is a machine-neutral store. `pull.sh --dir
<launch-dir>` derives the encoded live dir name via `encode_path` and materializes into
it. This is exactly the "normalize on copy" design proposed earlier; it is already built.
Gzip + KEEP pruning also resolves the >50MB warnings, so Git LFS is NOT needed.

### Corrections to earlier advice in this session

1. **Repo must live at `~/.claude-sessions`, NOT `~/.claude`.** Making `~/.claude` the
   repo (done earlier this session) is wrong: `pull.sh` step 5 does
   `ln -s "$REPO_DIR/rules" "$LIVE_DIR/rules"`, which self-links when the two are equal;
   and step 1 runs `git pull --ff-only`, which is exactly what failed. The SOP preamble
   ("repo at ~/.claude") contradicts its own Phase 1 (`clone ~/.claude-sessions`);
   Phase 1 is correct. `~/.claude` is rewritten live by the running app and must not be
   a git worktree.
2. **Do not merge.** The `--squash` in Phase 0 created an orphan commit, so local `main`
   and `origin/main` share zero history. Any merge needs
   `--allow-unrelated-histories` and conflicts on every common path. Re-clone instead.
3. **`cp -n *.jsonl` (given earlier) copies nothing** - remote files are `.jsonl.gz`.
   Superseded by `pull.sh` anyway.

### Landmines identified

- **Phase 0 must NOT be run on this box.** It contains
  `rm -rf projects/-home-blu-bridge25-CP` - "stale" only on the `/root/BluTrain` box that
  authored the SOP; here it is ~500MB of live CP work. Phase 0 has already been executed
  there (it is the source of the force-push and of the loss of commit 0755765). This box
  is a Phase 1 box.
- **All SOP commands hardcode `--dir /root/BluTrain`.** This box needs
  `--dir /home/blu-bridge25/CP`. Per the SOP's own "one rule that breaks everything", a
  mismatch yields a silently empty `--resume`.
- **`pull.sh --adopt` overwrites store entries unconditionally**
  (`gzip -9 -c "$f" > "$STORE/$b.gz"`, no newer/older test - unlike step 4 which does
  guard with `-nt`). Session `d1268623-8176-4a0f-a203-614db704d9b6` exists on BOTH sides
  (local 70.79MB raw, store gzipped). Compare line counts via `zcat | wc -l` before
  adopting; a staler local copy would clobber the store's.
- **`push.sh` prunes the store to `$KEEP` newest** and `rm -rf`s sidecar dirs. ~26 local
  sessions adopted into a KEEP=6 store will evict the other box's older sessions - and
  with `--squash`, from history too. Pin UUIDs in `KEEP_SESSIONS` in `sessions.conf`
  first.
- **`~/.claude/rules` is a real dir**, so pull.sh step 5 warns and skips. Rules stay safe
  but never sync until the local dir is moved aside (diff the two versions first).

### Verified-safe behaviours

- pull.sh step 4 refuses to overwrite a live file newer than the stored copy.
- Neither script ever deletes live session files; push.sh is one-way live -> store.
- `size_guard` pre-checks GitHub's size limit before pushing.
- Both scripts honour `--dry-run` throughout - use it first.

### Still unread at time of writing

`sessions.conf` and `scripts/lib-sessions.sh` - these define `STORE`, `KEEP`, and `MODE`.
`MODE=link` symlinks the live project dir into the git worktree (Claude writes directly
into a checkout), which materially changes the risk profile. Do not run `pull.sh` before
confirming MODE.

### Backups taken

- `~/claude-projects-backup-2026-08-20` (done)
- `~/claude-rules-backup-2026-08-20` (prescribed)
- `~/claude-outer-git-backup` (prescribed, for the `~/.claude/.git` being retired)
