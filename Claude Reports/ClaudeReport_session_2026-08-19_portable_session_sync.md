# Claude Report - 2026-08-19 - Portable Claude session + log sync

2026-08-19 - Built a replicable cross-instance sync for Claude Code sessions and work
logs, so an ephemeral GPU box can be destroyed without losing chat history.
Workspace: CP (/root/BluTrain/dist/Context_Parallelism), files under /root/.claude/.

## Problem
Sessions live at `~/.claude/projects/<launch-dir-with-slashes-as-dashes>/<uuid>.jsonl`.
Committing that encoded name breaks on the next box, so `--resume` cannot find it.
Two further walls were found while inspecting the existing setup:
- GitHub hard-rejects files >100MB; sessions were already 74/71/57MB and grow.
- 51 sessions / 537MB of logs had produced an 845MB .git plus a 604MB
  `projects/.git-inner-backup` (a git repo tracked inside the git repo).

## Solution
- Repo stores ONE canonical `projects/CP/`; `pull.sh` DERIVES the encoded dir name
  from the launch dir at restore time. Nothing box-specific is ever committed.
- Store holds `*.jsonl.gz` (measured 4.3x), plus a 95MB guard that fails before
  GitHub does. `KEEP=6` prune + `KEEP_SESSIONS` pin bound the working set.
- Subagent sidecars (`<uuid>/subagents/agent-*.jsonl`, 4.9MB) are synced verbatim.
- `.manifest.tsv` skips re-compressing unchanged sessions to limit git churn.
- Per-box paths go in `~/.claude-sync.local.conf` (outside the repo, uncommittable);
  shared policy in `sessions.conf`.
- Work log + reports on a SEPARATE `logs` branch, cloned single-branch into its own
  dir so it never shares a working tree with the session store. `*.md merge=union`
  so two boxes appending log lines never hard-conflict.

## Result
1986MB (projects + .git) -> ~47MB for the 6 newest sessions, ~42x smaller, every
file well inside GitHub limits. Fresh box bootstrap is clone + pull.sh + resume.

## Verified
Syntax-checked all 5 scripts; path encoding correct for /root/BluTrain and the
nested CP dir; config + per-box override resolution; `--list`; refuse-to-clobber
guard (correctly blocked 26 live sessions until `--adopt`); adopt plan; push
dry-run; VAULT_DIR and missing-checkout guards. Git-executing paths were NOT run
here (Rule 1) and need the user to run them.

## Files
/root/.claude/scripts/{lib-sessions,pull,push,logs-pull,logs-push}.sh,
/root/.claude/sessions.conf, /root/.claude/.gitignore, /root/.claude/README.md,
/root/.claude-sync.local.conf (per-box, untracked)
