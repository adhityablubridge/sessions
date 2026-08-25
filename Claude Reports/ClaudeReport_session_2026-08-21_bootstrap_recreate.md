# Claude Report - 2026-08-21 - bootstrap.sh recreated + auto-save made portable

2026-08-21 - Recreated the one-command new-box bootstrap and closed the gap that
made session auto-save non-portable. Workspace: CP (/home/blu-bridge25/CP),
files under ~/.claude-sessions/scripts/.

## Why it was missing
bootstrap.sh was written on the H200 at /root/.claude/scripts/bootstrap.sh but
never committed - it was authored after the --squash commit that published the
other scripts, and the follow-up push was never run. It died with the instance.
Same likely applies to Scripts/sync/ckpt-sync.sh in the CP repo (unverified).

## The real gap found while rebuilding
The SessionEnd hook that runs autopush.sh lives in ~/.claude/settings.json.
That file is NOT tracked (it also holds per-machine plugins/theme/model, so
tracking it wholesale would be wrong), and ~/.claude is no longer the repo on
this box. Net effect: on a fresh instance autopush would never fire, so the
"automatic" save silently would not exist. Bootstrap now installs the hook.

## What bootstrap.sh does
1. seeds ~/.claude-sync.local.conf from env when absent (chmod 600, outside repo)
2. preflight_auth - fails in ~1s instead of after minutes of compression
3. pull.sh  (sessions -> this box, keyed to the --dir workspace root)
4. logs-pull.sh (log + reports -> vault)
5. merges the SessionEnd autopush hook into settings.json

## Merge safety (verified in isolation, real config untouched)
theme/model preserved; an unrelated SessionEnd hook survived; re-running yields
exactly ONE autopush entry rather than stacking duplicates; the entry is
repointed at the current workspace root. Backs up settings.json first and exits
harmlessly if the JSON is invalid.

## Layout note
This box uses the separated layout: repo at ~/.claude-sessions, ~/.claude is a
plain dir (not a repo). That is the arrangement originally recommended - fresh
bootstrap is a plain clone with no clone-into-nonempty dance.

## Still to do
Commit and push bootstrap.sh, or it is lost again with this box. Verify whether
CP's Scripts/sync/ckpt-sync.sh survived; recreate if not.
