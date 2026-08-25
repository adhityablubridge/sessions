# ClaudeReport_session_2026-08-22_commit_author_rewrite

2026-08-22 - Re-attributed the last 4 commits that GitHub showed as `root` after being pushed from DataCrunch instances - Workspace: CP / repo history only (no source files changed)

## Problem

The four most recent commits on `main` carried both author and committer as `root`, with instance-local hostnames as the email domain:

| SHA | Author / Committer | Subject |
|---|---|---|
| bfc576b | root@metallic-life-shines-fin-03.datacrunch.io | test: add ckpt_probe.py |
| da56072 | root@metallic-life-shines-fin-03.datacrunch.io | H100 session3 |
| 4031bc5 | root@deep-star-slices-fin-02.datacrunch.io | 2nd H100 session |
| 94008fc | root@plain-machine-roars-fin-02.datacrunch.io | after H100 session1 |

Cause: git identity was never configured on the DataCrunch boxes before the first commit, so git fell back to the container's `root` user and hostname. GitHub cannot match those emails to an account, so the commits render unattributed.

## Constraint

Rule 1 forbids me from running git commands. I supplied the command sequence; the user executed every git operation. All terminal output quoted below is the user's.

## Two mistakes caught in the user's first attempt

1. `git config --local user.email " 239343338+adhityablubridge@users.noreply.github.com"` had a **leading space** inside the quotes - an invalid address.
2. A following `git config user.email "adhitya.charan@blubridge.com"` overwrote it anyway, so the noreply address was not actually in effect.

Recommended the `users.noreply.github.com` form because it always attributes correctly without needing a separate email verification on the GitHub account.

## Sequence executed

```bash
git config --local user.email "239343338+adhityablubridge@users.noreply.github.com"
git config --local user.name  "adhitya-blubridge"
git config --local --get-regexp '^user\.' | cat -A | head   # verify no stray space

git stash push -u -m "pre-author-rewrite"
git branch backup/pre-author-rewrite
git rebase HEAD~4 --exec 'git commit --amend --no-edit --reset-author'
git log -8 --format='%h | A:%an <%ae> | C:%cn <%ce> | %s'
git push --force-with-lease origin main
git stash pop
```

`--reset-author` rewrites **both** author and committer from the current `user.*` config, which is what GitHub needs for re-attribution - amending the author alone would leave the committer line unattributed.

## Result

Rebase succeeded; new SHAs, all four now `adhitya-blubridge <239343338+adhityablubridge@users.noreply.github.com>` for author and committer:

- ccce064 test: add ckpt_probe.py
- c772a74 H100 session3
- 24f3e98 2nd H100 session
- bbde9af after H100 session1

Push accepted: `+ bfc576b...ccce064 main -> main (forced update)`.

Stash popped clean - `.claude/settings.json` and `Tests/bluscriptcp/longrope_autorun.sh` still modified, `pi_s16.txt` / `pi_s32.txt` still untracked, as before the rewrite.

## Left in place deliberately

User scope was "the last few commits", so these older ones were not rewritten and remain unattributed on GitHub:

- 8fe38ef, b2532fb, 6758d09 - name `adhityablubridge`, email `root@neat-book-falls-fin-03.datacrunch.io`
- 2fd146f - `adhitya.blubridge@evoplus.in`

Sweeping them would need `HEAD~8`, which also rewrites the four just-fixed commits.

## Follow-ups for the user

1. Delete the safety branch once GitHub renders correctly: `git branch -D backup/pre-author-rewrite`.
2. Set `user.name` / `user.email` on every new DataCrunch instance before the first commit - this is the actual root cause and will recur otherwise. Worth folding into `.claude-sessions/scripts/bootstrap.sh`, which already handles per-box setup.
3. All four SHAs changed, so any other clone of `main` must `git fetch && git reset --hard origin/main`.
