#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# logs-push.sh - save the vault's Claude log + reports back to the logs branch.
#
#   ./scripts/logs-push.sh -m "stage 2 rope kernel"
#   ./scripts/logs-push.sh --dry-run
#
# Safe to run often - the payload is kilobytes of markdown, and the log file
# uses a union merge driver so two boxes appending lines never hard-conflict.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/lib-sessions.sh"
sess_init "$SCRIPT_DIR"

MSG=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    -m|--message) MSG="$2"; shift 2 ;;
    --dry-run)    DRY=1; shift ;;
    -h|--help)    sed -n '2,10p' "$0"; exit 0 ;;
    *)            die "unknown flag: $1" ;;
  esac
done
run() { if [ "$DRY" = 1 ]; then printf '  [dry-run] %s\n' "$*"; else eval "$@"; fi; }

[ -n "$VAULT_DIR" ]           || die "VAULT_DIR not set - see logs-pull.sh --help"
[ -d "$VAULT_DIR" ]           || die "vault dir does not exist: $VAULT_DIR"
[ -d "$LOGS_REPO_DIR/.git" ]  || die "no logs checkout at $LOGS_REPO_DIR
  Run first:  $SCRIPT_DIR/logs-pull.sh   (or --init to create the branch)"

info "vault: $VAULT_DIR"

[ "$DRY" = 1 ] || preflight_auth "$LOGS_REPO_DIR"
info "repo : $LOGS_REPO_DIR (branch $LOGS_BRANCH)"

# --- 1. vault -> repo ---------------------------------------------------------
run "mkdir -p $(shq "$LOGS_REPO_DIR/$REPORT_DIR")"

# Union-merge upward too, so a line another box pushed while you were working
# is not erased by your vault copy.
if [ -f "$VAULT_DIR/$LOG_FILE" ]; then
  if [ "$DRY" = 1 ]; then
    printf '  [dry-run] merge_log repo + vault -> %s\n' "$LOGS_REPO_DIR/$LOG_FILE"
  else
    merge_log "$LOGS_REPO_DIR/$LOG_FILE" "$VAULT_DIR/$LOG_FILE" "$LOGS_REPO_DIR/$LOG_FILE"
    cp "$LOGS_REPO_DIR/$LOG_FILE" "$VAULT_DIR/$LOG_FILE"
  fi
  ok "$LOG_FILE synced"
else
  warn "no $LOG_FILE in the vault yet"
fi

if [ -d "$VAULT_DIR/$REPORT_DIR" ]; then
  cnt=$(find "$VAULT_DIR/$REPORT_DIR" -name '*.md' 2>/dev/null | wc -l)
  run "cp -rn $(shq "$VAULT_DIR/$REPORT_DIR/.") $(shq "$LOGS_REPO_DIR/$REPORT_DIR/") 2>/dev/null || true"
  ok "$cnt report(s)"
fi

# Keep the union merge driver present even if the branch was made by hand.
[ -f "$LOGS_REPO_DIR/.gitattributes" ] || \
  run "printf '%s\n' '*.md merge=union' > '$LOGS_REPO_DIR/.gitattributes'"

# --- 2. commit + push ---------------------------------------------------------
: "${MSG:=logs: $(hostname) $(date -u '+%Y-%m-%d %H:%M')}"
run "git -C '$LOGS_REPO_DIR' add -A"
if [ "$DRY" = 1 ] || ! git -C "$LOGS_REPO_DIR" diff --cached --quiet; then
  run "git -C $(shq "$LOGS_REPO_DIR") commit -q -m $(shq "$MSG")"
  # First push of a freshly created orphan branch needs -u.
  if git -C "$LOGS_REPO_DIR" rev-parse --abbrev-ref '@{upstream}' >/dev/null 2>&1; then
    # Merge before pushing. Another box pushing logs in the meantime makes the
    # branch diverge, and a bare `git push` is then rejected non-fast-forward.
    # `*.md merge=union` resolves the log and reports automatically, so this is
    # normally silent - and it is the same reconciliation logs-pull.sh does.
    run "git -C '$LOGS_REPO_DIR' -c user.name=claude-sync -c user.email=sync@localhost \
         pull --no-rebase --no-edit" \
      || warn "merge with remote logs failed - push will likely be rejected"
    run "git -C '$LOGS_REPO_DIR' push"
  else
    run "git -C '$LOGS_REPO_DIR' push -u origin '$LOGS_BRANCH'"
  fi
  ok "logs pushed to branch '$LOGS_BRANCH'"
else
  info "nothing to commit"
fi
