#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# logs-pull.sh - restore the Claude work log + session reports onto this box.
#
# Uses a SEPARATE single-branch clone of the sessions remote on the 'logs'
# branch, so it never shares a working tree with the session store (switching
# branches in the session repo would blow the sessions away).
#
#   ./scripts/logs-pull.sh              # clone/pull logs branch -> vault
#   ./scripts/logs-pull.sh --init       # create the orphan 'logs' branch (once)
#   ./scripts/logs-pull.sh --list
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/lib-sessions.sh"
sess_init "$SCRIPT_DIR"

DO_INIT=0; DO_LIST=0; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --init)    DO_INIT=1; shift ;;
    --list)    DO_LIST=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *)         die "unknown flag: $1" ;;
  esac
done
run() { if [ "$DRY" = 1 ]; then printf '  [dry-run] %s\n' "$*"; else eval "$@"; fi; }

[ -n "$VAULT_DIR" ] || die "VAULT_DIR is not set.
  It is box-specific, so put it in your local override (never committed):
    echo 'VAULT_DIR=\"\$HOME/vault/Adhitya'\\''s Vault\"' >> $LOCAL_CONF
  On a box with no Obsidian vault, point it anywhere writable, e.g.
    echo 'VAULT_DIR=\"\$HOME/claude-vault\"' >> $LOCAL_CONF"

ORIGIN="$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null || true)"
[ -n "$ORIGIN" ] || die "cannot read origin from $REPO_DIR"

# --- 1. get the logs branch checkout -----------------------------------------
if [ ! -d "$LOGS_REPO_DIR/.git" ]; then
  if [ "$DO_INIT" = 1 ]; then
    # First ever run: build the orphan branch locally, then push it.
    info "creating orphan branch '$LOGS_BRANCH' in $LOGS_REPO_DIR"
    # A fresh `git init` has NO commits, so its first branch is already an
    # orphan. Do NOT clone-then-orphan: the clone drags down main's whole pack
    # and `checkout --orphan` keeps main's index, so the logs branch would
    # inherit every session file.
    run "mkdir -p '$LOGS_REPO_DIR'"
    run "git -C '$LOGS_REPO_DIR' init -q"
    run "git -C '$LOGS_REPO_DIR' symbolic-ref HEAD 'refs/heads/$LOGS_BRANCH'"
    run "git -C '$LOGS_REPO_DIR' remote add origin '$ORIGIN'"
    run "mkdir -p '$LOGS_REPO_DIR/$REPORT_DIR'"
    run "touch '$LOGS_REPO_DIR/$LOG_FILE'"
    # Append-only log: union merge concatenates both sides instead of
    # conflicting when two boxes each add a line.
    run "printf '%s\n' '*.md merge=union' > '$LOGS_REPO_DIR/.gitattributes'"
    ok "orphan branch staged - logs-push.sh will publish it"
  else
    info "cloning '$LOGS_BRANCH' branch into $LOGS_REPO_DIR"
    if ! run "git clone --single-branch --branch '$LOGS_BRANCH' '$ORIGIN' '$LOGS_REPO_DIR'"; then
      die "branch '$LOGS_BRANCH' does not exist on the remote yet.
  Create it once with:  $SCRIPT_DIR/logs-pull.sh --init && $SCRIPT_DIR/logs-push.sh"
    fi
  fi
else
  info "updating $LOGS_REPO_DIR"
  run "git -C '$LOGS_REPO_DIR' pull --ff-only" || warn "pull failed - using local copy"
fi

if [ "$DO_LIST" = 1 ]; then
  info "log entries: $(wc -l < "$LOGS_REPO_DIR/$LOG_FILE" 2>/dev/null || echo 0)"
  info "reports:"; ls -1 "$LOGS_REPO_DIR/$REPORT_DIR" 2>/dev/null | tail -10
  exit 0
fi

# --- 2. materialize into the vault -------------------------------------------
# Copy (not symlink): the vault holds unrelated personal notes and Obsidian is
# happier with real files. These are kilobytes, so copying is free.
run "mkdir -p $(shq "$VAULT_DIR/$REPORT_DIR")"

# Union-merge, never overwrite: the vault keeps its order and gains any lines
# only the repo has. Safe against a big local log meeting a small repo copy.
if [ -f "$LOGS_REPO_DIR/$LOG_FILE" ] || [ -f "$VAULT_DIR/$LOG_FILE" ]; then
  if [ "$DRY" = 1 ]; then
    printf '  [dry-run] merge_log vault + repo -> %s\n' "$VAULT_DIR/$LOG_FILE"
  else
    merge_log "$VAULT_DIR/$LOG_FILE" "$LOGS_REPO_DIR/$LOG_FILE" "$VAULT_DIR/$LOG_FILE"
  fi
  ok "merged $LOG_FILE into the vault"
fi

if [ -d "$LOGS_REPO_DIR/$REPORT_DIR" ]; then
  # -n (no-clobber): reports are per-session unique filenames, so an existing
  # vault report is always authoritative. Never overwrite one.
  run "cp -rn $(shq "$LOGS_REPO_DIR/$REPORT_DIR/.") $(shq "$VAULT_DIR/$REPORT_DIR/") 2>/dev/null || true"
  ok "restored $(find "$LOGS_REPO_DIR/$REPORT_DIR" -name '*.md' | wc -l) report(s)"
fi

cat <<EOF

Vault ready: $VAULT_DIR
  log     : $VAULT_DIR/$LOG_FILE
  reports : $VAULT_DIR/$REPORT_DIR/
After logging work:  $SCRIPT_DIR/logs-push.sh -m "session log"
EOF
