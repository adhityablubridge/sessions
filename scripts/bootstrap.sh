#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# bootstrap.sh - stand up a brand-new box in one command.
#
# Covers everything between "fresh machine" and "history is back and auto-save
# is armed":
#   1. per-box config (VAULT_DIR / GH_TOKEN) - seeded from env if absent
#   2. auth check, before any expensive work
#   3. restore sessions          (pull.sh)
#   4. restore log + reports     (logs-pull.sh)
#   5. install the SessionEnd autopush hook   <- the piece that does NOT
#      travel with the repo, because it lives in ~/.claude/settings.json
#      alongside machine-specific plugin/theme/model settings.
#
# On a fresh box:
#   scp ~/.claude-sync.local.conf newbox:~/
#   git clone https://github.com/adhityablubridge/sessions.git ~/.claude-sessions
#   ~/.claude-sessions/scripts/bootstrap.sh --dir /home/blu-bridge25/CP
#
# Or seed the secrets inline instead of copying the file:
#   GH_TOKEN=ghp_xxx VAULT_DIR="$HOME/vault" \
#     ~/.claude-sessions/scripts/bootstrap.sh --dir /home/blu-bridge25/CP
#
# --dir MUST be the VS Code workspace root you will open. Sessions are keyed to
# it, so a mismatch is the one way to end up with no history showing.
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LAUNCH_DIR=""; SKIP_LOGS=0; ADOPT=""; INSTALL_HOOK=1
while [ $# -gt 0 ]; do
  case "$1" in
    --dir)       LAUNCH_DIR="$2"; shift 2 ;;
    --adopt)     ADOPT="--adopt"; shift ;;
    --skip-logs) SKIP_LOGS=1; shift ;;
    --no-hook)   INSTALL_HOOK=0; shift ;;
    -h|--help)   sed -n '2,26p' "$0"; exit 0 ;;
    *)           echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

. "$SCRIPT_DIR/lib-sessions.sh"
CONF="${CLAUDE_SYNC_LOCAL_CONF:-$HOME/.claude-sync.local.conf}"

# --- 1. per-box config -------------------------------------------------------
if [ ! -f "$CONF" ]; then
  if [ -z "${GH_TOKEN:-}${SESSIONS_SSH_KEY:-}" ]; then
    cat >&2 <<EOF
error: no per-box config at $CONF and no credential in the environment.

  Copy it from a box that has it:
      scp ~/.claude-sync.local.conf $(hostname):~/
  or pass the secrets inline:
      GH_TOKEN=ghp_xxx VAULT_DIR="\$HOME/vault" $0 --dir <workspace-root>
EOF
    exit 1
  fi
  info "writing $CONF from the environment"
  {
    echo "# Per-box overrides. Never committed - outside the repo by design."
    [ -n "${VAULT_DIR:-}" ]        && echo "VAULT_DIR=\"$VAULT_DIR\""
    [ -n "${GH_TOKEN:-}" ]         && echo "GH_TOKEN=\"$GH_TOKEN\""
    [ -n "${SESSIONS_SSH_KEY:-}" ] && echo "SESSIONS_SSH_KEY=\"$SESSIONS_SSH_KEY\""
  } > "$CONF"
fi
chmod 600 "$CONF" 2>/dev/null || true

sess_init "$SCRIPT_DIR"

# --- 2. workspace root -------------------------------------------------------
if [ -z "$LAUNCH_DIR" ]; then
  for cand in "$HOME/CP" /root/BluTrain "$HOME/BluTrain" "$PWD"; do
    [ -d "$cand" ] && { LAUNCH_DIR="$cand"; break; }
  done
  warn "no --dir given, assuming the workspace root is $LAUNCH_DIR"
fi
check_launch_dir "$LAUNCH_DIR"

echo
info "bootstrapping $(hostname)"
info "  repo      : $REPO_DIR"
info "  workspace : $LAUNCH_DIR  -> projects/$(encode_path "$LAUNCH_DIR")"
info "  store     : $STORE (mode=$MODE, keep=$KEEP)"
info "  vault     : ${VAULT_DIR:-<unset, logs skipped>}"
echo

# --- 3. auth, before any expensive work --------------------------------------
preflight_auth "$REPO_DIR"
ok "auth verified"

# --- 4. sessions -------------------------------------------------------------
info "restoring sessions"
"$SCRIPT_DIR/pull.sh" --dir "$LAUNCH_DIR" $ADOPT

# --- 5. logs -----------------------------------------------------------------
if [ "$SKIP_LOGS" = 0 ] && [ -n "${VAULT_DIR:-}" ]; then
  info "restoring logs"
  "$SCRIPT_DIR/logs-pull.sh" || warn "logs-pull failed (logs branch may not exist yet)"
else
  info "skipping logs (no VAULT_DIR, or --skip-logs)"
fi

# --- 6. arm auto-save --------------------------------------------------------
# settings.json also holds plugins/theme/model, which are legitimately
# per-machine, so the file is NOT tracked. Merge only our hook into whatever is
# already there - idempotently, and with a backup.
if [ "$INSTALL_HOOK" = 1 ]; then
  SETTINGS="$HOME/.claude/settings.json"
  CMD="\$HOME/.claude-sessions/scripts/autopush.sh --dir $LAUNCH_DIR"
  mkdir -p "$(dirname "$SETTINGS")"
  [ -f "$SETTINGS" ] || echo '{}' > "$SETTINGS"
  cp "$SETTINGS" "$SETTINGS.bak.$(date +%s)"
  SETTINGS="$SETTINGS" CMD="$CMD" python3 - <<'PY'
import json, os, sys
p, cmd = os.environ["SETTINGS"], os.environ["CMD"]
try:
    cfg = json.load(open(p))
except Exception as e:
    print(f"  settings.json is not valid JSON ({e}); leaving it alone", file=sys.stderr)
    sys.exit(0)
hooks = cfg.setdefault("hooks", {})
ends  = hooks.setdefault("SessionEnd", [])
# Replace any existing autopush entry rather than stacking duplicates.
for blk in ends:
    inner = blk.get("hooks", [])
    blk["hooks"] = [h for h in inner if "autopush.sh" not in str(h.get("command", ""))]
ends[:] = [b for b in ends if b.get("hooks")]
ends.append({"matcher": "*",
             "hooks": [{"type": "command", "command": cmd, "timeout": 300}]})
json.dump(cfg, open(p, "w"), indent=2)
print("  SessionEnd -> autopush.sh armed")
PY
  ok "auto-save hook installed (sessions push themselves when a session ends)"
fi

cat <<EOF

$(ok "bootstrap complete")

Open $LAUNCH_DIR as the VS Code workspace root - the restored sessions appear in
the Claude panel's history. The workspace root must match the path above.

Auto-save is armed, so normally you need do nothing before destroying this box.
To force a save now, or if the hook log shows a skip:
  $SCRIPT_DIR/push.sh --dir $LAUNCH_DIR -m "what you did"
  $SCRIPT_DIR/logs-push.sh -m "session log"
Hook log: $REPO_DIR/.autopush.log
EOF
