#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# bootstrap.sh - take a bare box to "ready to work" in one command.
#
#   1. clone the work repos     CP -> <work-root>, BluTrain inside it, + submodules
#   2. per-box config           VAULT_DIR / GH_TOKEN, seeded from env if absent
#   3. auth check               fails in ~1s, not after minutes of compression
#   4. restore sessions         pull.sh, keyed to <work-root>
#   5. restore log + reports    logs-pull.sh
#   6. arm auto-save            SessionEnd -> autopush.sh (lives in the UNTRACKED
#                               settings.json, so it does not travel with the repo)
#
# On a fresh box:
#   git clone https://github.com/adhityablubridge/sessions.git ~/.claude-sessions
#   scp <otherbox>:~/.claude-sync.local.conf ~/          # or pass vars inline
#   ~/.claude-sessions/scripts/bootstrap.sh --dir /root/CP
#
# <work-root> need not exist - it is created and cloned into. It becomes the
# VS Code workspace root and is what sessions are keyed to.
#
#   --skip-work   sessions/logs only, do not touch the work repos
#   --work-only   clone the work repos and stop
#   --no-hook     do not install the SessionEnd auto-save hook
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORK_ROOT=""; SKIP_LOGS=0; ADOPT=""; INSTALL_HOOK=1; SKIP_WORK=0; WORK_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dir)        WORK_ROOT="$2"; shift 2 ;;
    --adopt)      ADOPT="--adopt"; shift ;;
    --skip-logs)  SKIP_LOGS=1; shift ;;
    --skip-work)  SKIP_WORK=1; shift ;;
    --work-only)  WORK_ONLY=1; shift ;;
    --no-hook)    INSTALL_HOOK=0; shift ;;
    -h|--help)    sed -n '2,25p' "$0"; exit 0 ;;
    *)            echo "unknown flag: $1" >&2; exit 1 ;;
  esac
done

. "$SCRIPT_DIR/lib-sessions.sh"
CONF="${CLAUDE_SYNC_LOCAL_CONF:-$HOME/.claude-sync.local.conf}"

# --- 1. per-box config (needed before auth) ----------------------------------
if [ ! -f "$CONF" ]; then
  if [ -z "${GH_TOKEN:-}${SESSIONS_SSH_KEY:-}" ]; then
    cat >&2 <<EOF
error: no per-box config at $CONF and no credential in the environment.

  Copy it from a box that has it (then fix VAULT_DIR for this box):
      scp <otherbox>:~/.claude-sync.local.conf ~/
  or pass the secrets inline:
      GH_TOKEN=ghp_xxx VAULT_DIR="\$HOME/claude-vault" $0 --dir <work-root>
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

# --- work root ---------------------------------------------------------------
if [ -z "$WORK_ROOT" ]; then
  for cand in "$HOME/CP" /root/CP "$HOME/BluTrain" /root/BluTrain; do
    [ -d "$cand" ] && { WORK_ROOT="$cand"; break; }
  done
  [ -n "$WORK_ROOT" ] || die "no --dir given and nothing obvious found.
  Pass the workspace root you want, e.g.  $0 --dir /root/CP"
  warn "no --dir given, using $WORK_ROOT"
fi
case "$WORK_ROOT" in /*) ;; *) die "--dir must be an absolute path: $WORK_ROOT" ;; esac
# Guard against the classic mix-up: projects/CP is the SESSION STORE, not code.
case "$WORK_ROOT" in
  "$REPO_DIR"/*) die "--dir points inside the sessions repo ($WORK_ROOT).
  projects/$CANON is the session STORE (compressed chat history), not your code.
  Use a real work root, e.g. --dir /root/CP" ;;
esac

echo
info "bootstrapping $(hostname)"
info "  work root : $WORK_ROOT  -> projects/$(encode_path "$WORK_ROOT")"
info "  sessions  : $STORE (mode=$MODE, keep=$KEEP)"
info "  vault     : ${VAULT_DIR:-<unset, logs skipped>}"
echo

# --- 2. work repos -----------------------------------------------------------
# Pre-seed GitHub's host keys so an SSH clone cannot stall on an interactive
# "authenticity of host can't be established" prompt. Fingerprints are fetched
# from GitHub's own metadata endpoint, not hardcoded, so they stay correct if
# GitHub rotates them.
seed_known_hosts() {
  local kh="$HOME/.ssh/known_hosts"
  grep -q '^github.com ' "$kh" 2>/dev/null && return 0
  mkdir -p "$HOME/.ssh"; chmod 700 "$HOME/.ssh"
  local keys
  keys="$(curl -fsS --max-time 15 https://api.github.com/meta 2>/dev/null \
          | python3 -c 'import sys,json
try:
  for k in json.load(sys.stdin).get("ssh_keys",[]): print("github.com "+k)
except Exception: pass' 2>/dev/null)"
  if [ -n "$keys" ]; then
    printf '%s\n' "$keys" >> "$kh"; chmod 600 "$kh"
    info "seeded github.com host keys into known_hosts"
  else
    warn "could not fetch github.com host keys; an SSH clone may prompt"
  fi
}

clone_or_update() {         # url branch dest label
  local url="$1" br="$2" dest="$3" label="$4"
  if [ -d "$dest/.git" ]; then
    info "$label present, fetching"
    git -C "$dest" pull --ff-only 2>/dev/null || warn "$label: pull failed, leaving as-is"
    return 0
  fi
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    warn "$label: $dest exists and is not empty but is not a git repo - skipping"
    return 1
  fi
  info "cloning $label -> $dest"
  git clone --branch "$br" "$url" "$dest" || { warn "$label: clone FAILED"; return 1; }
}

if [ "$SKIP_WORK" = 0 ]; then
  seed_known_hosts
  clone_or_update "$CP_URL" "$CP_BRANCH" "$WORK_ROOT" "CP" || true

  BT="$WORK_ROOT/${BLUTRAIN_PATH:-BluTrain}"
  if [ -d "$WORK_ROOT" ]; then
    if clone_or_update "$BLUTRAIN_URL" "$BLUTRAIN_BRANCH" "$BT" "BluTrain"; then
      info "initialising BluTrain submodules (Tensor-Implementations, BluBridge-BLAS)"
      git -C "$BT" submodule update --init --recursive \
        || warn "submodule init failed - run it manually in $BT"
    else
      cat >&2 <<EOF
$(warn "BluTrain could not be cloned")
  It is $BLUTRAIN_URL - a DIFFERENT org (BlubridgeAI) over SSH, so the sessions
  PAT does not grant access. Give this box an SSH key with BlubridgeAI access:
      ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
      cat ~/.ssh/id_ed25519.pub      # add at github.com/settings/keys
      ssh -T git@github.com          # expect "successfully authenticated"
  then re-run:  $0 --dir $WORK_ROOT
  Sessions and logs are restored regardless - only the code is missing.
EOF
    fi
  fi
fi
[ "$WORK_ONLY" = 0 ] || { ok "work repos done (--work-only)"; exit 0; }

[ -d "$WORK_ROOT" ] || die "$WORK_ROOT still does not exist - cannot key sessions to it"

# --- 3. auth for the sessions remote ----------------------------------------
preflight_auth "$REPO_DIR"
ok "auth verified"

# --- 4. sessions -------------------------------------------------------------
info "restoring sessions"
"$SCRIPT_DIR/pull.sh" --dir "$WORK_ROOT" $ADOPT

# --- 5. logs -----------------------------------------------------------------
if [ "$SKIP_LOGS" = 0 ] && [ -n "${VAULT_DIR:-}" ]; then
  info "restoring logs"
  "$SCRIPT_DIR/logs-pull.sh" || warn "logs-pull failed (logs branch may not exist yet)"
else
  info "skipping logs (no VAULT_DIR, or --skip-logs)"
fi

# --- 6. arm auto-save --------------------------------------------------------
# settings.json also holds plugins/theme/model, which are legitimately
# per-machine, so it is NOT tracked. Merge only our hook into whatever exists.
if [ "$INSTALL_HOOK" = 1 ]; then
  SETTINGS="$HOME/.claude/settings.json"
  CMD="\$HOME/.claude-sessions/scripts/autopush.sh --dir $WORK_ROOT"
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
ends = cfg.setdefault("hooks", {}).setdefault("SessionEnd", [])
for blk in ends:                      # replace, never stack duplicates
    blk["hooks"] = [h for h in blk.get("hooks", [])
                    if "autopush.sh" not in str(h.get("command", ""))]
ends[:] = [b for b in ends if b.get("hooks")]
ends.append({"matcher": "*",
             "hooks": [{"type": "command", "command": cmd, "timeout": 300}]})
json.dump(cfg, open(p, "w"), indent=2)
print("  SessionEnd -> autopush.sh armed")
PY
  ok "auto-save hook installed"
fi

cat <<EOF

$(ok "bootstrap complete")

Open $WORK_ROOT as the VS Code workspace root, or:
  cd $WORK_ROOT && claude --resume

Auto-save is armed, so normally nothing is needed before destroying this box.
Force a save:  $SCRIPT_DIR/push.sh --dir $WORK_ROOT -m "msg"
Hook log:      $REPO_DIR/.autopush.log
EOF
