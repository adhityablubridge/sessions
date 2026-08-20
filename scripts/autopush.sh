#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# autopush.sh - non-interactive wrapper around push.sh, for a SessionEnd hook.
#
# Why a wrapper and not push.sh directly:
#   * SessionEnd fires while Claude is still tearing down, so the transcript
#     may still be open. push.sh (correctly) refuses a mid-write snapshot, so
#     we wait for the handle to clear instead of forcing past the guard.
#   * A hook must never block or fail a shutdown: everything here is
#     best-effort, bounded in time, and logged rather than surfaced.
#   * Auto-push must never TAKE the store from another box. If someone else
#     owns it we skip and say so; taking over stays a deliberate --force.
#
#   scripts/autopush.sh --dir /home/blu-bridge25/CP
#
# Log: ~/.claude-sessions/.autopush.log   (read this if a session went missing)
# ---------------------------------------------------------------------------
set -uo pipefail        # NOT -e: a failed push must not abort the shutdown
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/lib-sessions.sh"
sess_init "$SCRIPT_DIR"

LOG="$REPO_DIR/.autopush.log"
WAIT_SECS="${AUTOPUSH_WAIT:-20}"
LAUNCH_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dir)  LAUNCH_DIR="$2"; shift 2 ;;
    --wait) WAIT_SECS="$2"; shift 2 ;;
    *)      shift ;;            # tolerate unknown flags: never fail a hook
  esac
done

say() { printf '%s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$LOG"; }

[ -n "$LAUNCH_DIR" ] || { say "SKIP: no --dir given"; exit 0; }
[ -d "$LAUNCH_DIR" ] || { say "SKIP: launch dir missing: $LAUNCH_DIR"; exit 0; }

TARGET="$(live_project_dir "$LAUNCH_DIR")"
[ -d "$TARGET" ] || { say "SKIP: no live project dir: $TARGET"; exit 0; }

# --- do not take the store from another box ----------------------------------
OWNER_HOST=$(read_owner | sed -n 's/^host=//p')
if [ -n "${OWNER_HOST:-}" ] && [ "$OWNER_HOST" != "$(hostname)" ]; then
  say "SKIP: store owned by '$OWNER_HOST'. Run push.sh --force by hand once that box is done."
  exit 0
fi

# --- auth must already be configured; never prompt ---------------------------
if [ -z "${GH_TOKEN:-}" ] && [ -z "${SESSIONS_SSH_KEY:-}" ]; then
  say "SKIP: no GH_TOKEN or SESSIONS_SSH_KEY in $LOCAL_CONF - cannot push unattended"
  exit 0
fi

# --- wait for the transcript to be closed ------------------------------------
if command -v fuser >/dev/null 2>&1; then
  waited=0
  while [ "$waited" -lt "$WAIT_SECS" ]; do
    busy=$(fuser "$TARGET"/*.jsonl 2>/dev/null | tr -s ' ' || true)
    [ -n "$busy" ] || break
    sleep 1; waited=$((waited+1))
  done
  if [ -n "${busy:-}" ]; then
    say "SKIP: transcript still open after ${WAIT_SECS}s (pids:$busy) - not snapshotting mid-write"
    exit 0
  fi
  [ "$waited" -eq 0 ] || say "waited ${waited}s for the transcript to close"
fi

say "pushing (dir=$LAUNCH_DIR)"
if "$SCRIPT_DIR/push.sh" --dir "$LAUNCH_DIR" \
     -m "auto: $(hostname) $(date -u '+%Y-%m-%d %H:%M')" >>"$LOG" 2>&1; then
  say "OK"
else
  say "FAILED (exit $?) - sessions are NOT backed up; run push.sh by hand"
fi
exit 0
