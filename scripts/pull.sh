#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# pull.sh - materialize the synced Claude session store onto THIS box.
#
# Box-agnostic: the live project dir name is DERIVED from the directory you
# will launch Claude from, so the repo never stores a box-specific name.
#
#   ./scripts/pull.sh                      # key sessions to $PWD
#   ./scripts/pull.sh --dir /root/BluTrain # key sessions to an explicit dir
#   ./scripts/pull.sh --list               # just show what's in the store
#   ./scripts/pull.sh --adopt              # absorb an existing live dir first
#   ./scripts/pull.sh --no-fetch           # skip git pull (offline)
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/lib-sessions.sh"
sess_init "$SCRIPT_DIR"

LAUNCH_DIR="$PWD"; DO_LIST=0; DO_ADOPT=0; DO_FETCH=1; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dir)      LAUNCH_DIR="$2"; shift 2 ;;
    --list)     DO_LIST=1; shift ;;
    --adopt)    DO_ADOPT=1; shift ;;
    --no-fetch) DO_FETCH=0; shift ;;
    --dry-run)  DRY=1; shift ;;
    -h|--help)  sed -n '2,15p' "$0"; exit 0 ;;
    *)          die "unknown flag: $1" ;;
  esac
done
run() { if [ "$DRY" = 1 ]; then printf '  [dry-run] %s\n' "$*"; else eval "$@"; fi; }

# --- 1. refresh the store from the remote -------------------------------------
if [ "$DO_FETCH" = 1 ]; then
  if [ -d "$REPO_DIR/.git" ]; then
    info "fetching latest sessions into $REPO_DIR"
    run "git -C '$REPO_DIR' pull --ff-only" || warn "git pull failed - continuing with local store"
  else
    warn "$REPO_DIR is not a git repo; using it as a plain directory"
  fi
fi

if [ "$DO_LIST" = 1 ]; then
  info "store: $STORE  (mode=$MODE)"; read_owner; echo; list_sessions "$STORE"; exit 0
fi

# --- 2. work out where this box wants the sessions ----------------------------
check_launch_dir "$LAUNCH_DIR"
NAME="$(encode_path "$LAUNCH_DIR")"
TARGET="$LIVE_DIR/projects/$NAME"
info "launch dir : $LAUNCH_DIR"
info "project dir: $TARGET"
info "store      : $STORE (mode=$MODE)"

# Divergence guard: tell the user if another box last held this store.
OWNER_HOST=$(read_owner | sed -n 's/^host=//p')
if [ -n "${OWNER_HOST:-}" ] && [ "$OWNER_HOST" != "$(hostname)" ]; then
  warn "store was last pushed from '$OWNER_HOST' - make sure that box is done writing"
fi

mkdir -p "$LIVE_DIR/projects" "$STORE"

# --- 3. adopt an existing live project dir (first-time migration) -------------
if [ -e "$TARGET" ] && [ ! -L "$TARGET" ]; then
  n=$(find "$TARGET" -maxdepth 1 -name '*.jsonl' 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ] && [ "$DO_ADOPT" = 0 ]; then
    die "a REAL project dir already exists with $n session(s): $TARGET
  Re-run with --adopt to move those sessions into the shared store,
  or move them aside yourself first. Refusing to touch existing sessions."
  fi
  if [ "$DO_ADOPT" = 1 ] && [ "$n" -gt 0 ]; then
    info "adopting $n existing session(s) into the store"
    for f in "$TARGET"/*.jsonl; do
      [ -e "$f" ] || continue
      b=$(basename "$f")
      if [ "$MODE" = compress ]; then run "gzip -9 -c '$f' > '$STORE/$b.gz'"
      else                             run "cp -n '$f' '$STORE/$b'"; fi
    done
    [ "$DRY" = 1 ] || sync_sidecars "$TARGET" "$STORE" "adopting"
  fi
fi

# --- 4. materialize the store into the live project dir -----------------------
if [ "$MODE" = link ]; then
  # Symlink: Claude writes straight into the repo working tree, so there is no
  # copy-back step and no chance of a stale duplicate.
  if [ -L "$TARGET" ]; then run "rm '$TARGET'"
  elif [ -e "$TARGET" ]; then run "mv '$TARGET' '$TARGET.pre-sync.$(date +%s)'"; fi
  run "ln -s '$STORE' '$TARGET'"
  ok "linked $TARGET -> $STORE"
else
  # Compressed store: decompress into the live dir. Never overwrite a live file
  # that is NEWER than the stored copy (that would destroy unpushed work).
  run "mkdir -p '$TARGET'"
  cnt=0
  for f in "$STORE"/*.jsonl.gz; do
    [ -e "$f" ] || continue
    b=$(basename "$f" .gz); dst="$TARGET/$b"
    if [ -f "$dst" ] && [ "$dst" -nt "$f" ]; then
      warn "keeping NEWER live copy, not overwriting: $b (push it before pulling)"
      continue
    fi
    run "gzip -dc '$f' > '$dst'"; cnt=$((cnt+1))
  done
  ok "restored $cnt session(s) into $TARGET"
  [ "$DRY" = 1 ] || sync_sidecars "$STORE" "$TARGET" "restoring"
fi

# --- 5. rules/ (small, always safe to share) ----------------------------------
if [ -d "$REPO_DIR/rules" ]; then
  if [ -L "$LIVE_DIR/rules" ] || [ ! -e "$LIVE_DIR/rules" ]; then
    run "rm -f '$LIVE_DIR/rules'"; run "ln -s '$REPO_DIR/rules' '$LIVE_DIR/rules'"
    ok "linked rules -> $REPO_DIR/rules"
  else
    warn "$LIVE_DIR/rules is a real dir; leaving it alone"
  fi
fi

[ "$DRY" = 1 ] || write_owner "$LAUNCH_DIR"

echo; info "sessions available (resume from: $LAUNCH_DIR)"
list_sessions "$TARGET"
cat <<EOF

Next:
  cd $LAUNCH_DIR
  claude --resume <UUID>          # or: claude --resume   (interactive picker)
When you are done on this box (BEFORE deleting it):
  $SCRIPT_DIR/push.sh -m "work from \$(hostname)"
EOF
