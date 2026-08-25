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
    # Older versions of this script wrote .owner on pull, so a stale local edit
    # may be blocking the fast-forward. push.sh is the only legitimate writer,
    # so a local modification here is always a discardable artifact.
    if ! git -C "$REPO_DIR" diff --quiet -- 'projects/*/.owner' 2>/dev/null; then
      info "discarding locally-modified .owner (push.sh owns that file)"
      run "git -C '$REPO_DIR' checkout -- 'projects/*/.owner' 2>/dev/null || true"
    fi
    # Same for the store blobs. adopt/push write .gz straight into the worktree,
    # so an interrupted push leaves it dirty and `pull --ff-only` then refuses
    # FOREVER - the script warns once and carries on with a store that silently
    # never updates again. Discarding is safe: an uncommitted .gz is only ever a
    # recompression of a live transcript that is still on disk, so the next
    # push.sh regenerates it. The manifest is dropped too so that push does not
    # skip it as "unchanged".
    if ! git -C "$REPO_DIR" diff --quiet -- "projects/$CANON" 2>/dev/null; then
      warn "discarding uncommitted store changes (regenerated from live transcripts on next push)"
      git -C "$REPO_DIR" diff --name-only -- "projects/$CANON" 2>/dev/null | sed 's/^/    /' >&2
      run "git -C '$REPO_DIR' checkout -- 'projects/$CANON' 2>/dev/null || true"
      run "rm -f '$MANIFEST'"
    fi
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

# A RUNNING claude owns its transcript: it holds the session in memory and
# rewrites the file from that state, so anything restored underneath it is
# reverted seconds later - and worse, the rewrite can DIVERGE from the stored
# copy rather than just truncate it, turning a clean continuation into a real
# fork. Refuse up front for exactly the sessions that are loaded.
#
# NOTE: an earlier version of this guard used `fuser` on the transcripts. That
# does not work - claude appends in bursts and closes the file in between, so
# the handle check reads empty while the process is very much still in charge.
# loaded_sessions() reads --resume=<uuid> off the process table instead.
# Collect the loaded uuids into a lookup string. This is PER SESSION, not a
# global abort: an earlier version died if anything in the dir was loaded, which
# meant two open VS Code panels blocked the repair of an unrelated third
# session. Only the sessions actually held by a process are skipped.
BUSY_UUIDS=" "
while read -r u pid; do
  [ -n "$u" ] || continue
  BUSY_UUIDS="$BUSY_UUIDS$u "
  [ -f "$TARGET/$u.jsonl" ] && info "in use, will be skipped: ${u%%-*} (pid $pid)"
done <<EOF
$(loaded_sessions)
EOF
session_busy() { case "$BUSY_UUIDS" in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
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
    # Adopt is the mirror of step 4 and needs the same content test. Writing
    # live -> store unconditionally (as this did) clobbers a stored copy that is
    # AHEAD of the live one - e.g. a session continued on another box and pushed
    # from there, which is precisely the case adopt runs into on a second box.
    for f in "$TARGET"/*.jsonl; do
      [ -e "$f" ] || continue
      b=$(basename "$f")
      # A loaded session is mid-conversation: snapshotting it now stores a
      # partial transcript that the next push would then have to reconcile.
      if session_busy "${b%.jsonl}"; then
        info "  in use, not adopting: $b"; continue
      fi
      if [ "$MODE" = compress ]; then
        if [ -f "$STORE/$b.gz" ]; then
          case "$(classify_session "$STORE/$b.gz" "$f")" in
            take) info "  store is ahead, not adopting: $b"; continue ;;
            fork) warn "  DIVERGED, store kept, NOT adopting: $b"
                  warn "    resolve by hand: zcat '$STORE/$b.gz' | wc -l; wc -l '$f'"
                  continue ;;
          esac
        fi
        run "gzip -9 -c '$f' > '$STORE/$b.gz'"
      else
        run "cp -n '$f' '$STORE/$b'"
      fi
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
  # Compressed store: decompress into the live dir. A stored copy may overwrite
  # the live one ONLY when it is a strict continuation of it - decided by
  # content, not mtime (see classify_session in lib-sessions.sh for why mtime
  # cannot work here).
  run "mkdir -p '$TARGET'"
  cnt=0; kept=0; forked=0; busy=0
  for f in "$STORE"/*.jsonl.gz; do
    [ -e "$f" ] || continue
    b=$(basename "$f" .gz); dst="$TARGET/$b"
    # Writing under a running claude is futile - it rewrites the file from
    # memory seconds later, and that rewrite can DIVERGE rather than merely
    # truncate, turning a clean continuation into a real fork.
    if session_busy "${b%.jsonl}"; then
      busy=$((busy+1))
      warn "IN USE, not restored: $b (close the Claude panel, then re-run)"
      continue
    fi
    case "$(classify_session "$f" "$dst")" in
      keep)
        kept=$((kept+1)) ;;                       # live is equal or ahead
      fork)
        # Both sides have lines the other lacks. Either could be the one you
        # want, so refuse to choose: keep live untouched and park the store
        # copy beside it for inspection.
        forked=$((forked+1))
        warn "DIVERGED, live kept: $b"
        warn "  store copy saved as $b.store - compare, then keep one:"
        warn "    wc -l '$dst' '$dst.store'"
        run "gzip -dc '$f' > '$dst.store'" ;;
      take)
        run "gzip -dc '$f' > '$dst'"; cnt=$((cnt+1)) ;;
    esac
  done
  ok "restored $cnt session(s) into $TARGET ($kept already current, $forked diverged, $busy in use)"
  [ "$forked" -eq 0 ] || warn "$forked session(s) DIVERGED - resolve the .store files above"
  [ "$busy" -eq 0 ] || warn "$busy session(s) SKIPPED as in use - close those panels and re-run"
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

# NOTE: deliberately does NOT write .owner. That file is tracked and records
# who last PUSHED; writing it here dirties the working tree and makes the next
# `git pull --ff-only` abort with "local changes would be overwritten".

echo; info "sessions available (resume from: $LAUNCH_DIR)"
list_sessions "$TARGET"
cat <<EOF

Next:
  cd $LAUNCH_DIR
  claude --resume <UUID>          # or: claude --resume   (interactive picker)
When you are done on this box (BEFORE deleting it):
  $SCRIPT_DIR/push.sh -m "work from \$(hostname)"
EOF
