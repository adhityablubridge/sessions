#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# push.sh - save this box's sessions back into the shared store and push.
#
# Run this BEFORE destroying an instance. It is the only thing standing between
# you and losing the session history.
#
#   ./scripts/push.sh -m "stage 3 hopper fork"
#   ./scripts/push.sh --keep 4          # prune store to 4 newest sessions
#   ./scripts/push.sh --force           # take over a store owned by another box,
#                                       #   or snapshot while claude is running
#   ./scripts/push.sh --squash          # collapse history - DESTRUCTIVE, see SOP
#   ./scripts/push.sh --dry-run
# ---------------------------------------------------------------------------
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$SCRIPT_DIR/lib-sessions.sh"
sess_init "$SCRIPT_DIR"

LAUNCH_DIR="$PWD"; MSG=""; DO_SQUASH=0; DRY=0; FORCE=0
SCRATCH="_sync_tmp"; SQUASH_BRANCH=""
# Everything the squash must carry. A --squash rebuilds the branch from this
# list alone, so anything omitted here is DELETED from the remote.
ADD_PATHS="projects rules scripts sessions.conf .gitignore README.md SOP.md .gitattributes"
while [ $# -gt 0 ]; do
  case "$1" in
    --dir)     LAUNCH_DIR="$2"; shift 2 ;;
    -m|--message) MSG="$2"; shift 2 ;;
    --keep)    KEEP="$2"; shift 2 ;;
    --squash)  DO_SQUASH=1; shift ;;
    --branch)  SQUASH_BRANCH="$2"; shift 2 ;;
    --force)   FORCE=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *)         die "unknown flag: $1" ;;
  esac
done
run() { if [ "$DRY" = 1 ]; then printf '  [dry-run] %s\n' "$*"; else eval "$@"; fi; }

check_launch_dir "$LAUNCH_DIR"
TARGET="$(live_project_dir "$LAUNCH_DIR")"
[ -e "$TARGET" ] || die "no live project dir for $LAUNCH_DIR ($TARGET). Wrong --dir?"
info "live : $TARGET"
info "store: $STORE (mode=$MODE)"

# --- 0a. ownership lock -------------------------------------------------------
# read_owner is advisory in pull.sh (a warn), which is not enough: two boxes
# each pushing a --squash means the second SILENTLY force-overwrites the first,
# and the loser's commit is unrecoverable after the gc the SOP prescribes.
# Pushing is the destructive direction, so the check belongs here and it hard-
# fails. Taking the store over is legitimate - it just has to be deliberate.
OWNER_HOST=$(read_owner | sed -n 's/^host=//p')
if [ -n "${OWNER_HOST:-}" ] && [ "$OWNER_HOST" != "$(hostname)" ]; then
  if [ "$FORCE" = 1 ]; then
    warn "taking over the store from '$OWNER_HOST' (--force)"
  else
    die "the store is currently owned by '$OWNER_HOST', not $(hostname).

  Pushing now would overwrite whatever that box has not yet pulled.
  Confirm '$OWNER_HOST' is finished writing, then re-run with --force:
    $0 --dir '$LAUNCH_DIR' --force -m '<msg>'

  Current owner record:
$(read_owner | sed 's/^/    /')"
  fi
fi

# --- 0b. refuse to snapshot a session that is being written right now ----------
if command -v fuser >/dev/null 2>&1; then
  busy=$(fuser "$TARGET"/*.jsonl 2>/dev/null | tr -s ' ' || true)
  if [ -n "$busy" ]; then
    # A mid-write snapshot silently truncates the tail - i.e. loses exactly the
    # work you are pushing in order to preserve. Only --force may proceed.
    if [ "$FORCE" = 1 ]; then
      warn "a session file is OPEN (claude still running?) - snapshot may be mid-write (--force)"
    else
      die "a session file is OPEN (claude still running?) - pid(s):$busy

  Close Claude first; a mid-write snapshot loses the tail of the session.
  To snapshot anyway: re-run with --force"
    fi
  fi
fi

mkdir -p "$STORE"

[ "$DRY" = 1 ] || preflight_auth "$REPO_DIR"

# --- 1. copy live -> store ----------------------------------------------------
# link mode needs no copy: the live dir IS the store.
if [ "$MODE" = link ]; then
  info "link mode: live dir is the store, nothing to copy"
else
  # Only re-compress sessions whose size changed since the last push. Rewriting
  # an unchanged 70MB blob would add a fresh ~17MB object to git for nothing.
  declare -A prev=()
  if [ -f "$MANIFEST" ]; then
    while IFS=$'\t' read -r u s; do prev["$u"]="$s"; done < "$MANIFEST"
  fi
  [ "$DRY" = 1 ] || : > "$MANIFEST.new"
  changed=0; skipped=0
  for f in "$TARGET"/*.jsonl; do
    [ -e "$f" ] || continue
    b=$(basename "$f"); sz=$(stat -c%s "$f")
    [ "$DRY" = 1 ] || printf '%s\t%s\n' "$b" "$sz" >> "$MANIFEST.new"
    if [ "${prev[$b]:-}" = "$sz" ] && [ -f "$STORE/$b.gz" ]; then
      skipped=$((skipped+1)); continue
    fi
    # Never let a shorter live copy overwrite a stored one that is ahead. Normal
    # pushes have live ahead of store, so this only trips after a --force
    # takeover where this box never pulled the other box's continuation.
    if [ -f "$STORE/$b.gz" ]; then
      case "$(classify_session "$STORE/$b.gz" "$f")" in
        take) warn "store is AHEAD of live, not overwriting: $b (pull first)"
              skipped=$((skipped+1)); continue ;;
        fork) warn "DIVERGED, store kept, not overwriting: $b"
              warn "  compare: zcat '$STORE/$b.gz' | wc -l ; wc -l '$f'"
              skipped=$((skipped+1)); continue ;;
      esac
    fi
    info "compressing $b ($(human "$sz"))"
    run "gzip -9 -c '$f' > '$STORE/$b.gz'"
    changed=$((changed+1))
  done
  run "mv '$MANIFEST.new' '$MANIFEST'"
  ok "$changed session(s) updated, $skipped unchanged"
  [ "$DRY" = 1 ] || sync_sidecars "$TARGET" "$STORE" "saving"
fi

# --- 2. prune the store to the N newest sessions ------------------------------
# KEEP_SESSIONS (from sessions.conf) is an always-keep allowlist of UUIDs.
if [ "${KEEP:-0}" -gt 0 ]; then
  mapfile -t all < <(ls -1t "$STORE"/*.jsonl.gz "$STORE"/*.jsonl 2>/dev/null || true)
  if [ "${#all[@]}" -gt "$KEEP" ]; then
    for f in "${all[@]:$KEEP}"; do
      u=$(basename "$f"); u=${u%.gz}; u=${u%.jsonl}
      case " ${KEEP_SESSIONS:-} " in *" $u "*) info "keeping pinned $u"; continue ;; esac
      info "pruning old session $u"
      # Remove the sidecar dir too - otherwise <uuid>/subagents/ survives every
      # prune and the store slowly refills with transcripts for dead sessions.
      run "rm -f '$f'"
      run "rm -rf '$STORE/$u'"
    done
  fi
fi

# --- 3. hard guard: never try to push something GitHub will reject ------------
[ "$DRY" = 1 ] || size_guard

[ "$DRY" = 1 ] || write_owner "$LAUNCH_DIR"
du -sh "$STORE" 2>/dev/null | awk '{printf "-- store size: %s\n",$1}'

# --- 4. commit + push ---------------------------------------------------------
[ -d "$REPO_DIR/.git" ] || { warn "$REPO_DIR is not a git repo - stopping after local save"; exit 0; }
: "${MSG:=sessions: $(hostname) $(date -u '+%Y-%m-%d %H:%M')}"

if [ "$DO_SQUASH" = 1 ]; then
  # Single-commit history: the remote never accumulates old session blobs.
  #
  # The target branch must NOT be read from HEAD: a previous failed run can
  # leave us stranded on the scratch branch, and we would then rename it onto
  # itself. Resolve it from the remote (or --branch), and make the scratch
  # branch reusable so a retry after a failure just works.
  cur="$(git -C "$REPO_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
  br="${SQUASH_BRANCH:-}"
  if [ -z "$br" ]; then
    br="$(git -C "$REPO_DIR" symbolic-ref --quiet --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##')"
  fi
  [ -n "$br" ] || br=main
  [ "$br" != "$SCRATCH" ] || die "refusing to target the scratch branch; pass --branch <name>"
  info "squashing history onto '$br' (force-push); scratch=$SCRATCH"
  if [ "$cur" = "$SCRATCH" ]; then
    info "already on scratch branch from a previous run - reusing it"
  else
    run "git -C '$REPO_DIR' branch -D '$SCRATCH' 2>/dev/null || true"
    run "git -C '$REPO_DIR' checkout --orphan '$SCRATCH'"
  fi
  run "git -C '$REPO_DIR' add -A -- $ADD_PATHS"
  run "git -C '$REPO_DIR' commit -q -m '$MSG'"
  run "git -C '$REPO_DIR' branch -M '$br'"
  run "git -C '$REPO_DIR' push -f origin '$br'"
else
  run "git -C '$REPO_DIR' add -A -- $ADD_PATHS"
  if [ "$DRY" = 1 ] || ! git -C "$REPO_DIR" diff --cached --quiet; then
    run "git -C '$REPO_DIR' commit -q -m '$MSG'"
    run "git -C '$REPO_DIR' push"
  else
    info "nothing to commit"
  fi
fi
ok "pushed. Safe to destroy this instance."
