#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# lib-sessions.sh - shared helpers for claude session sync (pull/push).
# Sourced by pull.sh / push.sh. Not meant to be run directly.
# ---------------------------------------------------------------------------

# Repo root = parent of the dir holding these scripts. Self-locating, so the
# repo can be cloned ANYWHERE on ANY box and the scripts still find their store.
_self_dir() { cd "$(dirname "${BASH_SOURCE[1]}")" && pwd; }

sess_init() {
  SCRIPT_DIR="$1"
  REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
  LIVE_DIR="${CLAUDE_LIVE_DIR:-$HOME/.claude}"
  CANON="${CLAUDE_PROJECT:-CP}"
  MODE="${CLAUDE_SYNC_MODE:-compress}"     # compress | link
  KEEP="${CLAUDE_KEEP:-6}"                 # prune to N newest sessions
  MAX_FILE_MB=95                           # GitHub hard-rejects >100MB

  # --- logs branch (separate single-branch clone of the same remote) ---------
  LOGS_BRANCH="${LOGS_BRANCH:-logs}"
  LOGS_REPO_DIR="${LOGS_REPO_DIR:-$HOME/.claude-logs}"
  LOG_FILE="${LOG_FILE:-Claude Logs.md}"
  REPORT_DIR="${REPORT_DIR:-Claude Reports}"
  # VAULT_DIR is BOX-SPECIFIC - always set it in the local override, not here.
  VAULT_DIR="${VAULT_DIR:-}"

  # Shared policy, committed and identical on every box.
  [ -f "$REPO_DIR/sessions.conf" ] && . "$REPO_DIR/sessions.conf"
  # Per-box overrides (paths that differ per machine). NEVER committed - it
  # lives outside the repo precisely so it cannot be.
  LOCAL_CONF="${CLAUDE_SYNC_LOCAL_CONF:-$HOME/.claude-sync.local.conf}"
  [ -f "$LOCAL_CONF" ] && . "$LOCAL_CONF"

  STORE="$REPO_DIR/projects/$CANON"
  MANIFEST="$STORE/.manifest.tsv"
}

# Session sidecars: <uuid>/subagents/agent-*.jsonl hold subagent transcripts.
# Small (a few MB) but part of the record, so they are synced verbatim rather
# than compressed per-file.
sync_sidecars() {
  local src="$1" dst="$2" label="$3"
  local n; n=$(find "$src" -mindepth 2 -name '*.jsonl' 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] || return 0
  info "$label $n subagent transcript(s)"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --include='*/' --include='*.jsonl' --exclude='*' \
          --prune-empty-dirs "$src/" "$dst/"
  else
    ( cd "$src" && find . -mindepth 2 -name '*.jsonl' -print0 \
        | tar --null -cf - -T - ) | ( cd "$dst" && tar -xf - )
  fi
}

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
warn() { printf '\033[33mwarn :\033[0m %s\n' "$*" >&2; }
info() { printf '\033[36m--\033[0m %s\n' "$*"; }
ok()   { printf '\033[32mok   :\033[0m %s\n' "$*"; }

# Claude Code names a project dir after the launch directory's absolute path
# with every "/" replaced by "-".  /root/BluTrain -> -root-BluTrain
encode_path() { printf '%s' "$1" | sed 's#/#-#g'; }

# The live project dir for a given launch directory.
live_project_dir() { printf '%s/projects/%s' "$LIVE_DIR" "$(encode_path "$1")"; }

human() { numfmt --to=iec --suffix=B "$1" 2>/dev/null || printf '%sB' "$1"; }

# Verify a launch dir is sane before we key a session store to it.
check_launch_dir() {
  local d="$1"
  [ -n "$d" ] || die "launch dir is empty"
  case "$d" in /*) ;; *) die "launch dir must be absolute: $d" ;; esac
  [ -d "$d" ] || die "launch dir does not exist: $d"
}

# Guard: refuse to stage any file GitHub would reject.
size_guard() {
  local bad=0 f sz
  while IFS= read -r -d '' f; do
    sz=$(stat -c%s "$f")
    if [ "$sz" -gt $((MAX_FILE_MB * 1024 * 1024)) ]; then
      warn "$(basename "$f") is $(human "$sz") - over the ${MAX_FILE_MB}MB guard"
      bad=1
    fi
  done < <(find "$STORE" -type f \( -name '*.jsonl' -o -name '*.jsonl.gz' \) -print0 2>/dev/null)
  [ "$bad" -eq 0 ] || die "size guard tripped. GitHub rejects files >100MB.
  Fix: raise compression (MODE=compress), or drop the session:
    rm '$STORE/<uuid>.jsonl.gz' && $SCRIPT_DIR/push.sh"
}

# One line per session: uuid, mtime, size, first-user-message preview.
list_sessions() {
  local dir="$1" f uuid sz d prev
  [ -d "$dir" ] || { warn "no session store at $dir"; return; }
  printf '%-38s %-16s %8s  %s\n' UUID MODIFIED SIZE PREVIEW
  for f in "$dir"/*.jsonl "$dir"/*.jsonl.gz; do
    [ -e "$f" ] || continue
    uuid=$(basename "$f"); uuid=${uuid%.gz}; uuid=${uuid%.jsonl}
    sz=$(human "$(stat -c%s "$f")")
    d=$(date -r "$f" '+%Y-%m-%d %H:%M')
    if [ "${f##*.}" = gz ]; then prev=$(zcat "$f" 2>/dev/null | head -c 400000 || true)
    else                         prev=$(head -c 400000 "$f" 2>/dev/null || true); fi
    # A preview is cosmetic: never let it abort the listing. Both stages can
    # exit non-zero for benign reasons - grep -m1 takes SIGPIPE (141) when
    # python3 exits after its single readline, and pipefail would otherwise
    # propagate that and kill the whole script on the FIRST session.
    prev=$(printf '%s' "$prev" | grep -m1 '"role":"user"' 2>/dev/null | \
      python3 -c 'import sys,json
try:
  o=json.loads(sys.stdin.readline())
  c=o.get("message",{}).get("content",o.get("content",""))
  if isinstance(c,list): c=" ".join(x.get("text","") for x in c if isinstance(x,dict))
  print(str(c).replace(chr(10)," ")[:60])
except Exception: print("")' 2>/dev/null || true)
    printf '%-38s %-16s %8s  %s\n' "$uuid" "$d" "$sz" "$prev"
  done
}

# --- decide whether a stored session may overwrite the live one ---------------
# Sessions are APPEND-ONLY JSONL, which is what makes this decidable: if one
# copy is a line-exact prefix of the other, the longer one is strictly the more
# complete and can be taken without losing anything.
#
# Why not mtime (the previous `[ "$dst" -nt "$f" ]` test): a fresh `git clone`
# stamps every store file with the CHECKOUT time, so the store always looks
# newer than a live session written days ago and the guard never fires. That is
# the same trap merge_log already documents for the log file - it just was not
# applied to sessions.
#
# Echoes one of: take | keep | fork   (and never exits non-zero)
classify_session() {
  local gz="$1" live="$2"
  [ -f "$live" ] || { echo take; return 0; }          # nothing local to lose
  local nl ns
  # The fallback MUST be outside the substitution. `x=$(cmd || echo 0)` captures
  # BOTH the real output and the 0 when cmd printed and then exited non-zero -
  # which zcat does (warning) on a .gz written from a file being appended to.
  # That yielded "1555\n0" and broke every [ -eq ] below.
  nl=$(wc -l < "$live" 2>/dev/null) || nl=""
  ns=$(zcat "$gz" 2>/dev/null | wc -l) || ns=""
  # Anything not a plain integer means we could not measure a side. Refusing is
  # the only safe verdict - never silently overwrite on a failed measurement.
  case "$nl$ns" in *[!0-9]*|'') echo fork; return 0 ;; esac
  if [ "$ns" -eq "$nl" ]; then
    # Same length: identical content is a no-op, differing content is a fork.
    if [ "$(md5sum < "$live" | cut -d' ' -f1)" = \
         "$(zcat "$gz" 2>/dev/null | md5sum | cut -d' ' -f1)" ]
    then echo keep; else echo fork; fi
    return 0
  fi
  local shorter longer n
  if [ "$ns" -gt "$nl" ]; then shorter=live; n=$nl; else shorter=store; n=$ns; fi
  # Is the shorter one a line-exact prefix of the longer one?
  local h_short h_long
  if [ "$shorter" = live ]; then
    h_short=$(md5sum < "$live" | cut -d' ' -f1)
    h_long=$(zcat "$gz" 2>/dev/null | head -n "$n" | md5sum | cut -d' ' -f1)
  else
    h_short=$(zcat "$gz" 2>/dev/null | md5sum | cut -d' ' -f1)
    h_long=$(head -n "$n" "$live" | md5sum | cut -d' ' -f1)
  fi
  if [ "$h_short" != "$h_long" ]; then echo fork; return 0; fi
  # True continuation: take the store copy only when IT is the longer one.
  if [ "$ns" -gt "$nl" ]; then echo take; else echo keep; fi
}

# --- which sessions are LOADED by a running claude ---------------------------
# fuser is not enough: claude appends in bursts and closes the file between
# them, so a handle check almost always comes back empty even though the
# process holds the whole transcript in memory and will rewrite it from that
# state. The reliable signal is the process's own --resume=<uuid> argument.
#
# Echoes "<uuid> <pid>" per line for every running claude with a session loaded.
loaded_sessions() {
  ps -eo pid,args 2>/dev/null | while read -r pid rest; do
    case "$rest" in *claude*) ;; *) continue ;; esac
    local u
    u=$(printf '%s' "$rest" | grep -o -- '--resume=[0-9a-fA-F-]\{36\}' | cut -d= -f2)
    [ -n "$u" ] && printf '%s %s\n' "$u" "$pid"
  done
}

# Record who last held the store, so two boxes can't silently diverge.
write_owner() {
  printf 'host=%s\nuser=%s\nlaunch_dir=%s\nstamp=%s\n' \
    "$(hostname)" "$(id -un)" "$1" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$STORE/.owner"
}

read_owner() { [ -f "$STORE/.owner" ] && cat "$STORE/.owner" || true; }

# --- append-only log merge ---------------------------------------------------
# Claude Logs.md is append-only and edited from multiple boxes. Copying it in
# either direction can destroy entries (and an mtime check is useless right
# after a clone, which stamps every file with the checkout time). So we UNION:
# keep the destination's existing order, then append any lines only the source
# has. Nothing is ever dropped, nothing is reordered.
merge_log() {
  local a="$1" b="$2" out="$3"     # a = keeps its order, b = contributes new lines
  local tmp; tmp="$(mktemp)"
  { [ -f "$a" ] && cat "$a"; [ -f "$b" ] && cat "$b"; } 2>/dev/null \
    | awk '!seen[$0]++' > "$tmp"
  local na nb no
  na=$([ -f "$a" ] && wc -l < "$a" || echo 0)
  nb=$([ -f "$b" ] && wc -l < "$b" || echo 0)
  no=$(wc -l < "$tmp")
  mv "$tmp" "$out"
  info "log merged: $na + $nb -> $no unique line(s)"
}

# --- non-interactive auth -----------------------------------------------------
# GH_TOKEN comes from $LOCAL_CONF (~/.claude-sync.local.conf): outside the repo,
# gitignored, chmod 600. It is NEVER written into git config or a tracked file -
# the helper below reads it from the environment each time git asks. That is the
# difference between "the script uses my PAT" (fine) and "my PAT is in a file I
# push to GitHub" (leaked and auto-revoked by secret scanning).
setup_auth() {
  local repo="$1"
  [ -d "$repo/.git" ] || return 0
  if [ -n "${GH_TOKEN:-}" ]; then
    export GH_TOKEN
    git -C "$repo" config credential.helper \
      '!f() { test "$1" = get && printf "username=x-access-token\npassword=%s\n" "$GH_TOKEN"; }; f'
    return 0
  fi
  if [ -n "${SESSIONS_SSH_KEY:-}" ] && [ -f "$SESSIONS_SSH_KEY" ]; then
    chmod 600 "$SESSIONS_SSH_KEY" 2>/dev/null || true
    git -C "$repo" config core.sshCommand \
      "ssh -i $SESSIONS_SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new"
    return 0
  fi
  return 0
}

# Fail on auth BEFORE minutes of compression, with actionable guidance.
preflight_auth() {
  local repo="$1"
  [ -d "$repo/.git" ] || return 0
  setup_auth "$repo"
  git -C "$repo" ls-remote --heads origin >/dev/null 2>&1 && return 0
  die "cannot authenticate to the remote.

  Put ONE of these in $LOCAL_CONF (chmod 600, never committed), then retry:

    GH_TOKEN=\"github_pat_...\"                       # fine-grained PAT,
                                                      # Contents: read+write
    SESSIONS_SSH_KEY=\"\$HOME/.ssh/sessions_deploy_key\"  # or a deploy key

  Do NOT put the token in scripts/push.sh - that file is committed, so the
  token would be published and auto-revoked by GitHub secret scanning."
}
