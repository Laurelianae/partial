#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

die() {
    printf 'partial: %s\n' "$*" >&2
    exit 1
}

[[ $# -ge 2 ]] || die "Usage: remote.sh NODE {init|sync [--dry-run]|shell|exec COMMAND...}"
node="$1"
action="$2"
shift 2

case "$node" in
    0|1) ;;
    *) die "Node must be 0 or 1." ;;
esac

host_var="PARTIAL_NODE_${node}"
remote="${!host_var:-}"
[[ -n "$remote" ]] || die "Set $host_var in .env.local and invoke this through just."
[[ "$remote" =~ ^[a-zA-Z0-9][a-zA-Z0-9._-]*$ ]] || 
    die "$host_var must be a hostname or SSH alias, without user@ or a port."

remote_user="${PARTIAL_USER:-${USER:-$(id -un)}}"
target="${remote_user}@${remote}"
remote_dir="${PARTIAL_REMOTE_DIR:-Development/Projects/partial}"
remote_dir="${remote_dir%/}"

# Refuse absolute/home/parent paths: this directory is subject to --delete.
case "$remote_dir" in
    ""|/*|\~*) die "PARTIAL_REMOTE_DIR must be a dedicated home-relative directory (no ~/)." ;;
esac
case "/$remote_dir/" in
    *"/../"*|*"/./"*|*"//"*) die "Refusing ambiguous project path: $remote_dir" ;;
esac

# Quote one argument for the remote login shell, including embedded apostrophes.
shell_quote() {
    printf "'%s'" "${1//\'/\'\\\'\'}"
}

# SSH joins command arguments into a shell command. Serialize them explicitly.
remote_exec() {
    local tty_flag="$1"
    shift
    local command="cd -- \"\$HOME\" && cd -- $(shell_quote "$remote_dir") && exec"
    local arg
    for arg in "$@"; do
        command+=" $(shell_quote "$arg")"
    done
    exec ssh "$tty_flag" "$target" "$command"
}

case "$action" in
    init)
        [[ $# -eq 0 ]] || die "init does not accept additional arguments."
        exec ssh -T "$target" \
            "cd -- \"\$HOME\" && mkdir -p -- $(shell_quote "$remote_dir")"
        ;;
    sync)
        if [[ $# -gt 1 ]] || [[ $# -eq 1 && "$1" != --dry-run ]]; then
            die "sync only accepts the optional --dry-run flag."
        fi
        [[ -f "$ROOT/.rsyncignore" ]] || die "Missing $ROOT/.rsyncignore"

        # A real sync may create the destination. A dry run must not do so.
        if [[ $# -eq 0 ]]; then
            ssh -T "$target" \
                "cd -- \"\$HOME\" && mkdir -p -- $(shell_quote "$remote_dir")"
        fi
        printf 'Syncing node %s\n' "$node"
        exec rsync \
            --archive --compress --delete --itemize-changes --protect-args \
            --exclude='/.env.local' \
            --exclude-from="$ROOT/.rsyncignore" \
            "$@" -- "$ROOT/" "$target:$remote_dir/"
        ;;
    shell)
        [[ $# -eq 0 ]] || die "shell does not accept additional arguments."
        remote_exec -t bash -c \
            '[[ -f .venv/bin/activate ]] || { echo "Missing .venv: bootstrap this node first." >&2; exit 1; }; exec bash --rcfile <(printf "%s\n" "[[ -f ~/.bashrc ]] && source ~/.bashrc" "source .venv/bin/activate") -i'
        ;;
    exec)
        [[ $# -gt 0 ]] || die "exec needs a command."
        remote_exec -T "$@"
        ;;
    *)
        die "Unknown action: $action"
        ;;
esac
