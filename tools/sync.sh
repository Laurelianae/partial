#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${PARTIAL_REMOTE:-$SPARK_ADDRESS}"
REMOTE_USER="${PARTIAL_USER:-$(id -un)}"
REMOTE_DIR="${PARTIAL_REMOTE_DIR:-~/Development/Projects/partial}"

cd "$ROOT"

rsync \
    --archive \
    --compress \
    --delete \
    --exclude-from=.rsyncignore \
    --itemize-changes \
    ./ \
    "${REMOTE_USER}@${REMOTE}:${REMOTE_DIR}/"