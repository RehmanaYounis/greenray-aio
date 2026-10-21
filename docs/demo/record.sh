#!/usr/bin/env bash
# Record the README demo in a throwaway profile. Type the lines from session.txt.
set -euo pipefail
cd "$(dirname "$0")/../.."
export GREENRAY_HOME="$(mktemp -d)"
exec asciinema rec --overwrite --cols 100 --rows 28 \
    --title "GreenRay AIO: a GPU fell off the bus" \
    --command "uv run greenray lab tsh-xid79-drained-node" \
    docs/demo/greenray.cast
