#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

ensure_python
log "Python ready: $PYTHON_BIN ($("$PYTHON_BIN" --version))"
log "uv ready: $UV_BIN ($("$UV_BIN" --version))"
