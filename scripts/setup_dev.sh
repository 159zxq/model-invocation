#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

create_venv dev-tests
uv_pip_install -r "$ROOT/requirements-dev.txt"

log "Development environment ready: $VENV"
log "Run tests with: $VENV_PY -m pytest -q"
