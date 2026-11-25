#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

create_venv orchestrator
uv_pip_install -r "$ROOT/services/orchestrator/requirements.txt"

log "Orchestrator environment ready: $VENV"
