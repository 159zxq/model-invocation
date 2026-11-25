#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

create_venv download-tools
uv_pip_install "huggingface-hub==0.34.0"
log "Download tools ready: $VENV"
