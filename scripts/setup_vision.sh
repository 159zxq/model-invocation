#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

create_venv vision-service
install_torch \
  "${TORCH_VERSION:-2.5.1}" \
  "${TORCHVISION_VERSION:-0.20.1}"
uv_pip_install -r "$ROOT/services/vision/requirements.txt"
verify_torch \
  "${TORCH_VERSION:-2.5.1}" \
  "${TORCH_CUDA_VERSION:-12.1}"
"$VENV_PY" -m pip check

log "Vision environment ready: $VENV"
