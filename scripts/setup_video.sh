#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

create_venv video-service
clone_repo "${ECHOMIMIC_REPO_URL}" "$REPOS_ROOT/echomimic_v3" "${ECHOMIMIC_COMMIT}"

install_torch \
  "${TORCH_VERSION:-2.5.1}" \
  "${TORCHVISION_VERSION:-0.20.1}"

# Do not install the upstream full requirements file: it pulls TensorFlow,
# Gradio, retina-face, training tools, and an unmanaged Torch version.
uv_pip_install -r "$ROOT/services/video/requirements.txt"
verify_torch \
  "${TORCH_VERSION:-2.5.1}" \
  "${TORCH_CUDA_VERSION:-12.1}"
"$VENV_PY" -m pip check

log "Video environment ready: $VENV"
