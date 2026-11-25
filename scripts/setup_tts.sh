#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

create_venv tts-service
clone_repo "${COSYVOICE_REPO}" "$REPOS_ROOT/CosyVoice" "${COSYVOICE_COMMIT}" \
  --recursive
git -C "$REPOS_ROOT/CosyVoice" submodule update --init --recursive

install_torch \
  "${TORCH_VERSION:-2.5.1}" \
  - \
  "${TORCHAUDIO_VERSION:-2.5.1}"

# Install the filtered inference-only dependency set. The pinned upstream
# requirements file includes training and TensorRT packages that are not used
# by this service.
uv_pip_install -r "$ROOT/services/tts/cosyvoice-requirements.txt"
uv_pip_install -r "$ROOT/services/tts/requirements.txt"
verify_torch \
  "${TORCH_VERSION:-2.5.1}" \
  "${TORCH_CUDA_VERSION:-12.1}"
"$VENV_PY" -m pip check

log "TTS environment ready: $VENV"
