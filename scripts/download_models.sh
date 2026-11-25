#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

bash "$ROOT/scripts/download_vision_models.sh"
bash "$ROOT/scripts/download_tts_models.sh"
bash "$ROOT/scripts/download_video_models.sh"
log "All selected model files are present in $MODELS_ROOT"
