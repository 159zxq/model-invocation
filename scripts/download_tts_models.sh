#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

hf_download \
  "${COSYVOICE_MODEL_ID:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}" \
  "$MODELS_ROOT/Fun-CosyVoice3-0.5B-2512" \
  "${COSYVOICE_MODEL_REVISION:-main}"

log "CosyVoice model ready: $MODELS_ROOT/Fun-CosyVoice3-0.5B-2512"
