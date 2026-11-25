#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

hf_download \
  "${ECHOMIMIC_MODEL_ID:-BadToBest/EchoMimicV3}" \
  "$MODELS_ROOT/EchoMimicV3" \
  "${ECHOMIMIC_MODEL_REVISION:-main}" \
  --include "${ECHOMIMIC_MODEL_INCLUDE:-echomimicv3-flash-pro/**}"

hf_download \
  "${ECHOMIMIC_BASE_MODEL_ID:-alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP}" \
  "$MODELS_ROOT/Wan2.1-Fun-V1.1-1.3B-InP" \
  "${ECHOMIMIC_BASE_MODEL_REVISION:-main}"

hf_download \
  "${ECHOMIMIC_WAV2VEC_MODEL_ID:-TencentGameMate/chinese-wav2vec2-base}" \
  "$MODELS_ROOT/chinese-wav2vec2-base" \
  "${ECHOMIMIC_WAV2VEC_REVISION:-main}"

log "EchoMimicV3 weights ready under $MODELS_ROOT/EchoMimicV3"
