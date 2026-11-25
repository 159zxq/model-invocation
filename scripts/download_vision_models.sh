#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

require_cmd curl
require_cmd unzip

hf_download \
  "${VISION_BASE_MODEL_ID:-stabilityai/stable-diffusion-xl-base-1.0}" \
  "$MODELS_ROOT/stable-diffusion-xl-base-1.0" \
  "${VISION_BASE_MODEL_REVISION:-main}"

hf_download \
  "${REALVIS_MODEL_ID:-SG161222/RealVisXL_V5.0}" \
  "$MODELS_ROOT/realvisxl-v5.0" \
  "${REALVIS_MODEL_REVISION:-main}" \
  --include "*.json" \
  --include "*.txt" \
  --include "*fp16*.safetensors" \
  --include "*.yaml"

hf_download \
  "${CONTROLNET_MODEL_ID:-xinsir/controlnet-openpose-sdxl-1.0}" \
  "$MODELS_ROOT/controlnet-openpose-sdxl-1.0" \
  "${CONTROLNET_MODEL_REVISION:-main}"

hf_download \
  "${IP_ADAPTER_MODEL_ID:-h94/IP-Adapter-FaceID}" \
  "$MODELS_ROOT/IP-Adapter-FaceID" \
  "${IP_ADAPTER_MODEL_REVISION:-main}"

INSIGHTFACE_ROOT="${VISION_INSIGHTFACE_ROOT:-$MODELS_ROOT/insightface}"
BUFFALO_DIR="$INSIGHTFACE_ROOT/models/buffalo_l"
if [[ ! -f "$BUFFALO_DIR/det_10g.onnx" ]]; then
  mkdir -p "$INSIGHTFACE_ROOT/models"
  tmp_zip="$RUNTIME_ROOT/cache/buffalo_l.zip"
  log "Downloading InsightFace buffalo_l"
  rm -f "$tmp_zip"
  retry 3 curl -L --fail --retry 3 --connect-timeout 20 \
    -o "$tmp_zip" \
    "${INSIGHTFACE_BUFFALO_URL:-https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip}"
  rm -rf "$BUFFALO_DIR.tmp"
  mkdir -p "$BUFFALO_DIR.tmp"
  unzip -q "$tmp_zip" -d "$BUFFALO_DIR.tmp"
  if [[ -d "$BUFFALO_DIR.tmp/buffalo_l" ]]; then
    mv "$BUFFALO_DIR.tmp/buffalo_l" "$BUFFALO_DIR"
    rmdir "$BUFFALO_DIR.tmp"
  else
    mv "$BUFFALO_DIR.tmp" "$BUFFALO_DIR"
  fi
fi

checkpoint="${VISION_BASE_CHECKPOINT:-$MODELS_ROOT/realvisxl-v5.0/RealVisXL_V5.0_fp16.safetensors}"
if [[ -f "$checkpoint" ]]; then
  log "Vision checkpoint ready: $checkpoint"
else
  warn "RealVis checkpoint name not found at $checkpoint; the service will fall back to the SDXL base model."
fi
