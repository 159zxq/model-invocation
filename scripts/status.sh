#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

declare -A SERVICES=(
  [orchestrator]="${ORCHESTRATOR_PORT:-8000}"
  [vision]="${VISION_PORT:-8001}"
  [tts]="${TTS_PORT:-8002}"
  [video]="${VIDEO_PORT:-8003}"
)

DOWN=0
for name in orchestrator vision tts video; do
  port="${SERVICES[$name]}"
  pid_file="$(service_pid_file "$name")"
  if pid_is_running "$pid_file" && \
    curl --fail --silent --max-time 3 \
      "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
    printf '%-14s OK   pid=%s port=%s\n' "$name" "$(cat "$pid_file")" "$port"
  else
    printf '%-14s DOWN port=%s\n' "$name" "$port"
    DOWN=1
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  printf '\nGPU:\n'
  nvidia-smi \
    --query-gpu=name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader || true
fi

exit "$DOWN"
