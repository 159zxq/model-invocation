#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

mkdir -p "$LOG_ROOT" "$PID_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_DEVICE:-0}}"

assert_venv() {
  local name="$1"
  local python="$VENVS_ROOT/$name/bin/python"
  [[ -x "$python" ]] \
    || die "venv missing for $name. Run bash scripts/setup_all.sh on the CPU instance first."
  printf '%s\n' "$python"
}

start_service() {
  local name="$1"
  local port="$2"
  local service_dir="$3"
  local venv_name="$4"
  local python
  local pid_file
  local log_file

  python="$(assert_venv "$venv_name")"
  pid_file="$(service_pid_file "$name")"
  log_file="$(service_log_file "$name")"

  if pid_is_running "$pid_file"; then
    log "$name is already running (pid $(cat "$pid_file"))"
    return
  fi
  rm -f "$pid_file"

  [[ -d "$ROOT/services/$service_dir/app" ]] \
    || die "service directory is missing: $ROOT/services/$service_dir"

  log "Starting $name on port $port"
  (
    cd "$ROOT/services/$service_dir"
    exec nohup env \
      PYTHONPATH="$ROOT/services${PYTHONPATH:+:$PYTHONPATH}" \
      PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
      MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}" \
      CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
      "$python" -m uvicorn app.main:app \
        --host "${HOST:-0.0.0.0}" \
        --port "$port" \
        --workers "${WORKERS:-1}" \
        --no-server-header
  ) >>"$log_file" 2>&1 &

  echo "$!" >"$pid_file"
  if ! wait_for_http "$name" "http://127.0.0.1:$port/health"; then
    warn "$name did not become healthy. Recent log:"
    tail -n 60 "$log_file" >&2 || true
    exit 1
  fi
}

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader || warn "nvidia-smi failed"
else
  warn "nvidia-smi was not found; GPU inference will fail."
fi

start_service vision "${VISION_PORT:-8001}" vision vision-service
start_service tts "${TTS_PORT:-8002}" tts tts-service
start_service video "${VIDEO_PORT:-8003}" video video-service
start_service orchestrator "${ORCHESTRATOR_PORT:-8000}" orchestrator orchestrator

log "All services are healthy"
log "Orchestrator: http://127.0.0.1:${ORCHESTRATOR_PORT:-8000}/docs"
log "Logs: $LOG_ROOT"
log "Stop with: bash scripts/stop_all.sh"
