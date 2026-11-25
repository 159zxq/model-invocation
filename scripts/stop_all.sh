#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

terminate_tree() {
  local pid="$1"
  local child
  if command -v pgrep >/dev/null 2>&1; then
    while IFS= read -r child; do
      [[ -n "$child" ]] && terminate_tree "$child"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
  fi
  kill -TERM "$pid" >/dev/null 2>&1 || true
}

stop_service() {
  local name="$1"
  local pid_file
  local pid
  local waited=0
  pid_file="$(service_pid_file "$name")"

  if ! pid_is_running "$pid_file"; then
    log "$name is not running"
    rm -f "$pid_file"
    return
  fi

  pid="$(cat "$pid_file")"
  log "Stopping $name (pid $pid)"
  terminate_tree "$pid"
  while kill -0 "$pid" >/dev/null 2>&1 && ((waited < 30)); do
    sleep 1
    waited=$((waited + 1))
  done
  if kill -0 "$pid" >/dev/null 2>&1; then
    warn "$name did not stop after 30s; sending SIGKILL"
    kill -KILL "$pid" >/dev/null 2>&1 || true
  fi
  rm -f "$pid_file"
}

stop_service orchestrator
stop_service video
stop_service tts
stop_service vision

log "All services stopped"
