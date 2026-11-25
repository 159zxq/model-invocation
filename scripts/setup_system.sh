#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

PACKAGES=(
  git curl unzip ffmpeg sox libsox-fmt-all
  libgl1 libglib2.0-0 libsndfile1
  build-essential python3-dev
)

if command -v apt-get >/dev/null 2>&1; then
  if [[ "$(id -u)" -eq 0 ]]; then
    APT_PREFIX=()
  elif command -v sudo >/dev/null 2>&1; then
    APT_PREFIX=(sudo)
  else
    warn "No root/sudo. Install manually: ${PACKAGES[*]}"
    exit 0
  fi
  "${APT_PREFIX[@]}" apt-get update
  "${APT_PREFIX[@]}" apt-get install -y "${PACKAGES[@]}"
else
  missing_commands=()
  for command_name in git curl unzip ffmpeg sox; do
    command -v "$command_name" >/dev/null 2>&1 || missing_commands+=("$command_name")
  done
  if ((${#missing_commands[@]} == 0)); then
    log "System commands already installed"
  else
    warn "Unsupported package manager. Install manually: ${PACKAGES[*]}"
  fi
fi
