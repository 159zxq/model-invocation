#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

RUN_SYSTEM="${RUN_SYSTEM:-1}"
RUN_SERVICES="${RUN_SERVICES:-1}"
RUN_DOWNLOADS="${RUN_DOWNLOADS:-1}"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_all.sh [options]

Options:
  --skip-system       Skip apt packages
  --skip-services     Skip Python venvs and Git repositories
  --skip-downloads    Skip model downloads
  --downloads-only    Only run model downloads
  -h, --help          Show this help
EOF
}

while (($#)); do
  case "$1" in
    --skip-system) RUN_SYSTEM=0 ;;
    --skip-services) RUN_SERVICES=0 ;;
    --skip-downloads) RUN_DOWNLOADS=0 ;;
    --downloads-only)
      RUN_SYSTEM=0
      RUN_SERVICES=0
      RUN_DOWNLOADS=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
  shift
done

if [[ "$RUN_SYSTEM" == "1" ]]; then
  bash "$ROOT/scripts/setup_system.sh"
fi

bash "$ROOT/scripts/setup_python.sh"

if [[ "$RUN_SERVICES" == "1" ]]; then
  bash "$ROOT/scripts/setup_vision.sh"
  bash "$ROOT/scripts/setup_tts.sh"
  bash "$ROOT/scripts/setup_video.sh"
  bash "$ROOT/scripts/setup_orchestrator.sh"
fi

if [[ "$RUN_DOWNLOADS" == "1" ]]; then
  bash "$ROOT/scripts/download_models.sh"
fi

log "CPU preparation complete"
log "On the GPU instance run: bash scripts/start_all.sh"
