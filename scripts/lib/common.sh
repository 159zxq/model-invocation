#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PROJECT_ROOT="$ROOT"

if [[ -f "$ROOT/configs/services.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/configs/services.env"
  set +a
fi
if [[ -f "$ROOT/configs/download.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/configs/download.env"
  set +a
fi
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

export PROJECT_ROOT="$ROOT"
export RUNTIME_ROOT="${RUNTIME_ROOT:-$ROOT/runtime}"
export MODELS_ROOT="${MODELS_ROOT:-$RUNTIME_ROOT/models}"
export REPOS_ROOT="${REPOS_ROOT:-$RUNTIME_ROOT/repos}"
export VENVS_ROOT="${VENVS_ROOT:-$RUNTIME_ROOT/venvs}"
export LOG_ROOT="${LOG_ROOT:-$RUNTIME_ROOT/logs}"
export PID_ROOT="${PID_ROOT:-$RUNTIME_ROOT/pids}"
export SHARED_JOBS_ROOT="${SHARED_JOBS_ROOT:-$RUNTIME_ROOT/jobs}"
export GPU_LOCK_FILE="${GPU_LOCK_FILE:-$RUNTIME_ROOT/gpu.lock}"
export JOB_LOCK_FILE="${JOB_LOCK_FILE:-$RUNTIME_ROOT/job.lock}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RUNTIME_ROOT/cache/uv}"
export HF_HOME="${HF_HOME:-$RUNTIME_ROOT/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$RUNTIME_ROOT/cache/modelscope}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$RUNTIME_ROOT/tools/python}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p \
  "$MODELS_ROOT" "$REPOS_ROOT" "$VENVS_ROOT" "$LOG_ROOT" \
  "$PID_ROOT" "$SHARED_JOBS_ROOT" "$HF_HOME" "$UV_CACHE_DIR" \
  "$UV_PYTHON_INSTALL_DIR"

log() {
  printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

warn() {
  printf '[%s] WARN: %s\n' "$(date '+%H:%M:%S')" "$*" >&2
}

die() {
  printf '[%s] ERROR: %s\n' "$(date '+%H:%M:%S')" "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

retry() {
  local attempts="$1"
  shift
  local delay="${RETRY_DELAY_SECONDS:-5}"
  local n=1
  until "$@"; do
    if (( n >= attempts )); then
      return 1
    fi
    warn "command failed (attempt $n/$attempts), retrying in ${delay}s: $*"
    sleep "$delay"
    n=$((n + 1))
  done
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    export UV_BIN
    return
  fi

  local tool_dir="$RUNTIME_ROOT/tools"
  mkdir -p "$tool_dir"
  if [[ ! -x "$tool_dir/uv" ]]; then
    require_cmd curl
    log "Installing uv into $tool_dir"
    UV_INSTALL_DIR="$tool_dir" retry 3 bash -c \
      'curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$UV_INSTALL_DIR" sh' \
      || die "failed to install uv"
  fi
  UV_BIN="$tool_dir/uv"
  export UV_BIN
}

ensure_python() {
  ensure_uv
  local version="${PYTHON_VERSION:-3.10.12}"

  if [[ -x "$RUNTIME_ROOT/tools/python/$version/bin/python3.10" ]]; then
    PYTHON_BIN="$RUNTIME_ROOT/tools/python/$version/bin/python3.10"
  elif command -v "python${version%.*}" >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v "python${version%.*}")"
  else
    log "Installing Python $version with uv"
    "$UV_BIN" python install "$version"
    PYTHON_BIN="$("$UV_BIN" python find "$version")"
  fi

  local actual
  actual="$("$PYTHON_BIN" -c 'import platform; print(platform.python_version())')"
  [[ "$actual" == 3.10.* ]] || die "Python 3.10 is required, found $actual"
  export PYTHON_BIN
}

create_venv() {
  local name="$1"
  local venv="$VENVS_ROOT/$name"
  ensure_python
  if [[ ! -f "$venv/pyvenv.cfg" ]]; then
    log "Creating venv: $name"
    "$UV_BIN" venv --seed --python "$PYTHON_VERSION" "$venv"
  fi
  export VENV="$venv"
  export VENV_PY="$venv/bin/python"
}

uv_pip_install() {
  "$UV_BIN" pip install --python "$VENV_PY" "$@"
}

install_torch() {
  local version="$1"
  local torchvision="$2"
  local torchaudio="${3:-}"
  local index="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
  local packages=("torch==$version")
  if [[ -n "$torchvision" && "$torchvision" != "-" ]]; then
    packages+=("torchvision==$torchvision")
  fi
  if [[ -n "$torchaudio" ]]; then
    packages+=("torchaudio==$torchaudio")
  fi
  log "Installing ${packages[*]} from $index"
  uv_pip_install --index-url "$index" "${packages[@]}"
}

verify_torch() {
  local expected_version="$1"
  local expected_cuda="${2:-12.1}"
  local index="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
  log "Verifying Torch $expected_version (CUDA $expected_cuda build)"
  "$VENV_PY" - "$expected_version" "$expected_cuda" "$index" <<'PY'
import sys

import torch

expected_version, expected_cuda, index = sys.argv[1:4]
actual_version = torch.__version__.split("+", 1)[0]
actual_cuda = torch.version.cuda
if actual_version != expected_version:
    raise SystemExit(
        f"expected torch {expected_version}, found {torch.__version__}; "
        "a service dependency replaced the pinned CUDA wheel"
    )
if actual_cuda != expected_cuda:
    raise SystemExit(
        f"expected a CUDA {expected_cuda} torch build from {index}, "
        f"found {actual_cuda}"
    )
print(
    f"torch={torch.__version__} cuda_build={actual_cuda} "
    f"cuda_available={torch.cuda.is_available()}"
)
PY
}

clone_repo() {
  local url="$1"
  local destination="$2"
  local commit="$3"
  shift 3

  if [[ -d "$destination/.git" ]]; then
    log "Repository already exists: $destination"
  else
    [[ ! -e "$destination" || -z "$(ls -A "$destination" 2>/dev/null)" ]] \
      || die "destination exists and is not a Git repository: $destination"
    mkdir -p "$(dirname "$destination")"
    log "Cloning $url"
    retry 3 git clone "$@" "$url" "$destination"
  fi

  git -C "$destination" fetch --tags --prune
  if ! git -C "$destination" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$destination" fetch origin "$commit"
  fi
  git -C "$destination" checkout --detach "$commit"
  log "Pinned $(basename "$destination") to $commit"
}

service_pid_file() {
  printf '%s/%s.pid' "$PID_ROOT" "$1"
}

service_log_file() {
  printf '%s/%s.log' "$LOG_ROOT" "$1"
}

pid_is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid="$(cat "$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" >/dev/null 2>&1
}

wait_for_http() {
  local name="$1"
  local url="$2"
  local timeout="${3:-$HEALTH_TIMEOUT_SECONDS}"
  local elapsed=0
  require_cmd curl
  while (( elapsed < timeout )); do
    if curl --fail --silent --max-time 2 "$url" >/dev/null 2>&1; then
      log "$name is healthy: $url"
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  return 1
}

download_tools_python() {
  local python="$VENVS_ROOT/download-tools/bin/python"
  if [[ ! -x "$python" ]]; then
    bash "$ROOT/scripts/setup_download_tools.sh"
  fi
  printf '%s\n' "$python"
}

hf_download() {
  local repo_id="$1"
  local local_dir="$2"
  local revision="${3:-main}"
  shift 3 || true
  local python
  python="$(download_tools_python)"
  mkdir -p "$local_dir"
  HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$python" "$ROOT/scripts/download_hf.py" \
    --repo-id "$repo_id" \
    --local-dir "$local_dir" \
    --revision "$revision" \
    "$@"
}
