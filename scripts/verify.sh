#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib/common.sh"

STRICT="${VERIFY_STRICT:-0}"
FAILURES=0
WARNINGS=0

fail() {
  printf 'FAIL: %s\n' "$*" >&2
  FAILURES=$((FAILURES + 1))
}

warn_check() {
  printf 'WARN: %s\n' "$*" >&2
  WARNINGS=$((WARNINGS + 1))
}

ok() {
  printf 'OK:   %s\n' "$*"
}

for script in "$ROOT"/scripts/*.sh "$ROOT"/scripts/lib/*.sh; do
  if bash -n "$script"; then
    ok "shell syntax $(basename "$script")"
  else
    fail "shell syntax $script"
  fi
done

PYTHON="$VENVS_ROOT/orchestrator/bin/python"
if [[ -x "$PYTHON" ]]; then
  if "$PYTHON" -m compileall -q "$ROOT/services" "$ROOT/scripts"; then
    ok "Python compileall"
  else
    fail "Python compileall"
  fi
  if "$PYTHON" -m pip check >/dev/null 2>&1; then
    ok "orchestrator dependency check"
  else
    fail "orchestrator dependency check"
  fi
else
  fail "orchestrator venv is missing; run scripts/setup_all.sh"
fi

for venv_name in vision-service tts-service video-service orchestrator; do
  service_python="$VENVS_ROOT/$venv_name/bin/python"
  if [[ -x "$service_python" ]]; then
    if "$service_python" -m pip check >/dev/null 2>&1; then
      ok "$venv_name dependencies"
    else
      fail "$venv_name dependency conflicts; run setup again"
    fi
  else
    fail "$venv_name venv is missing"
  fi
done

DEV_PYTHON="$VENVS_ROOT/dev-tests/bin/python"
if [[ -x "$DEV_PYTHON" ]]; then
  if (cd "$ROOT" && "$DEV_PYTHON" -m pytest -q); then
    ok "repository pytest suite"
  else
    fail "repository pytest suite"
  fi
  if (cd "$ROOT" && "$DEV_PYTHON" -m ruff check services scripts tests); then
    ok "Ruff checks"
  else
    fail "Ruff checks"
  fi
else
  warn_check "dev-tests venv is missing; pytest and Ruff were skipped"
fi

MODEL_CHECKS=(
  "$MODELS_ROOT/stable-diffusion-xl-base-1.0"
  "$MODELS_ROOT/realvisxl-v5.0/RealVisXL_V5.0_fp16.safetensors"
  "$MODELS_ROOT/controlnet-openpose-sdxl-1.0"
  "$MODELS_ROOT/IP-Adapter-FaceID"
  "$MODELS_ROOT/insightface/models/buffalo_l"
  "$MODELS_ROOT/Fun-CosyVoice3-0.5B-2512/cosyvoice3.yaml"
  "$REPOS_ROOT/CosyVoice/cosyvoice"
  "$REPOS_ROOT/echomimic_v3/infer_flash.py"
  "$MODELS_ROOT/Wan2.1-Fun-V1.1-1.3B-InP"
  "$MODELS_ROOT/chinese-wav2vec2-base"
  "$MODELS_ROOT/EchoMimicV3/echomimicv3-flash-pro/diffusion_pytorch_model.safetensors"
)
for model_path in "${MODEL_CHECKS[@]}"; do
  if [[ -e "$model_path" ]]; then
    ok "model path $(basename "$model_path")"
  else
    if [[ "$STRICT" == "1" ]]; then
      fail "missing $model_path"
    else
      warn_check "missing $model_path"
    fi
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader || fail "nvidia-smi failed"
  VISION_PYTHON="$VENVS_ROOT/vision-service/bin/python"
  if [[ -x "$VISION_PYTHON" ]]; then
    if "$VISION_PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
print(f"CUDA {torch.version.cuda}, device {torch.cuda.get_device_name(0)}")
PY
    then
      ok "PyTorch CUDA"
    else
      fail "PyTorch CUDA check"
    fi
  fi
else
  warn_check "nvidia-smi not found; GPU checks skipped"
fi

printf '\nService health (informational):\n'
bash "$ROOT/scripts/status.sh" || warn_check "one or more services are not running"

printf '\nVerification summary: %d failure(s), %d warning(s)\n' \
  "$FAILURES" "$WARNINGS"
if ((FAILURES > 0)); then
  exit 1
fi
