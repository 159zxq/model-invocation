from __future__ import annotations

import gc
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from common.errors import ServiceError
from common.settings import MODELS_ROOT, REPOS_ROOT, env, env_bool, env_int

logger = logging.getLogger(__name__)

EMOTION_INSTRUCTIONS = {
    "neutral": "平静、自然",
    "happy": "开心、亲切",
    "sad": "低落但清晰",
    "excited": "兴奋、有活力",
    "serious": "认真、沉稳",
    "gentle": "温柔、舒缓",
    "angry": "克制而严肃",
}


@dataclass(frozen=True)
class TTSSettings:
    model_dir: Path
    repo_dir: Path
    device: str
    fp16: bool
    keep_warm: bool
    sample_rate: int
    max_text_chars: int
    instruct_prefix: str

    @classmethod
    def from_env(cls) -> "TTSSettings":
        return cls(
            model_dir=Path(
                env(
                    "TTS_MODEL",
                    str(MODELS_ROOT / "Fun-CosyVoice3-0.5B-2512"),
                )
            ),
            repo_dir=Path(
                env("TTS_REPO", str(REPOS_ROOT / "CosyVoice"))
            ),
            device=env("TTS_DEVICE", "cuda"),
            fp16=env_bool("TTS_FP16", True),
            keep_warm=env_bool("TTS_KEEP_WARM", False),
            sample_rate=env_int("TTS_SAMPLE_RATE", 24000),
            max_text_chars=env_int("TTS_MAX_TEXT_CHARS", 2000),
            instruct_prefix=env(
                "TTS_INSTRUCT_PREFIX",
                "You are a helpful assistant.",
            ).strip(),
        )


class TTSEngine:
    def __init__(self, settings: TTSSettings) -> None:
        self.settings = settings
        self._model = None
        self._load_lock = threading.Lock()

    def _load_model(self):
        import torch

        model_config = self.settings.model_dir / "cosyvoice3.yaml"
        if not model_config.is_file():
            raise ServiceError(
                f"CosyVoice3 model is missing: {model_config}",
                code="tts_model_missing",
                status_code=503,
            )
        if not (self.settings.repo_dir / "cosyvoice").is_dir():
            raise ServiceError(
                f"CosyVoice repository is missing: {self.settings.repo_dir}",
                code="tts_repo_missing",
                status_code=503,
            )
        if self.settings.device.startswith("cuda") and not torch.cuda.is_available():
            raise ServiceError(
                "CUDA is not available in the TTS service environment.",
                code="cuda_unavailable",
                status_code=503,
            )

        repo = str(self.settings.repo_dir)
        matcha = str(self.settings.repo_dir / "third_party" / "Matcha-TTS")
        for path in (matcha, repo):
            if path not in sys.path:
                sys.path.insert(0, path)

        from cosyvoice.cli.cosyvoice import AutoModel

        logger.info(
            "Loading CosyVoice3 on %s (fp16=%s)",
            self.settings.device,
            self.settings.fp16,
        )
        return AutoModel(
            model_dir=str(self.settings.model_dir),
            fp16=self.settings.fp16,
            load_trt=False,
            load_vllm=False,
        )

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                self._model = self._load_model()
        return self._model

    def generate(
        self,
        *,
        text: str,
        reference_wav: Path,
        output_path: Path,
        emotion: str = "neutral",
        speed: float = 1.0,
    ) -> dict[str, object]:
        clean_text = " ".join(text.split())
        if not clean_text:
            raise ServiceError(
                "text must not be empty.",
                code="empty_text",
                status_code=422,
            )
        if len(clean_text) > self.settings.max_text_chars:
            raise ServiceError(
                f"text exceeds the {self.settings.max_text_chars} character limit.",
                code="text_too_long",
                status_code=422,
            )
        if not reference_wav.is_file():
            raise ServiceError(
                "The normalized reference audio is missing.",
                code="reference_audio_missing",
                status_code=422,
            )
        if not 0.5 <= speed <= 2.0:
            raise ServiceError(
                "speed must be between 0.5 and 2.0.",
                code="invalid_speed",
                status_code=422,
            )

        import soundfile as sf
        import torch

        emotion_key = emotion.strip().lower()
        if emotion_key not in EMOTION_INSTRUCTIONS:
            emotion_key = "neutral"
        instruction = (
            f"{self.settings.instruct_prefix} "
            f"请用{EMOTION_INSTRUCTIONS[emotion_key]}的语气说话。"
            "<|endofprompt|>"
        )

        model = self._ensure_model()
        chunks = []
        try:
            for item in model.inference_instruct2(
                clean_text,
                instruction,
                str(reference_wav),
                stream=False,
                speed=speed,
            ):
                speech = item.get("tts_speech")
                if speech is not None and speech.numel() > 0:
                    chunks.append(speech.detach().cpu())
        except Exception as exc:
            logger.exception("CosyVoice inference failed")
            raise ServiceError(
                "Voice generation failed. Check the TTS service log.",
                code="tts_inference_failed",
                status_code=500,
                details={"exception": type(exc).__name__},
            ) from exc

        if not chunks:
            raise ServiceError(
                "Voice generation returned no audio.",
                code="tts_empty_output",
                status_code=500,
            )

        speech = torch.cat(chunks, dim=1).squeeze(0).float().numpy()
        sample_rate = int(
            getattr(model, "sample_rate", self.settings.sample_rate)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(output_path),
            speech,
            sample_rate,
            subtype="PCM_16",
            format="WAV",
        )
        duration = float(len(speech) / sample_rate)

        if not self.settings.keep_warm:
            self.unload()

        return {
            "audio_path": str(output_path),
            "sample_rate": sample_rate,
            "duration_seconds": round(duration, 3),
            "emotion": emotion_key,
            "instruction": instruction,
            "text": clean_text,
            "speed": speed,
        }

    def unload(self) -> None:
        if self._model is None:
            return
        self._model = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            logger.debug("CUDA cache cleanup skipped", exc_info=True)


ENGINE = TTSEngine(TTSSettings.from_env())
