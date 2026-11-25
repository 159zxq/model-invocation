from __future__ import annotations

import logging
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from common.errors import ServiceError
from common.media import media_tail, probe_duration, validate_video
from common.settings import MODELS_ROOT, REPOS_ROOT, env, env_float, env_int

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoSettings:
    repo_dir: Path
    model_root: Path
    base_model: Path
    transformer_path: Path
    wav2vec_dir: Path
    config_path: Path
    fps: int
    max_frames: int
    max_segments: int
    steps: int
    sample_size: int
    timeout_seconds: int
    guidance_scale: float
    audio_guidance_scale: float
    teacache_threshold: float
    default_prompt: str
    max_prompt_chars: int

    @property
    def segment_seconds(self) -> float:
        return self.max_frames / self.fps

    @property
    def max_total_seconds(self) -> float:
        return self.segment_seconds * self.max_segments

    @classmethod
    def from_env(cls) -> "VideoSettings":
        repo_dir = Path(
            env("ECHOMIMIC_REPO", str(REPOS_ROOT / "echomimic_v3"))
        )
        model_root = Path(
            env(
                "ECHOMIMIC_MODEL_ROOT",
                str(MODELS_ROOT / "EchoMimicV3" / "echomimicv3-flash-pro"),
            )
        )
        return cls(
            repo_dir=repo_dir,
            model_root=model_root,
            base_model=Path(
                env(
                    "ECHOMIMIC_BASE_MODEL",
                    str(MODELS_ROOT / "Wan2.1-Fun-V1.1-1.3B-InP"),
                )
            ),
            transformer_path=Path(
                env(
                    "ECHOMIMIC_TRANSFORMER",
                    str(
                        model_root / "diffusion_pytorch_model.safetensors"
                    ),
                )
            ),
            wav2vec_dir=Path(
                env(
                    "ECHOMIMIC_WAV2VEC",
                    str(MODELS_ROOT / "chinese-wav2vec2-base"),
                )
            ),
            config_path=Path(
                env(
                    "ECHOMIMIC_CONFIG",
                    str(repo_dir / "config" / "config.yaml"),
                )
            ),
            fps=env_int("VIDEO_FPS", 25),
            max_frames=env_int("VIDEO_MAX_FRAMES", 81),
            max_segments=env_int("VIDEO_MAX_SEGMENTS", 3),
            steps=env_int("VIDEO_STEPS", 8),
            sample_size=env_int("VIDEO_SAMPLE_SIZE", 768),
            timeout_seconds=env_int("VIDEO_TIMEOUT_SECONDS", 7200),
            guidance_scale=env_float("VIDEO_GUIDANCE_SCALE", 6.0),
            audio_guidance_scale=env_float(
                "VIDEO_AUDIO_GUIDANCE_SCALE", 3.0
            ),
            teacache_threshold=env_float(
                "VIDEO_TEACACHE_THRESHOLD", 0.1
            ),
            default_prompt=env(
                "VIDEO_PROMPT",
                "A person is speaking naturally, full body visible head to toe, "
                "subtle natural head movement and restrained gestures.",
            ),
            max_prompt_chars=env_int("VIDEO_MAX_PROMPT_CHARS", 800),
        )


class EchoMimicEngine:
    def __init__(self, settings: VideoSettings) -> None:
        self.settings = settings

    def _validate_installation(self) -> None:
        checks = (
            (self.settings.repo_dir / "infer_flash.py", "inference script"),
            (self.settings.config_path, "config"),
            (self.settings.base_model, "Wan2.1 base model"),
            (self.settings.transformer_path, "Flash transformer"),
            (self.settings.wav2vec_dir, "wav2vec2 encoder"),
        )
        for path, label in checks:
            if not path.exists():
                raise ServiceError(
                    f"EchoMimic {label} is missing: {path}",
                    code="video_model_missing",
                    status_code=503,
                )

    @staticmethod
    def _run_command(
        command: list[str],
        *,
        cwd: Path,
        log_path: Path,
        timeout_seconds: int,
    ) -> None:
        env = os.environ.copy()
        repo_path = str(cwd)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{repo_path}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else repo_path
        )
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        env.setdefault("OMP_NUM_THREADS", "4")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        popen_options: dict[str, object] = {}
        if os.name == "posix":
            popen_options["start_new_session"] = True

        with log_path.open("ab") as log_file:
            log_file.write(
                f"\n$ {' '.join(command)}\n".encode("utf-8")
            )
            log_file.flush()
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                **popen_options,
            )
            started = time.monotonic()
            while process.poll() is None:
                if time.monotonic() - started > timeout_seconds:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                    raise ServiceError(
                        "EchoMimic inference timed out.",
                        code="video_timeout",
                        status_code=504,
                        details={"log_tail": media_tail(log_path)},
                    )
                time.sleep(1.0)

        if process.returncode != 0:
            raise ServiceError(
                "EchoMimic inference failed.",
                code="video_inference_failed",
                status_code=500,
                details={
                    "returncode": process.returncode,
                    "log_tail": media_tail(log_path),
                },
            )

    def _split_audio(
        self,
        audio_path: Path,
        parts_dir: Path,
        *,
        duration: float,
        segment_count: int,
    ) -> list[tuple[Path, float]]:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ServiceError(
                "ffmpeg is not installed. Run scripts/setup_system.sh.",
                code="ffmpeg_missing",
                status_code=503,
            )
        parts_dir.mkdir(parents=True, exist_ok=True)
        segment_duration = duration / segment_count
        parts: list[tuple[Path, float]] = []
        for index in range(segment_count):
            start = index * segment_duration
            length = (
                duration - start
                if index == segment_count - 1
                else segment_duration
            )
            output = parts_dir / f"speech-{index:03d}.wav"
            result = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(audio_path),
                    "-ss",
                    f"{start:.3f}",
                    "-t",
                    f"{length:.3f}",
                    "-ac",
                    "1",
                    "-ar",
                    "24000",
                    "-c:a",
                    "pcm_s16le",
                    str(output),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise ServiceError(
                    "Audio could not be split for video generation.",
                    code="audio_split_failed",
                    status_code=500,
                    details={"ffmpeg": result.stderr[-500:]},
                )
            parts.append((output, length))
        return parts

    def _concat_videos(self, videos: list[Path], output_path: Path) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ServiceError(
                "ffmpeg is not installed. Run scripts/setup_system.sh.",
                code="ffmpeg_missing",
                status_code=503,
            )
        concat_file = output_path.parent / "concat.txt"
        lines = ["ffconcat version 1.0"]
        for video in videos:
            escaped = str(video.resolve()).replace("'", "'\\''")
            lines.append(f"file '{escaped}'")
        concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        base_command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
        ]
        copy_result = subprocess.run(
            [
                *base_command,
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if copy_result.returncode == 0:
            return

        logger.warning(
            "Stream-copy concat failed; re-encoding final video: %s",
            copy_result.stderr[-500:],
        )
        encode_result = subprocess.run(
            [
                *base_command,
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if encode_result.returncode != 0:
            raise ServiceError(
                "Generated video segments could not be joined.",
                code="video_concat_failed",
                status_code=500,
                details={"ffmpeg": encode_result.stderr[-800:]},
            )

    def _build_command(
        self,
        *,
        image_path: Path,
        audio_path: Path,
        output_dir: Path,
        prompt: str,
        seed: int,
    ) -> list[str]:
        return [
            sys.executable,
            str(self.settings.repo_dir / "infer_flash.py"),
            "--image_path",
            str(image_path.resolve()),
            "--audio_path",
            str(audio_path.resolve()),
            "--prompt",
            prompt,
            "--num_inference_steps",
            str(self.settings.steps),
            "--config_path",
            str(self.settings.config_path.resolve()),
            "--model_name",
            str(self.settings.base_model.resolve()),
            "--ckpt_idx",
            "50000",
            "--transformer_path",
            str(self.settings.transformer_path.resolve()),
            "--save_path",
            str(output_dir.resolve()),
            "--wav2vec_model_dir",
            str(self.settings.wav2vec_dir.resolve()),
            "--sampler_name",
            "Flow_Unipc",
            "--video_length",
            str(self.settings.max_frames),
            "--guidance_scale",
            f"{self.settings.guidance_scale:g}",
            "--audio_guidance_scale",
            f"{self.settings.audio_guidance_scale:g}",
            "--seed",
            str(seed),
            "--enable_teacache",
            "--teacache_threshold",
            f"{self.settings.teacache_threshold:g}",
            "--num_skip_start_steps",
            "5",
            "--ulysses_degree",
            "1",
            "--ring_degree",
            "1",
            "--weight_dtype",
            "float16",
            "--sample_size",
            str(self.settings.sample_size),
            str(self.settings.sample_size),
            "--fps",
            str(self.settings.fps),
            "--shift",
            "5.0",
        ]

    def generate(
        self,
        *,
        image_path: Path,
        audio_path: Path,
        output_path: Path,
        prompt: str | None = None,
        seed: int | None = None,
        log_path: Path,
    ) -> dict[str, object]:
        if not image_path.is_file() or not audio_path.is_file():
            raise ServiceError(
                "Video inputs are missing.",
                code="video_input_missing",
                status_code=422,
            )

        source_duration = probe_duration(audio_path)
        if source_duration < 0.4:
            raise ServiceError(
                "Audio is too short for video generation.",
                code="audio_too_short",
                status_code=422,
            )
        effective_duration = min(
            source_duration,
            self.settings.max_total_seconds,
        )
        segment_count = min(
            self.settings.max_segments,
            max(1, math.ceil(effective_duration / self.settings.segment_seconds)),
        )
        clean_prompt = " ".join((prompt or self.settings.default_prompt).split())
        if not clean_prompt:
            clean_prompt = self.settings.default_prompt
        if len(clean_prompt) > self.settings.max_prompt_chars:
            raise ServiceError(
                "video prompt is too long.",
                code="video_prompt_too_long",
                status_code=422,
            )
        self._validate_installation()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        parts_dir = output_path.parent / "segments"
        audio_parts = self._split_audio(
            audio_path,
            parts_dir / "audio",
            duration=effective_duration,
            segment_count=segment_count,
        )

        part_videos: list[Path] = []
        resolved_seed = 43 if seed is None else seed
        for index, (part_audio, part_duration) in enumerate(audio_parts):
            part_output_dir = parts_dir / f"video-{index:03d}"
            part_output_dir.mkdir(parents=True, exist_ok=True)
            command = self._build_command(
                image_path=image_path,
                audio_path=part_audio,
                output_dir=part_output_dir,
                prompt=clean_prompt,
                seed=resolved_seed + index,
            )
            logger.info(
                "Generating video segment %d/%d: %.2fs audio",
                index + 1,
                segment_count,
                part_duration,
            )
            self._run_command(
                command,
                cwd=self.settings.repo_dir,
                log_path=log_path,
                timeout_seconds=self.settings.timeout_seconds,
            )
            candidates = sorted(
                path
                for path in part_output_dir.glob("*.mp4")
                if not path.name.endswith("_tmp.mp4")
            )
            if not candidates:
                raise ServiceError(
                    "EchoMimic produced no MP4 segment.",
                    code="invalid_generated_video",
                    status_code=500,
                    details={"log_tail": media_tail(log_path)},
                )
            part_video = candidates[0]
            validate_video(part_video)
            part_videos.append(part_video)

        if len(part_videos) == 1:
            shutil.copy2(part_videos[0], output_path)
        else:
            self._concat_videos(part_videos, output_path)
        validate_video(output_path)

        return {
            "video_path": str(output_path),
            "duration_seconds": round(effective_duration, 3),
            "source_audio_duration_seconds": round(source_duration, 3),
            "truncated": source_duration > effective_duration + 0.05,
            "segment_count": len(part_videos),
            "segment_seconds": round(self.settings.segment_seconds, 3),
            "prompt": clean_prompt,
            "seed": resolved_seed,
            "weight_dtype": "float16",
            "log_path": str(log_path),
        }


ENGINE = EchoMimicEngine(VideoSettings.from_env())
