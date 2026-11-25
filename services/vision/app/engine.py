from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from common.errors import ServiceError
from common.media import contain_square_image
from common.settings import MODELS_ROOT, env, env_bool, env_float, env_int

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisionSettings:
    base_model: Path
    checkpoint: Path
    controlnet: Path
    ip_adapter: Path
    ip_weight: str
    insightface_root: Path
    width: int
    height: int
    steps: int
    guidance: float
    control_scale: float
    ip_scale: float
    device: str
    keep_warm: bool
    cpu_offload: bool
    attention_slicing: bool

    @classmethod
    def from_env(cls) -> "VisionSettings":
        return cls(
            base_model=Path(
                env(
                    "VISION_BASE_MODEL",
                    str(MODELS_ROOT / "stable-diffusion-xl-base-1.0"),
                )
            ),
            checkpoint=Path(
                env(
                    "VISION_BASE_CHECKPOINT",
                    str(
                        MODELS_ROOT
                        / "realvisxl-v5.0"
                        / "RealVisXL_V5.0_fp16.safetensors"
                    ),
                )
            ),
            controlnet=Path(
                env(
                    "VISION_CONTROLNET",
                    str(MODELS_ROOT / "controlnet-openpose-sdxl-1.0"),
                )
            ),
            ip_adapter=Path(
                env(
                    "VISION_IP_ADAPTER",
                    str(MODELS_ROOT / "IP-Adapter-FaceID"),
                )
            ),
            ip_weight=env("VISION_IP_WEIGHT", "ip-adapter-faceid_sdxl.bin"),
            insightface_root=Path(
                env(
                    "VISION_INSIGHTFACE_ROOT",
                    str(MODELS_ROOT / "insightface"),
                )
            ),
            width=env_int("VISION_WIDTH", 832),
            height=env_int("VISION_HEIGHT", 1216),
            steps=env_int("VISION_STEPS", 28),
            guidance=env_float("VISION_GUIDANCE", 5.0),
            control_scale=env_float("VISION_CONTROL_SCALE", 0.65),
            ip_scale=env_float("VISION_IP_SCALE", 0.72),
            device=env("VISION_DEVICE", "cuda"),
            keep_warm=env_bool("VISION_KEEP_WARM", False),
            cpu_offload=env_bool("VISION_CPU_OFFLOAD", False),
            attention_slicing=env_bool("VISION_ATTENTION_SLICING", True),
        )


class VisionEngine:
    def __init__(self, settings: VisionSettings) -> None:
        self.settings = settings
        self._pipeline = None
        self._face_app = None
        self._load_lock = threading.Lock()

    def _load_pipeline(self):
        import torch
        from diffusers import (
            ControlNetModel,
            DDIMScheduler,
            StableDiffusionXLControlNetPipeline,
        )
        for path, label in (
            (self.settings.controlnet, "ControlNet"),
            (self.settings.ip_adapter, "IP-Adapter-FaceID"),
        ):
            if not path.exists():
                raise ServiceError(
                    f"{label} path is missing: {path}",
                    code="vision_model_missing",
                    status_code=503,
                )

        if not torch.cuda.is_available() and self.settings.device.startswith("cuda"):
            raise ServiceError(
                "CUDA is not available in the vision service environment.",
                code="cuda_unavailable",
                status_code=503,
            )

        logger.info("Loading vision models on %s", self.settings.device)
        dtype = torch.float16 if self.settings.device.startswith("cuda") else torch.float32
        controlnet = ControlNetModel.from_pretrained(
            str(self.settings.controlnet),
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipeline = None
        if self.settings.checkpoint.is_file():
            try:
                pipeline = StableDiffusionXLControlNetPipeline.from_single_file(
                    str(self.settings.checkpoint),
                    controlnet=controlnet,
                    torch_dtype=dtype,
                    local_files_only=True,
                    use_safetensors=True,
                )
            except Exception:
                logger.exception(
                    "RealVis checkpoint could not be loaded; falling back to SDXL base"
                )

        if pipeline is None:
            if not self.settings.base_model.exists():
                raise ServiceError(
                    f"SDXL base model directory is missing: {self.settings.base_model}",
                    code="vision_model_missing",
                    status_code=503,
                )
            pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                str(self.settings.base_model),
                controlnet=controlnet,
                torch_dtype=dtype,
                local_files_only=True,
                use_safetensors=True,
            )

        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        pipeline.load_ip_adapter(
            str(self.settings.ip_adapter),
            subfolder=None,
            weight_name=self.settings.ip_weight,
            image_encoder_folder=None,
            local_files_only=True,
        )
        pipeline.set_ip_adapter_scale(self.settings.ip_scale)
        if self.settings.attention_slicing:
            pipeline.enable_attention_slicing()
            pipeline.vae.enable_slicing()
        if self.settings.cpu_offload:
            pipeline.enable_model_cpu_offload()
        else:
            pipeline.to(self.settings.device)
        pipeline.set_progress_bar_config(disable=True)
        return pipeline

    def _ensure_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._load_lock:
            if self._pipeline is None:
                self._pipeline = self._load_pipeline()
        return self._pipeline

    def _ensure_face_app(self):
        if self._face_app is not None:
            return self._face_app
        from insightface.app import FaceAnalysis

        model_dir = self.settings.insightface_root / "models" / "buffalo_l"
        if not model_dir.is_dir():
            raise ServiceError(
                f"InsightFace buffalo_l is missing: {model_dir}",
                code="face_model_missing",
                status_code=503,
            )
        app = FaceAnalysis(
            name="buffalo_l",
            root=str(self.settings.insightface_root),
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        self._face_app = app
        return app

    def _extract_face(self, image_path: Path):
        import cv2

        app = self._ensure_face_app()
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ServiceError(
                "The input photo cannot be read.",
                code="invalid_image",
                status_code=422,
            )
        faces = app.get(bgr)
        if not faces:
            raise ServiceError(
                "No face was detected. Use a clear frontal portrait.",
                code="face_not_detected",
                status_code=422,
            )
        face = max(
            faces,
            key=lambda item: float(
                (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
            ),
        )

        return face.normed_embedding

    @staticmethod
    def _draw_pose(width: int, height: int, bmi: float) -> Image.Image:
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        cx = width // 2
        top = int(height * 0.055)
        bottom = int(height * 0.95)
        torso = bottom - top
        shoulder_y = top + int(torso * 0.23)
        hip_y = top + int(torso * 0.51)
        knee_y = top + int(torso * 0.74)
        ankle_y = top + int(torso * 0.94)
        head_radius = max(18, int(width * 0.075))
        head_y = top + head_radius

        width_factor = min(max(0.22 + (bmi - 18.5) * 0.012, 0.17), 0.34)
        shoulder = int(width * width_factor)
        hip = int(width * (width_factor * 0.78))
        elbow_dx = int(shoulder * 1.55)
        wrist_dx = int(shoulder * 1.95)
        elbow_y = shoulder_y + int(torso * 0.17)
        wrist_y = shoulder_y + int(torso * 0.32)
        knee_dx = max(12, hip // 2)
        ankle_dx = max(10, hip // 3)
        limb_width = max(4, int(width * 0.009))

        colors = [
            (255, 0, 0),
            (255, 85, 0),
            (255, 170, 0),
            (255, 255, 0),
            (170, 255, 0),
            (85, 255, 0),
            (0, 255, 0),
            (0, 255, 85),
            (0, 255, 170),
            (0, 255, 255),
            (0, 170, 255),
            (0, 85, 255),
            (0, 0, 255),
            (85, 0, 255),
            (170, 0, 255),
        ]

        points = {
            "nose": (cx, head_y - head_radius // 3),
            "neck": (cx, shoulder_y - int(torso * 0.05)),
            "l_shoulder": (cx - shoulder, shoulder_y),
            "r_shoulder": (cx + shoulder, shoulder_y),
            "l_elbow": (cx - elbow_dx, elbow_y),
            "r_elbow": (cx + elbow_dx, elbow_y),
            "l_wrist": (cx - wrist_dx, wrist_y),
            "r_wrist": (cx + wrist_dx, wrist_y),
            "l_hip": (cx - hip, hip_y),
            "r_hip": (cx + hip, hip_y),
            "l_knee": (cx - hip - knee_dx, knee_y),
            "r_knee": (cx + hip + knee_dx, knee_y),
            "l_ankle": (cx - hip - ankle_dx, ankle_y),
            "r_ankle": (cx + hip + ankle_dx, ankle_y),
        }
        bones = [
            ("neck", "l_shoulder"),
            ("neck", "r_shoulder"),
            ("l_shoulder", "l_elbow"),
            ("l_elbow", "l_wrist"),
            ("r_shoulder", "r_elbow"),
            ("r_elbow", "r_wrist"),
            ("neck", "l_hip"),
            ("neck", "r_hip"),
            ("l_hip", "r_hip"),
            ("l_hip", "l_knee"),
            ("l_knee", "l_ankle"),
            ("r_hip", "r_knee"),
            ("r_knee", "r_ankle"),
        ]
        for index, (start, end) in enumerate(bones):
            draw.line(
                [points[start], points[end]],
                fill=colors[index % len(colors)],
                width=limb_width,
            )
        for index, point in enumerate(points.values()):
            radius = limb_width + 2
            draw.ellipse(
                [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
                fill=colors[index % len(colors)],
            )
        return canvas

    @staticmethod
    def _make_video_reference(fullbody: Image.Image) -> Image.Image:
        return contain_square_image(fullbody, size=768)

    def generate(
        self,
        *,
        photo_path: Path,
        output_dir: Path,
        height_cm: float,
        weight_kg: float,
        age: int,
        gender: str,
        style: str,
        seed: int | None = None,
    ) -> dict[str, object]:
        if not 120 <= height_cm <= 230:
            raise ServiceError(
                "height_cm must be between 120 and 230.",
                code="invalid_height",
                status_code=422,
            )
        if not 30 <= weight_kg <= 250:
            raise ServiceError(
                "weight_kg must be between 30 and 250.",
                code="invalid_weight",
                status_code=422,
            )
        if not 16 <= age <= 100:
            raise ServiceError(
                "age must be between 16 and 100.",
                code="invalid_age",
                status_code=422,
            )
        if not photo_path.is_file():
            raise ServiceError(
                "The normalized input photo is missing.",
                code="invalid_image",
                status_code=422,
            )

        import numpy as np
        import torch

        output_dir.mkdir(parents=True, exist_ok=True)
        face_embedding = self._extract_face(photo_path)
        bmi = weight_kg / ((height_cm / 100.0) ** 2)
        pose = self._draw_pose(self.settings.width, self.settings.height, bmi)

        pipeline = self._ensure_pipeline()
        device = torch.device(self.settings.device)
        embeds = torch.from_numpy(np.asarray(face_embedding)).float()
        embeds = embeds.unsqueeze(0).unsqueeze(0)
        # SDXL CFG splits this pair into negative and positive image embeddings.
        id_embeds = torch.cat([torch.zeros_like(embeds), embeds], dim=0)
        id_embeds = id_embeds.to(device=device, dtype=torch.float16)

        person = {
            "male": "man",
            "female": "woman",
            "nonbinary": "person",
        }.get(gender.lower(), "person")
        body = "slim" if bmi < 18.5 else "average build" if bmi < 25 else "fuller build"
        style_text = {
            "realistic": "photorealistic editorial full-body portrait",
            "business": "professional business full-body portrait, tailored clothing",
            "casual": "natural casual outfit",
            "fashion": "high fashion editorial styling",
            "studio": "clean studio portrait, seamless backdrop",
        }.get(style.lower(), "photorealistic portrait")
        prompt = (
            f"full body portrait of one adult {person}, head to toe, standing straight, "
            f"front facing, age {age}, height {height_cm:.0f} centimeters, {body}, "
            f"{style_text}, anatomically correct, natural hands, detailed face, "
            "balanced proportions, high quality, sharp focus, realistic skin texture"
        )
        negative = (
            "cropped body, close-up, portrait only, out of frame, duplicate person, "
            "deformed anatomy, extra limbs, extra fingers, fused fingers, bad hands, "
            "bad feet, asymmetrical eyes, blurry, low resolution, watermark, text, logo"
        )
        generator = torch.Generator(device="cpu")
        resolved_seed = seed if seed is not None else int(torch.randint(0, 2**31 - 1, (1,)))
        generator.manual_seed(resolved_seed)

        logger.info(
            "Generating avatar: height=%.1f weight=%.1f age=%d gender=%s bmi=%.1f",
            height_cm,
            weight_kg,
            age,
            gender,
            bmi,
        )
        result = pipeline(
            prompt=prompt,
            negative_prompt=negative,
            image=pose,
            controlnet_conditioning_scale=self.settings.control_scale,
            ip_adapter_image_embeds=[id_embeds],
            width=self.settings.width,
            height=self.settings.height,
            num_inference_steps=self.settings.steps,
            guidance_scale=self.settings.guidance,
            generator=generator,
        )
        fullbody = result.images[0].convert("RGB")
        fullbody_path = output_dir / "fullbody.png"
        fullbody.save(fullbody_path, format="PNG")

        video_reference = self._make_video_reference(fullbody)
        reference_path = output_dir / "video_reference.png"
        video_reference.save(reference_path, format="PNG")
        pose_path = output_dir / "pose.png"
        pose.save(pose_path, format="PNG")

        if not self.settings.keep_warm:
            self.unload()
        return {
            "fullbody_path": str(fullbody_path),
            "video_reference_path": str(reference_path),
            "pose_path": str(pose_path),
            "seed": resolved_seed,
            "prompt": prompt,
            "bmi": round(bmi, 2),
            "width": fullbody.width,
            "height": fullbody.height,
        }

    def unload(self) -> None:
        if self._pipeline is None:
            return
        self._pipeline = None
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            logger.debug("CUDA cache cleanup skipped", exc_info=True)


ENGINE = VisionEngine(VisionSettings.from_env())
