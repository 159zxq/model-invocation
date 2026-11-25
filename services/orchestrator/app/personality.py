from __future__ import annotations

import json
import logging
import math
from typing import Any

import httpx
from pydantic import BaseModel, Field

from common.errors import ServiceError
from common.settings import env, env_bool, env_float, env_int

logger = logging.getLogger(__name__)

VALID_EMOTIONS = {
    "neutral",
    "happy",
    "sad",
    "excited",
    "serious",
    "gentle",
    "angry",
}
VALID_GESTURES = {"neutral", "greet", "explain", "emphasis", "nod"}
GESTURE_PROMPTS = {
    "neutral": (
        "A person is speaking naturally with the full body visible head to toe, "
        "subtle head movement."
    ),
    "greet": (
        "A person is speaking naturally with the full body visible head to toe "
        "and greeting the viewer warmly."
    ),
    "explain": (
        "A person is speaking naturally with the full body visible head to toe "
        "and small explanatory gestures."
    ),
    "emphasis": (
        "A person is speaking naturally with the full body visible head to toe "
        "and restrained emphasis gestures."
    ),
    "nod": (
        "A person is speaking naturally with the full body visible head to toe, "
        "small nods and attentive expression."
    ),
}


class PersonalityPlan(BaseModel):
    text: str = Field(min_length=1)
    emotion: str = "neutral"
    gesture: str = "neutral"
    speed: float = 1.0
    video_prompt: str = Field(min_length=1)


def _max_reply_chars() -> int:
    return env_int("PERSONALITY_MAX_REPLY_CHARS", 60)


def _clean_text(value: Any, *, max_chars: int, default: str = "") -> str:
    if value is None:
        return default
    text = " ".join(str(value).strip().split())
    return text[:max_chars] if text else default


def fallback_plan(
    user_text: str,
    *,
    emotion_hint: str = "auto",
    gesture_hint: str = "auto",
    reason: str | None = None,
) -> dict[str, Any]:
    emotion = emotion_hint if emotion_hint in VALID_EMOTIONS else "neutral"
    gesture = gesture_hint if gesture_hint in VALID_GESTURES else "greet"
    text = user_text.strip()
    if not text:
        text = "你好，我是你的数字人助手。很高兴认识你，有什么想聊的吗？"
    plan = PersonalityPlan(
        text=_clean_text(text, max_chars=_max_reply_chars()),
        emotion=emotion,
        gesture=gesture,
        speed=1.0,
        video_prompt=GESTURE_PROMPTS[gesture],
    )
    return {
        "plan": plan.model_dump(),
        "source": "fallback",
        "fallback_reason": reason,
    }


def extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in model response.")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Model response JSON is not an object.")
    return payload


def normalize_plan(
    payload: dict[str, Any],
    *,
    fallback_text: str,
    emotion_hint: str = "auto",
    gesture_hint: str = "auto",
) -> PersonalityPlan:
    text = _clean_text(
        payload.get("text"),
        max_chars=_max_reply_chars(),
        default=fallback_text or "你好，很高兴认识你。",
    )
    emotion = _clean_text(payload.get("emotion"), max_chars=24).lower()
    if emotion not in VALID_EMOTIONS:
        emotion = (
            emotion_hint
            if emotion_hint in VALID_EMOTIONS
            else "neutral"
        )
    gesture = _clean_text(payload.get("gesture"), max_chars=24).lower()
    if gesture not in VALID_GESTURES:
        gesture = (
            gesture_hint
            if gesture_hint in VALID_GESTURES
            else "neutral"
        )
    try:
        speed = float(payload.get("speed", 1.0))
    except (TypeError, ValueError):
        speed = 1.0
    speed = (
        min(1.2, max(0.8, speed))
        if math.isfinite(speed)
        else 1.0
    )
    video_prompt = _clean_text(
        payload.get("video_prompt"),
        max_chars=500,
        default=GESTURE_PROMPTS[gesture],
    )
    return PersonalityPlan(
        text=text,
        emotion=emotion,
        gesture=gesture,
        speed=speed,
        video_prompt=video_prompt,
    )


def build_messages(
    *,
    user_text: str,
    persona: str,
    scene: str,
    age: int,
    gender: str,
) -> list[dict[str, str]]:
    system = (
        "你是中文数字人的数字人格引擎。你的任务是为一次语音和视频生成"
        "规划自然、简短、适合口播的回复。只输出一个 JSON 对象，不要输出"
        "Markdown、解释或思考过程。JSON 必须包含：text、emotion、gesture、"
        "speed、video_prompt。"
        "emotion 只能是 neutral、happy、sad、excited、serious、gentle、angry；"
        "gesture 只能是 neutral、greet、explain、emphasis、nod；"
        "speed 在 0.8 到 1.2 之间。"
        f"text 使用自然中文口语，通常 15 到 45 个汉字，最多 {_max_reply_chars()} 个汉字，"
        "不能提及自己是模型、API 或 AI 生成的视频，也不能冒充真实人类。"
        "video_prompt 使用英文，描述说话、头部动作、全身站姿和可见的手部动作，"
        "要求 full body visible head to toe；不要描述具体身份或添加画面文字。"
    )
    profile = {
        "user_text": user_text,
        "persona": persona[:500],
        "scene": scene[:500],
        "age": age,
        "gender": gender,
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "请根据以下 JSON 生成回复规划：\n"
            + json.dumps(profile, ensure_ascii=False),
        },
    ]


async def create_personality_plan(
    *,
    user_text: str,
    persona: str,
    scene: str,
    age: int,
    gender: str,
    emotion_hint: str = "auto",
    gesture_hint: str = "auto",
) -> dict[str, Any]:
    api_key = env("SILICONFLOW_API_KEY").strip()
    allow_fallback = env_bool("PERSONALITY_ALLOW_FALLBACK", True)
    if not api_key:
        if allow_fallback:
            return fallback_plan(
                user_text,
                emotion_hint=emotion_hint,
                gesture_hint=gesture_hint,
                reason="SILICONFLOW_API_KEY is not configured.",
            )
        raise ServiceError(
            "SILICONFLOW_API_KEY is not configured.",
            code="personality_not_configured",
            status_code=503,
        )

    base_url = env(
        "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"
    ).rstrip("/")
    payload: dict[str, Any] = {
        "model": env("PERSONALITY_MODEL", "Qwen/Qwen3-8B"),
        "messages": build_messages(
            user_text=user_text,
            persona=persona,
            scene=scene,
            age=age,
            gender=gender,
        ),
        "temperature": env_float("PERSONALITY_TEMPERATURE", 0.7),
        "max_tokens": env_int("PERSONALITY_MAX_TOKENS", 700),
        "stream": False,
    }
    if env("PERSONALITY_PROVIDER", "siliconflow") == "siliconflow":
        payload["enable_thinking"] = False

    timeout = env_float("PERSONALITY_TIMEOUT_SECONDS", 120.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        raw_plan = extract_json_object(str(content))
        plan = normalize_plan(
            raw_plan,
            fallback_text=user_text,
            emotion_hint=emotion_hint,
            gesture_hint=gesture_hint,
        )
        return {
            "plan": plan.model_dump(),
            "source": "siliconflow",
            "model": payload["model"],
        }
    except Exception as exc:
        logger.exception("SiliconFlow personality planning failed")
        if allow_fallback:
            return fallback_plan(
                user_text,
                emotion_hint=emotion_hint,
                gesture_hint=gesture_hint,
                reason=f"{type(exc).__name__}: {str(exc)[:240]}",
            )
        raise ServiceError(
            "SiliconFlow personality planning failed.",
            code="personality_upstream_failed",
            status_code=502,
            details={"exception": type(exc).__name__},
        ) from exc
