from __future__ import annotations

import math
import sys


def test_extracts_json_from_qwen_markdown_fence(load_service) -> None:
    load_service("orchestrator")
    personality = sys.modules["app.personality"]

    payload = personality.extract_json_object(
        '```json\n{"text": "你好", "emotion": "happy"}\n```'
    )

    assert payload == {"text": "你好", "emotion": "happy"}


def test_normalizes_invalid_emotion_gesture_and_non_finite_speed(
    load_service,
) -> None:
    load_service("orchestrator")
    personality = sys.modules["app.personality"]

    plan = personality.normalize_plan(
        {
            "text": "  你好   世界  ",
            "emotion": "unknown",
            "gesture": "jump",
            "speed": math.nan,
        },
        fallback_text="备用文本",
        emotion_hint="gentle",
        gesture_hint="nod",
    )

    assert plan.text == "你好 世界"
    assert plan.emotion == "gentle"
    assert plan.gesture == "nod"
    assert plan.speed == 1.0
