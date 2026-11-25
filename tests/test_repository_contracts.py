from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gpu_setup_installs_pinned_torch_before_service_dependencies() -> None:
    for script_name in ("setup_vision.sh", "setup_tts.sh", "setup_video.sh"):
        content = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert content.index("install_torch") < content.index("uv_pip_install -r")
        assert "verify_torch" in content


def test_model_snapshots_are_not_floating_main_revisions() -> None:
    content = (ROOT / "configs" / "download.env").read_text(encoding="utf-8")
    revisions = [
        line.split("=", 1)[1]
        for line in content.splitlines()
        if line.endswith("_REVISION=main")
    ]
    assert revisions == []


def test_cosyvoice_inference_dependencies_include_runtime_imports() -> None:
    content = (
        ROOT / "services" / "tts" / "cosyvoice-requirements.txt"
    ).read_text(encoding="utf-8")

    for package in (
        "openai-whisper",
        "tqdm",
        "omegaconf",
        "einops",
        "regex",
        "scipy",
        "hydra-core",
        "lightning",
    ):
        assert package in content


def test_env_values_with_spaces_are_quoted() -> None:
    offenders: list[str] = []
    env_files = [ROOT / ".env.example", *sorted((ROOT / "configs").glob("*.env"))]

    for path in env_files:
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line.removeprefix("export ").lstrip()

            _, separator, raw_value = line.partition("=")
            if not separator:
                continue
            value = raw_value.strip()
            if any(character.isspace() for character in value) and not (
                len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}
            ):
                offenders.append(f"{path.relative_to(ROOT)}:{line_number}")

    assert not offenders, (
        "Environment values containing spaces must be wrapped in matching quotes: "
        + ", ".join(offenders)
    )
