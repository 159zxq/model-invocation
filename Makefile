SHELL := /bin/bash
ROOT := $(CURDIR)

.PHONY: setup setup-python setup-system setup-vision setup-tts setup-video setup-orchestrator setup-dev downloads start stop status verify test

setup:
	bash scripts/setup_all.sh

setup-python:
	bash scripts/setup_python.sh

setup-system:
	bash scripts/setup_system.sh

setup-vision:
	bash scripts/setup_vision.sh

setup-tts:
	bash scripts/setup_tts.sh

setup-video:
	bash scripts/setup_video.sh

setup-orchestrator:
	bash scripts/setup_orchestrator.sh

setup-dev:
	bash scripts/setup_dev.sh

downloads:
	bash scripts/download_vision_models.sh
	bash scripts/download_tts_models.sh
	bash scripts/download_video_models.sh

start:
	bash scripts/start_all.sh

stop:
	bash scripts/stop_all.sh

status:
	bash scripts/status.sh

verify:
	bash scripts/verify.sh

test:
	runtime/venvs/dev-tests/bin/python -m pytest -q
