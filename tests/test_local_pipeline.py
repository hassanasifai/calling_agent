from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nrsp_agent.voice.local_pipeline import (
    LocalVoicePipelineSettings,
    load_local_voice_settings,
)


def test_local_pipeline_defaults_are_urdu_and_offline() -> None:
    settings = LocalVoicePipelineSettings()

    assert settings.tts_voice == "ur_PK-fasih-medium"
    assert settings.asr_model == "small"
    assert settings.llm_model == "qwen3:1.7b"
    assert settings.ollama_base_url == "http://127.0.0.1:11434/v1"
    assert "OTP" in settings.system_prompt


def test_ollama_endpoint_rejects_remote_service() -> None:
    with pytest.raises(ValidationError, match="must be loopback"):
        LocalVoicePipelineSettings(ollama_base_url="https://paid.example/v1")


def test_demo_profile_loads() -> None:
    settings = load_local_voice_settings(Path("config/demo.toml"))

    assert settings.llm_model == "qwen3:1.7b"
    assert settings.asr_compute_type == "int8"
