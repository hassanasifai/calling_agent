from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from nrsp_agent.conversation import script
from nrsp_agent.conversation.policy import ConversationPolicy
from nrsp_agent.devtools.loopback import TEST_CUSTOMER
from nrsp_agent.voice.local_pipeline import (
    LocalVoicePipelineSettings,
    build_local_voice_agent,
    load_local_voice_settings,
)
from nrsp_agent.voice.pipecat_transport import DiagoPipecatTransport

MODELS_PRESENT = Path("data/models/piper/ur_PK-fasih-medium.onnx").exists() and any(
    Path("data/models/faster-whisper").glob("models--*small*")
)


def test_local_pipeline_defaults_are_urdu_and_offline() -> None:
    settings = LocalVoicePipelineSettings()

    assert settings.tts_voice == "ur_PK-fasih-medium"
    assert settings.asr_model == "small"
    assert settings.llm_model == "qwen3:1.7b"
    assert settings.ollama_base_url == "http://127.0.0.1:11434/v1"
    assert settings.asr_model_dir == Path("data/models/faster-whisper")
    # Slow ASR must never push a transcript into the following turn.
    assert settings.user_turn_stop_timeout_seconds >= 15
    assert "OTP" in script.DISCUSS_INSTRUCTION


def test_ollama_endpoint_rejects_remote_service() -> None:
    with pytest.raises(ValidationError, match="must be loopback"):
        LocalVoicePipelineSettings(ollama_base_url="https://paid.example/v1")


def test_demo_profile_loads() -> None:
    settings = load_local_voice_settings(Path("config/demo.toml"))

    assert settings.llm_model == "qwen3:1.7b"
    assert settings.asr_compute_type == "int8"


def test_server_profile_loads() -> None:
    settings = load_local_voice_settings(Path("config/server.toml"))

    assert settings.llm_model == "qwen3:8b"


@pytest.mark.skipif(not MODELS_PRESENT, reason="local model weights not downloaded")
def test_real_pipeline_builds_with_shared_models() -> None:
    class Port:  # construction only; no media flows
        pass

    settings = LocalVoicePipelineSettings()
    first = build_local_voice_agent(
        DiagoPipecatTransport(Port(), "call-1"),  # type: ignore[arg-type]
        settings,
        ConversationPolicy(TEST_CUSTOMER),
    )
    second = build_local_voice_agent(
        DiagoPipecatTransport(Port(), "call-2"),  # type: ignore[arg-type]
        settings,
        ConversationPolicy(TEST_CUSTOMER),
    )

    def whisper(processor):
        if "Whisper" in processor.name:
            return [processor]
        children = getattr(processor, "processors", None) or []
        return [found for child in children for found in whisper(child)]

    stt = whisper(first.worker._pipeline)
    stt2 = whisper(second.worker._pipeline)
    assert len(stt) == len(stt2) == 1
    # Two calls share one loaded ASR model instead of reloading weights.
    assert stt[0]._model is stt2[0]._model
    assert first.record is not second.record
