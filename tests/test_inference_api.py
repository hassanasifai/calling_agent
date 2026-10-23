from __future__ import annotations

import struct
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer
from pydantic import ValidationError

import nrsp_agent.inference_api as api
from nrsp_agent.voice.local_pipeline import LocalVoicePipelineSettings

TOKEN = "t" * 32
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakeVoice:
    config = SimpleNamespace(sample_rate=22050)

    def synthesize(self, text: str):
        for _ in text.split("۔"):
            yield SimpleNamespace(audio_int16_bytes=b"\x01\x00" * 100)


@pytest.fixture
def stub_models(monkeypatch):
    calls = []
    monkeypatch.setattr(api, "shared_whisper_model", lambda *a, **k: "model")

    def transcribe(model, pcm, language, prompt, threshold):
        calls.append((len(pcm), language, prompt, threshold))
        return "جی ہاں"

    monkeypatch.setattr(api, "transcribe_pcm16", transcribe)
    monkeypatch.setattr(api._CachedPiperVoice, "load", lambda *a, **k: FakeVoice())
    return calls


def settings(**overrides) -> api.InferenceApiSettings:
    values = {"bind_host": "127.0.0.1", "allowed_clients": ["127.0.0.1/32"]}
    return api.InferenceApiSettings(**{**values, **overrides})


async def client_for(s: api.InferenceApiSettings) -> TestClient:
    client = TestClient(TestServer(api.build_app(s, TOKEN)))
    await client.start_server()
    return client


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bind_host": "0.0.0.0"},
        {"bind_host": "8.8.8.8"},
        {"allowed_clients": []},
        {"allowed_clients": ["0.0.0.0/0"]},
        {"llm_upstream": "http://10.0.0.5:8080"},
    ],
)
def test_settings_refuse_public_or_open_configuration(kwargs) -> None:
    with pytest.raises(ValidationError):
        settings(**kwargs)


def test_short_token_is_refused() -> None:
    with pytest.raises(ValueError):
        api.build_app(settings(), "short")


async def test_requests_need_the_token(stub_models) -> None:
    client = await client_for(settings())
    try:
        assert (await client.post("/v1/asr", data=b"\0\0")).status == 401
        bad = {"Authorization": "Bearer " + "x" * 32}
        assert (await client.post("/v1/asr", data=b"\0\0", headers=bad)).status == 401
        ok = await client.post(
            "/v1/asr?language=ur&prompt=%D9%82%D8%B3%D8%B7&no_speech_prob=0.4",
            data=b"\0\0" * 1600,
            headers=AUTH,
        )
        assert ok.status == 200 and (await ok.json())["text"] == "جی ہاں"
        assert stub_models[-1] == (3200, "ur", "قسط", 0.4)
    finally:
        await client.close()


async def test_clients_outside_the_allowlist_are_refused(stub_models) -> None:
    client = await client_for(settings(allowed_clients=["10.99.0.0/16"]))
    try:
        response = await client.post("/v1/asr", data=b"\0\0", headers=AUTH)
        assert response.status == 403
    finally:
        await client.close()


async def test_invalid_inputs_are_rejected(stub_models) -> None:
    client = await client_for(settings())
    try:
        odd = await client.post("/v1/asr", data=b"\0", headers=AUTH)
        assert odd.status == 400
        voice = await client.post(
            "/v1/tts", json={"text": "سلام", "voice": "../../etc"}, headers=AUTH
        )
        assert voice.status == 400
        long = await client.post("/v1/tts", json={"text": "ا" * 5000}, headers=AUTH)
        assert long.status == 400
    finally:
        await client.close()


async def test_tts_streams_wav_with_the_voice_rate(stub_models) -> None:
    client = await client_for(settings())
    try:
        response = await client.post(
            "/v1/tts", json={"text": "سلام۔ شکریہ۔"}, headers=AUTH
        )
        body = await response.read()
        assert response.status == 200 and body[:4] == b"RIFF"
        assert struct.unpack("<I", body[24:28])[0] == 22050
        assert len(body) > 44
    finally:
        await client.close()


@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8:18440",
        "http://vm.example.com:18440",
        "http://0.0.0.0:1",
        "https://10.20.30.111",
        "http://10.20.30.111:18440/x",
    ],
)
def test_remote_url_must_be_a_private_ip(url: str) -> None:
    with pytest.raises(ValidationError):
        LocalVoicePipelineSettings(inference="remote", remote_api_url=url)


def test_remote_mode_needs_a_token(monkeypatch) -> None:
    s = LocalVoicePipelineSettings(
        inference="remote", remote_api_url="http://10.20.30.111:18440"
    )
    assert s.llm_base_url == "http://10.20.30.111:18440/v1"
    monkeypatch.delenv("NRSP_INFERENCE_TOKEN", raising=False)
    with pytest.raises(ValueError):
        s.auth_headers()
    monkeypatch.setenv("NRSP_INFERENCE_TOKEN", TOKEN)
    assert s.auth_headers() == AUTH


async def test_remote_stt_sends_segment_and_prompt(stub_models) -> None:
    from pipecat.frames.frames import TranscriptionFrame
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.transcriptions.language import Language

    from nrsp_agent.voice.local_pipeline import RemoteWhisperSTTService

    client = await client_for(settings())
    try:
        url = str(client.make_url("")).rstrip("/")
        async with aiohttp.ClientSession(headers=AUTH) as session:
            stt = RemoteWhisperSTTService(
                api_url=url,
                session=session,
                timeout_s=5,
                settings=WhisperSTTService.Settings(
                    model="remote",
                    language=Language.UR,
                    no_speech_prob=0.4,
                    initial_prompt="ہندسے",
                ),
            )
            stt._handle_transcription = _noop  # type: ignore[method-assign]
            stt.start_processing_metrics = _noop  # type: ignore[method-assign]
            stt.stop_processing_metrics = _noop  # type: ignore[method-assign]
            frames = [f async for f in stt.run_stt(b"\0\0" * 800)]
        assert isinstance(frames[0], TranscriptionFrame)
        assert frames[0].text == "جی ہاں"
        assert stub_models[-1][2] == "ہندسے"
    finally:
        await client.close()


async def _noop(*_: object, **__: object) -> None:
    return None
