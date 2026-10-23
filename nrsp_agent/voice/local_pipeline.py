"""Local Urdu voice pipeline assembled from Pipecat services."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import threading
import time
import tomllib
import wave
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlsplit

import aiohttp
import numpy as np
import pipecat.services.piper.tts as piper_tts
from faster_whisper import WhisperModel
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import (
    PipelineParams,
    PipelineWorker,
    ProcessorUnusablePolicy,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.ollama.llm import OLLamaLLMService
from pipecat.services.piper.tts import PiperHttpTTSService, PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_start.vad_user_turn_start_strategy import (
    VADUserTurnStartStrategy,
)
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import assert_given
from piper import PiperVoice
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nrsp_agent.agent.engine import CallEngine
from nrsp_agent.conversation.processors import (
    ConversationController,
    ConversationRecord,
    LLMOutputGuard,
    MetricsCollector,
)


class LocalVoicePipelineSettings(BaseModel):
    """Pinned local models and operational limits for one call pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asr_model: str = "small"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    asr_model_dir: Path = Path("data/models/faster-whisper")
    # CTranslate2 threads for ASR (the LLM is idle while ASR runs).
    asr_cpu_threads: int = Field(default=4, ge=1, le=64)
    tts_voice: str = "ur_PK-fasih-medium"
    tts_model_dir: Path = Path("data/models/piper")
    llm_model: str = "qwen3:1.7b"
    # Any loopback OpenAI-compatible server: Ollama (Windows) or llama.cpp's
    # llama-server (Ubuntu). Both cache the static system-prompt prefix.
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    # "remote": ASR, LLM and TTS run on the private inference API
    # (nrsp_agent.inference_api on the VM); this process loads no weights.
    inference: Literal["local", "remote"] = "local"
    # http://<private IP>:<port> of that API (VPN address); the bearer token
    # comes from the NRSP_INFERENCE_TOKEN environment variable.
    remote_api_url: str | None = None
    remote_timeout_seconds: float = Field(default=20.0, ge=2.0, le=120.0)
    llm_max_tokens: int = Field(default=80, ge=16, le=512)
    # Process the next turn's known prompt prefix while the bot speaks
    # (ConversationController._prefill_next_turn); any server that caches
    # prompt prefixes (llama.cpp, Ollama) benefits.
    llm_prefill: bool = True
    llm_temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    # Lexical fallback if the model has not produced a label by then.
    llm_understanding_timeout_seconds: float = Field(default=8.0, ge=1.0, le=30.0)
    # Profile action run when the caller stays silent after the task is done.
    finish_on_silence_action: str = "finish"
    # 0.3 s split Urdu phrases at comma pauses into fragments that Whisper
    # transcribes poorly and that each pay Whisper's fixed ~2 s encoder cost.
    vad_stop_secs: float = Field(default=0.5, ge=0.1, le=1.5)
    # Short answers ("جی ہاں", ~0.9 s) never reached SPEAKING at Pipecat's
    # 0.7 / 0.2 s defaults over G.711. Pipecat also resets Silero's state
    # every ~2 s of quiet; a reset that lands on a speech onset leaves the
    # model's confidence near 0.5 for the whole answer. Replay of 480 answers
    # after a previous turn and a 3-6 s gap (scripts/replay_vad.py): 0.5 /
    # 0.1 s missed 22, 0.4 / 0.064 s missed 8; noise false starts 0 -> 1/80
    # (babble only). 0.35 missed as many with more false starts.
    vad_confidence: float = Field(default=0.4, ge=0.2, le=0.95)
    vad_start_secs: float = Field(default=0.064, ge=0.03, le=0.5)
    user_speech_timeout_seconds: float = Field(default=0.6, ge=0.2, le=3.0)
    # Fallback that closes a turn when no transcript arrives. It must exceed
    # worst-case ASR latency, or a late transcript is attributed to the next
    # turn (measured: Whisper small took 8.6 s on the throttled demo laptop).
    user_turn_stop_timeout_seconds: float = Field(default=20.0, ge=5.0, le=60.0)
    user_idle_timeout_seconds: float = Field(default=8.0, ge=2.0, le=60.0)
    idle_timeout_seconds: float = Field(default=120.0, ge=15.0, le=900.0)

    @field_validator("ollama_base_url")
    @classmethod
    def ollama_must_be_local(cls, value: str) -> str:
        allowed = ("http://127.0.0.1:", "http://localhost:", "http://[::1]:")
        if not value.startswith(allowed):
            raise ValueError("Ollama endpoint must be loopback")
        return value.rstrip("/")

    @field_validator("remote_api_url")
    @classmethod
    def remote_must_be_private(cls, value: str | None) -> str | None:
        """Only an explicitly configured private IP literal (the VPN address of
        our VM): no hostnames, no public addresses, plain path-free URL."""
        if value is None:
            return value
        parts = urlsplit(value)
        if parts.scheme != "http" or parts.path not in ("", "/") or not parts.port:
            raise ValueError("remote_api_url must look like http://10.x.y.z:port")
        try:
            address = ipaddress.ip_address(parts.hostname or "")
        except ValueError as exc:
            raise ValueError("remote_api_url must use an IP address") from exc
        if not (address.is_private or address.is_loopback) or address.is_unspecified:
            raise ValueError("remote_api_url must be a private (VPN) address")
        return value.rstrip("/")

    @property
    def remote(self) -> bool:
        return self.inference == "remote"

    @property
    def llm_base_url(self) -> str:
        if self.remote:
            if not self.remote_api_url:
                raise ValueError("inference = remote needs remote_api_url")
            return f"{self.remote_api_url}/v1"
        return self.ollama_base_url

    def auth_headers(self) -> dict[str, str]:
        if not self.remote:
            return {}
        token = os.environ.get(REMOTE_TOKEN_ENV, "")
        if len(token) < 24:
            raise ValueError(f"{REMOTE_TOKEN_ENV} must hold the API token")
        return {"Authorization": f"Bearer {token}"}


REMOTE_TOKEN_ENV = "NRSP_INFERENCE_TOKEN"


# Model weights are loaded once per process and shared by every call's
# pipeline; CTranslate2 and ONNX Runtime sessions are safe for concurrent use.
_model_lock = threading.Lock()
_whisper_models: dict[tuple[str, str, str, str, int], WhisperModel] = {}
_piper_voices: dict[tuple[str, bool], PiperVoice] = {}


def shared_whisper_model(
    model: str, device: str, compute_type: str, download_root: Path, threads: int = 4
) -> WhisperModel:
    key = (model, device, compute_type, str(download_root), threads)
    with _model_lock:
        if key not in _whisper_models:
            _whisper_models[key] = WhisperModel(
                model,
                device=device,
                compute_type=compute_type,
                download_root=str(download_root),
                cpu_threads=threads,
            )
        return _whisper_models[key]


class _CachedPiperVoice:
    """Stand-in for ``PiperVoice`` inside Pipecat's Piper service module."""

    @staticmethod
    def load(model_path: str | Path, use_cuda: bool = False) -> PiperVoice:
        key = (str(Path(model_path).resolve()), use_cuda)
        with _model_lock:
            if key not in _piper_voices:
                _piper_voices[key] = PiperVoice.load(model_path, use_cuda=use_cuda)
            return _piper_voices[key]


piper_tts.PiperVoice = _CachedPiperVoice  # type: ignore[misc,assignment]


# Provider-specific request fields shared by every LLM request, including
# cache warm-ups: they change how the chat template renders the prompt, so a
# warm-up without them does not match (measured: 24.6 s cold first turn).
LLM_EXTRA_BODY = {
    "chat_template_kwargs": {"enable_thinking": False},
    "cache_prompt": True,
}


# One caller turn on the phone; Urdu runs ~2-3 tokens per word in Whisper.
WHISPER_MAX_TOKENS_PER_TURN = 96


def transcribe_pcm16(
    model: WhisperModel,
    pcm16: bytes,
    language: str,
    prompt: str | None,
    no_speech_prob: float | None,
) -> str:
    """One caller segment (16 kHz mono PCM16) -> text, with bounded decoding.

    Shared by the in-process STT service and the VM inference API so both
    paths decode identically."""
    samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        samples,
        language=language,
        initial_prompt=prompt,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
        max_new_tokens=WHISPER_MAX_TOKENS_PER_TURN,
    )
    return " ".join(
        s.text.strip()
        for s in segments
        if no_speech_prob is None or s.no_speech_prob < no_speech_prob
    ).strip()


class SharedWhisperSTTService(WhisperSTTService):
    """Whisper STT that reuses a process-wide model from the project cache."""

    def __init__(self, *, model_dir: Path, threads: int = 4, **kwargs: object) -> None:
        self._model_dir = model_dir
        self._threads = threads
        super().__init__(**kwargs)

    def _load(self) -> None:
        self._model = shared_whisper_model(
            str(self._settings.model),
            self._device,
            self._compute_type,
            self._model_dir,
            self._threads,
        )
        unsupported = self._unsupported_language()
        if unsupported:
            raise ValueError(unsupported)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame]:
        """Transcribe one VAD segment with bounded decoding.

        Pipecat's implementation uses faster-whisper defaults. On short
        telephone fragments Whisper was measured looping into repetition, and
        the default temperature fallback then re-decoded up to six times at up
        to 448 tokens each: about 40 s per segment on CPU. One greedy-to-beam
        pass at temperature 0 with a per-turn token cap bounds the cost.
        """
        if not self._model:
            yield ErrorFrame("Whisper model not available")
            return
        await self.start_processing_metrics()
        language = assert_given(self._settings.language)
        prompt = assert_given(self._settings.initial_prompt)
        threshold = assert_given(self._settings.no_speech_prob)
        text = await asyncio.to_thread(
            transcribe_pcm16, self._model, audio, str(language), prompt, threshold
        )
        await self.stop_processing_metrics()
        # Offline diagnostics only: NRSP_STT_DUMP_DIR keeps the exact segments
        # Whisper receives so ASR failures can be replayed. Never set it on a
        # customer call; the dump contains caller audio.
        if dump_dir := os.environ.get("NRSP_STT_DUMP_DIR"):
            await asyncio.to_thread(_dump_segment, Path(dump_dir), audio, [text])
        if text:
            await self._handle_transcription(text, True, language)
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), language)


class RemoteWhisperSTTService(WhisperSTTService):
    """Whisper STT whose decoding runs on the private inference API.

    Keeps WhisperSTTService's settings (language, initial_prompt,
    no_speech_prob), so the engine's per-turn ASR prompt switch works
    unchanged; segments come from the same Silero VAD as local mode."""

    def __init__(
        self,
        *,
        api_url: str,
        session: aiohttp.ClientSession,
        timeout_s: float,
        **kwargs: object,
    ) -> None:
        self._api_url = api_url
        self._session = session
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        super().__init__(**kwargs)

    def _load(self) -> None:  # no local weights
        unsupported = self._unsupported_language()
        if unsupported:
            raise ValueError(unsupported)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame]:
        await self.start_processing_metrics()
        language = assert_given(self._settings.language)
        prompt = assert_given(self._settings.initial_prompt)
        threshold = assert_given(self._settings.no_speech_prob)
        query = f"language={language}"
        if prompt:
            query += f"&prompt={quote(prompt)}"
        if threshold is not None:
            query += f"&no_speech_prob={threshold}"
        try:
            async with self._session.post(
                f"{self._api_url}/v1/asr?{query}",
                data=audio,
                headers={"Content-Type": "application/octet-stream"},
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                text = str((await response.json())["text"]).strip()
        except (aiohttp.ClientError, TimeoutError, KeyError, ValueError) as exc:
            # The turn falls through to the engine's silence/repeat prompts.
            logger.warning(f"remote ASR failed: {type(exc).__name__}: {exc}")
            text = ""
        await self.stop_processing_metrics()
        if text:
            await self._handle_transcription(text, True, language)
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), language)


def _dump_segment(directory: Path, pcm16: bytes, texts: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{time.time_ns()}"
    with wave.open(str(directory / f"{name}.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(pcm16)
    (directory / f"{name}.txt").write_text(" | ".join(texts), encoding="utf-8")


class EngineLLMService(OLLamaLLMService):
    """OpenAI-compatible LLM service whose output is constrained, per turn, to
    the engine's JSON schema (only the actions allowed at that moment).

    Works with llama.cpp's ``llama-server`` and Ollama, which both implement
    ``response_format`` JSON schemas with grammar-constrained decoding.
    """

    def __init__(self, engine: CallEngine, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._engine = engine

    def build_chat_completion_params(self, params_from_context: Any) -> dict:
        params = super().build_chat_completion_params(params_from_context)
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "turn",
                "schema": self._engine.response_schema(),
                "strict": True,
            },
        }
        return params


def preload_models(settings: LocalVoicePipelineSettings) -> None:
    """Load ASR/TTS weights before dialing so answer-to-greeting is fast."""
    if settings.remote:
        return  # weights live on the inference API
    shared_whisper_model(
        settings.asr_model,
        settings.asr_device,
        settings.asr_compute_type,
        settings.asr_model_dir,
        settings.asr_cpu_threads,
    )
    _CachedPiperVoice.load(settings.tts_model_dir / f"{settings.tts_voice}.onnx")


@dataclass(frozen=True)
class LocalVoiceAgent:
    """A constructed worker plus the controller that drives it."""

    worker: PipelineWorker
    controller: ConversationController
    record: ConversationRecord
    # Remote inference connection pool (None in local mode).
    http: aiohttp.ClientSession | None = None

    async def aclose(self) -> None:
        if self.http is not None and not self.http.closed:
            await self.http.close()

    async def queue_initial_greeting(self) -> None:
        await self.controller.greet()

    async def on_keypad(self, key: str) -> None:
        await self.controller.on_keypad(key)

    async def prewarm_llm(
        self, base_url: str, model: str, headers: dict[str, str] | None = None
    ) -> None:
        """Fill the LLM server's prefix cache with this call's system prompt
        while the phone rings, so the first caller turn pays only for its
        own tokens (measured ~87 prompt tokens/s on 4 CPU threads)."""
        body = {
            "model": model,
            # A user turn too: some chat templates (Qwen3.5) require one.
            "messages": [
                self.controller.system_message(),
                {"role": "user", "content": "."},
            ],
            "max_tokens": 1,
            "reasoning_effort": "none",
            **LLM_EXTRA_BODY,
        }
        await asyncio.to_thread(
            _post_json, f"{base_url}/chat/completions", body, headers
        )


def _post_json(url: str, body: dict, headers: dict[str, str] | None = None) -> None:
    import json
    import urllib.request

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            response.read()
    except OSError as exc:
        logger.warning(f"LLM prewarm failed: {exc}")


def load_local_voice_settings(path: Path) -> LocalVoicePipelineSettings:
    """Load a version-controlled deployment profile without reading secrets."""
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    return LocalVoicePipelineSettings.model_validate(document["voice"])


def build_local_voice_agent(
    transport: BaseTransport,
    settings: LocalVoicePipelineSettings,
    engine: CallEngine,
) -> LocalVoiceAgent:
    """Build one fully local conversation worker for an answered call.

    Task behaviour comes entirely from ``engine`` (agent profile + call data).
    """
    record = ConversationRecord()
    headers = settings.auth_headers()
    stt_settings = WhisperSTTService.Settings(
        model=settings.asr_model,
        language=Language.UR,
        no_speech_prob=0.4,
        initial_prompt=engine.profile.asr_prompts.get("default"),
    )
    voice = engine.profile.tts_voice or settings.tts_voice
    # Pipecat's 3 s default closes the audio context between slowly streamed
    # LLM sentences; three such contexts in a row mark TTS unusable, which
    # ends the call under ProcessorUnusablePolicy.END.
    tts_stop_timeout_s = 8.0
    http: aiohttp.ClientSession | None = None
    if settings.remote:
        assert settings.remote_api_url is not None
        # Bounded connection pool; per-request timeouts are set per call.
        http = aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=settings.remote_timeout_seconds),
            connector=aiohttp.TCPConnector(limit=4, keepalive_timeout=30),
        )
        stt: WhisperSTTService = RemoteWhisperSTTService(
            api_url=settings.remote_api_url,
            session=http,
            timeout_s=settings.remote_timeout_seconds,
            settings=stt_settings,
        )
        tts: Any = PiperHttpTTSService(
            base_url=f"{settings.remote_api_url}/v1/tts",
            aiohttp_session=http,
            stop_frame_timeout_s=tts_stop_timeout_s,
            settings=PiperHttpTTSService.Settings(voice=voice, language=Language.UR),
        )
    else:
        stt = SharedWhisperSTTService(
            model_dir=settings.asr_model_dir,
            threads=settings.asr_cpu_threads,
            device=settings.asr_device,
            compute_type=settings.asr_compute_type,
            settings=stt_settings,
        )
        tts = PiperTTSService(
            download_dir=settings.tts_model_dir,
            stop_frame_timeout_s=tts_stop_timeout_s,
            settings=PiperTTSService.Settings(voice=voice, language=Language.UR),
        )
    llm = EngineLLMService(
        engine,
        base_url=settings.llm_base_url,
        # Sent as "Authorization: Bearer" by the OpenAI client (remote API).
        **({"api_key": headers["Authorization"][7:]} if headers else {}),
        settings=OLLamaLLMService.Settings(
            model=settings.llm_model,
            # The controller sends the full message list per turn, starting
            # with the engine's system prompt.
            temperature=settings.llm_temperature,
            max_completion_tokens=settings.llm_max_tokens,
            # Qwen3 otherwise spends the token budget on hidden reasoning:
            # Ollama reads reasoning_effort, llama.cpp chat_template_kwargs.
            extra={
                "reasoning_effort": "none",
                "extra_body": LLM_EXTRA_BODY,
            },
        ),
    )
    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=16_000,
                params=VADParams(
                    confidence=settings.vad_confidence,
                    start_secs=settings.vad_start_secs,
                    stop_secs=settings.vad_stop_secs,
                ),
            ),
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[
                    SpeechTimeoutUserTurnStopStrategy(
                        user_speech_timeout=settings.user_speech_timeout_seconds
                    )
                ],
            ),
            user_turn_stop_timeout=settings.user_turn_stop_timeout_seconds,
            user_idle_timeout=settings.user_idle_timeout_seconds,
            # Unanswered empty turns fall through to the silence prompt.
            empty_user_turn=None,
        ),
    )

    async def prefill(messages: list[dict]) -> None:
        body = {
            "model": settings.llm_model,
            "messages": messages,
            "max_tokens": 1,
            "reasoning_effort": "none",
            **LLM_EXTRA_BODY,
        }
        url = f"{settings.llm_base_url}/chat/completions"
        await asyncio.to_thread(_post_json, url, body, headers)

    controller = ConversationController(
        engine,
        record,
        understanding_timeout_s=settings.llm_understanding_timeout_seconds,
        finish_on_silence=settings.finish_on_silence_action,
        prefill=prefill if settings.llm_prefill else None,
    )
    user_aggregator = aggregators.user()

    @user_aggregator.event_handler("on_user_turn_idle")
    async def on_user_turn_idle(*_: object) -> None:
        await controller.handle_silence()

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            controller,
            llm,
            LLMOutputGuard(record, controller),
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=16_000,
            audio_out_sample_rate=8_000,
            enable_metrics=True,
            enable_usage_metrics=False,
        ),
        observers=[MetricsCollector(record)],
        idle_timeout_secs=settings.idle_timeout_seconds,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    return LocalVoiceAgent(
        worker=worker, controller=controller, record=record, http=http
    )
