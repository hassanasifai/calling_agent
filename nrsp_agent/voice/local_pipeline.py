"""Local Urdu voice pipeline assembled from Pipecat services."""

from __future__ import annotations

import asyncio
import os
import threading
import time
import tomllib
import wave
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from pipecat.services.piper.tts import PiperTTSService
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

from nrsp_agent.conversation import script
from nrsp_agent.conversation.policy import ConversationPolicy
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
    tts_voice: str = "ur_PK-fasih-medium"
    tts_model_dir: Path = Path("data/models/piper")
    llm_model: str = "qwen3:1.7b"
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    llm_max_tokens: int = Field(default=80, ge=16, le=512)
    llm_temperature: float = Field(default=0.3, ge=0.0, le=1.5)
    # 0.3 s split Urdu phrases at comma pauses into fragments that Whisper
    # transcribes poorly and that each pay Whisper's fixed ~2 s encoder cost.
    vad_stop_secs: float = Field(default=0.5, ge=0.1, le=1.5)
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


# Model weights are loaded once per process and shared by every call's
# pipeline; CTranslate2 and ONNX Runtime sessions are safe for concurrent use.
_model_lock = threading.Lock()
_whisper_models: dict[tuple[str, str, str, str], WhisperModel] = {}
_piper_voices: dict[tuple[str, bool], PiperVoice] = {}


def shared_whisper_model(
    model: str, device: str, compute_type: str, download_root: Path
) -> WhisperModel:
    key = (model, device, compute_type, str(download_root))
    with _model_lock:
        if key not in _whisper_models:
            _whisper_models[key] = WhisperModel(
                model,
                device=device,
                compute_type=compute_type,
                download_root=str(download_root),
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


# One caller turn on the phone; Urdu runs ~2-3 tokens per word in Whisper.
WHISPER_MAX_TOKENS_PER_TURN = 96


class SharedWhisperSTTService(WhisperSTTService):
    """Whisper STT that reuses a process-wide model from the project cache."""

    def __init__(self, *, model_dir: Path, **kwargs: object) -> None:
        self._model_dir = model_dir
        super().__init__(**kwargs)

    def _load(self) -> None:
        self._model = shared_whisper_model(
            str(self._settings.model), self._device, self._compute_type, self._model_dir
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
        samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        language = assert_given(self._settings.language)
        prompt = assert_given(self._settings.initial_prompt)
        threshold = assert_given(self._settings.no_speech_prob)
        model = self._model

        def transcribe() -> list[Any]:
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
            return list(segments)

        segments = await asyncio.to_thread(transcribe)
        text = " ".join(
            s.text.strip()
            for s in segments
            if threshold is None or s.no_speech_prob < threshold
        ).strip()
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


def _dump_segment(directory: Path, pcm16: bytes, texts: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{time.time_ns()}"
    with wave.open(str(directory / f"{name}.wav"), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(pcm16)
    (directory / f"{name}.txt").write_text(" | ".join(texts), encoding="utf-8")


def preload_models(settings: LocalVoicePipelineSettings) -> None:
    """Load ASR/TTS weights before dialing so answer-to-greeting is fast."""
    shared_whisper_model(
        settings.asr_model,
        settings.asr_device,
        settings.asr_compute_type,
        settings.asr_model_dir,
    )
    _CachedPiperVoice.load(settings.tts_model_dir / f"{settings.tts_voice}.onnx")


@dataclass(frozen=True)
class LocalVoiceAgent:
    """A constructed worker plus the policy controller that drives it."""

    worker: PipelineWorker
    controller: ConversationController
    record: ConversationRecord

    async def queue_initial_greeting(self) -> None:
        await self.controller.greet()


def load_local_voice_settings(path: Path) -> LocalVoicePipelineSettings:
    """Load a version-controlled deployment profile without reading secrets."""
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    return LocalVoicePipelineSettings.model_validate(document["voice"])


def build_local_voice_agent(
    transport: BaseTransport,
    settings: LocalVoicePipelineSettings,
    policy: ConversationPolicy,
) -> LocalVoiceAgent:
    """Build one fully local Urdu conversation worker for an answered call."""
    record = ConversationRecord()
    stt = SharedWhisperSTTService(
        model_dir=settings.asr_model_dir,
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
        settings=WhisperSTTService.Settings(
            model=settings.asr_model,
            language=Language.UR,
            no_speech_prob=0.4,
            initial_prompt=script.ASR_PROMPT_DOMAIN,
        ),
    )
    llm = OLLamaLLMService(
        base_url=settings.ollama_base_url,
        settings=OLLamaLLMService.Settings(
            model=settings.llm_model,
            system_instruction=script.DISCUSS_INSTRUCTION,
            temperature=settings.llm_temperature,
            max_completion_tokens=settings.llm_max_tokens,
            # Over the OpenAI-compatible API Qwen3 otherwise spends the whole
            # token budget on hidden reasoning and returns no spoken content.
            extra={"reasoning_effort": "none"},
        ),
    )
    tts = PiperTTSService(
        download_dir=settings.tts_model_dir,
        # Pipecat's 3 s default closes the audio context between slowly
        # streamed LLM sentences; three such contexts in a row mark TTS
        # unusable, which ends the call under ProcessorUnusablePolicy.END.
        stop_frame_timeout_s=8.0,
        settings=PiperTTSService.Settings(
            voice=settings.tts_voice,
            language=Language.UR,
        ),
    )

    context = LLMContext()
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(
                sample_rate=16_000,
                params=VADParams(stop_secs=settings.vad_stop_secs),
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
    controller = ConversationController(policy, record)
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
            LLMOutputGuard(record),
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
    return LocalVoiceAgent(worker=worker, controller=controller, record=record)
