"""Local Urdu voice pipeline assembled from Pipecat services."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
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
from pydantic import BaseModel, ConfigDict, Field, field_validator

URDU_SYSTEM_PROMPT = """آپ NRSP کے مجاز فون معاون ہیں۔ صرف اردو میں مختصر، واضح
اور شائستہ جملوں میں بات کریں۔ پہلی گفتگو میں اپنا تعارف کروائیں،
بتائیں کہ آپ ایک AI معاون ہیں،
اور تصدیق کریں کہ آپ درست شخص سے بات کر رہے ہیں۔ کبھی پاس ورڈ، پن، OTP یا مکمل شناختی
کارڈ نمبر نہ مانگیں۔ غیر یقینی معلومات ایجاد نہ کریں؛ ضرورت پر انسانی نمائندے کی پیشکش
کریں۔ جواب بول کر سنایا جائے گا، اس لیے markdown، ایموجی اور لمبی فہرستیں استعمال
نہ کریں۔"""


class LocalVoicePipelineSettings(BaseModel):
    """Pinned local models and operational limits for one call pipeline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asr_model: str = "small"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    tts_voice: str = "ur_PK-fasih-medium"
    tts_model_dir: Path = Path("data/models/piper")
    llm_model: str = "qwen3:1.7b"
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    idle_timeout_seconds: float = Field(default=120.0, ge=15.0, le=900.0)
    system_prompt: str = URDU_SYSTEM_PROMPT

    @field_validator("ollama_base_url")
    @classmethod
    def ollama_must_be_local(cls, value: str) -> str:
        allowed = ("http://127.0.0.1:", "http://localhost:", "http://[::1]:")
        if not value.startswith(allowed):
            raise ValueError("Ollama endpoint must be loopback")
        return value.rstrip("/")


@dataclass(frozen=True)
class LocalVoiceAgent:
    """A constructed worker plus the explicit outbound greeting trigger."""

    worker: PipelineWorker

    async def queue_initial_greeting(self) -> None:
        await self.worker.queue_frame(LLMRunFrame())


def load_local_voice_settings(path: Path) -> LocalVoicePipelineSettings:
    """Load a version-controlled deployment profile without reading secrets."""
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    return LocalVoicePipelineSettings.model_validate(document["voice"])


def build_local_voice_agent(
    transport: BaseTransport,
    settings: LocalVoicePipelineSettings,
) -> LocalVoiceAgent:
    """Build one fully local Urdu conversation worker for an answered call."""
    stt = WhisperSTTService(
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
        settings=WhisperSTTService.Settings(
            model=settings.asr_model,
            language=Language.UR,
            no_speech_prob=0.4,
            initial_prompt="NRSP، قرض، قسط، شناخت، تصدیق، نمائندہ",
        ),
    )
    llm = OLLamaLLMService(
        base_url=settings.ollama_base_url,
        settings=OLLamaLLMService.Settings(model=settings.llm_model),
    )
    tts = PiperTTSService(
        download_dir=settings.tts_model_dir,
        settings=PiperTTSService.Settings(
            voice=settings.tts_voice,
            language=Language.UR,
        ),
    )

    context = LLMContext(
        messages=[{"role": "system", "content": settings.system_prompt}]
    )
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(sample_rate=16_000),
            filter_incomplete_user_turns=True,
        ),
    )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
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
        idle_timeout_secs=settings.idle_timeout_seconds,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    return LocalVoiceAgent(worker=worker)
