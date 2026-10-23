"""Private inference API for remote call orchestration (runs on the VM).

One small aiohttp process exposes the models the call pipeline already uses,
so a Windows laptop can run only SIP/RTP, Pipecat orchestration and VAD:

    GET  /health                 liveness + which models are loaded
    POST /v1/asr                 raw PCM16 16 kHz mono -> {"text": ...}
    POST /v1/tts                 {"text", "voice"} -> streamed WAV (Piper)
    POST /v1/chat/completions    proxied to the loopback llama-server
    GET  /v1/models              proxied to the loopback llama-server

Every request needs ``Authorization: Bearer <token>`` and must come from an
allowed client network; the server binds only to the configured private
address. ASR/TTS reuse the exact model loading and decoding of the local
pipeline. Settings come from a TOML file plus the token from the environment
(``NRSP_INFERENCE_TOKEN``), never from Git.

    python -m nrsp_agent.inference_api --config config/local/vm-api.toml
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import ipaddress
import os
import struct
import time
import tomllib
from collections.abc import Awaitable, Callable
from pathlib import Path

import aiohttp
from aiohttp import web
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nrsp_agent.voice.local_pipeline import (
    _CachedPiperVoice,
    shared_whisper_model,
    transcribe_pcm16,
)

ASR_SAMPLE_RATE = 16_000
TOKEN_ENV = "NRSP_INFERENCE_TOKEN"


class InferenceApiSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    # Private address of the VM's VPN-reachable interface; never 0.0.0.0.
    bind_host: str
    port: int = Field(default=18440, ge=1024, le=65535)
    # Client networks allowed to call the API (e.g. the laptop's VPN address).
    allowed_clients: list[str]
    llm_upstream: str = "http://127.0.0.1:18431"
    asr_model: str = "data/models/ct2/whisper-small-urdu"
    asr_compute_type: str = "int8"
    asr_cpu_threads: int = Field(default=6, ge=1, le=64)
    asr_model_dir: Path = Path("data/models/faster-whisper")
    tts_model_dir: Path = Path("data/models/piper")
    tts_voices: list[str] = ["ur_PK-aegis_female-medium"]
    max_asr_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    max_tts_chars: int = Field(default=1000, ge=10, le=10_000)
    max_llm_body_bytes: int = Field(default=512_000, ge=1_000)
    llm_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)

    @field_validator("bind_host")
    @classmethod
    def private_bind(cls, value: str) -> str:
        address = ipaddress.ip_address(value)
        if address.is_unspecified or not (address.is_private or address.is_loopback):
            raise ValueError("bind_host must be a private or loopback address")
        return value

    @field_validator("allowed_clients")
    @classmethod
    def valid_networks(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("allowed_clients must list at least one network")
        for network in value:
            net = ipaddress.ip_network(network, strict=False)
            if net.prefixlen == 0:
                raise ValueError("allowed_clients must not allow every address")
        return value

    @field_validator("llm_upstream")
    @classmethod
    def loopback_upstream(cls, value: str) -> str:
        if not value.startswith(("http://127.0.0.1:", "http://localhost:")):
            raise ValueError("llm_upstream must be the loopback llama-server")
        return value.rstrip("/")


def load_settings(path: Path) -> InferenceApiSettings:
    return InferenceApiSettings.model_validate(
        tomllib.loads(path.read_text(encoding="utf-8"))["api"]
    )


def _wav_stream_header(sample_rate: int) -> bytes:
    """44-byte PCM16 mono WAV header with unknown (streamed) length."""
    unknown = 0xFFFFFFFF
    return (
        b"RIFF"
        + struct.pack("<I", unknown)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", unknown)
    )


Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


def build_app(settings: InferenceApiSettings, token: str) -> web.Application:
    if len(token) < 24:
        raise ValueError(f"{TOKEN_ENV} must be at least 24 characters")
    networks = [ipaddress.ip_network(n, strict=False) for n in settings.allowed_clients]
    expected = f"Bearer {token}".encode()
    # One ASR and one TTS job at a time: each already uses several CPU
    # threads, and the VM is shared with Financial AI workloads.
    asr_lock, tts_lock = asyncio.Lock(), asyncio.Lock()
    state: dict[str, object] = {}

    @web.middleware
    async def guard(request: web.Request, handler: Handler) -> web.StreamResponse:
        peer = request.remote or ""
        try:
            allowed = any(ipaddress.ip_address(peer) in n for n in networks)
        except ValueError:
            allowed = False
        if not allowed:
            logger.warning(f"rejected client {peer}")
            raise web.HTTPForbidden(text="client not allowed")
        supplied = request.headers.get("Authorization", "").encode()
        if not hmac.compare_digest(supplied, expected):
            raise web.HTTPUnauthorized(text="missing or invalid token")
        return await handler(request)

    async def health(_: web.Request) -> web.Response:
        llm_ok = False
        session: aiohttp.ClientSession = state["session"]  # type: ignore[assignment]
        try:
            async with session.get(
                f"{settings.llm_upstream}/health",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as response:
                llm_ok = response.status == 200
        except (aiohttp.ClientError, TimeoutError):
            pass
        return web.json_response(
            {
                "status": "ok" if llm_ok else "degraded",
                "asr": settings.asr_model,
                "tts_voices": settings.tts_voices,
                "llm": llm_ok,
            }
        )

    async def asr(request: web.Request) -> web.Response:
        pcm = await request.read()
        if not pcm or len(pcm) % 2:
            raise web.HTTPBadRequest(text="body must be PCM16 mono 16 kHz")
        language = request.query.get("language", "ur")
        if not language.isalpha() or len(language) > 8:
            raise web.HTTPBadRequest(text="invalid language")
        prompt = request.query.get("prompt") or None
        if prompt and len(prompt) > 500:
            raise web.HTTPBadRequest(text="prompt too long")
        try:
            threshold = float(request.query["no_speech_prob"])
        except KeyError:
            threshold = None
        except ValueError as exc:
            raise web.HTTPBadRequest(text="invalid no_speech_prob") from exc
        model = state["whisper"]
        started = time.perf_counter()
        async with asr_lock:
            text = await asyncio.to_thread(
                transcribe_pcm16, model, pcm, language, prompt, threshold  # type: ignore[arg-type]
            )
        seconds = round(time.perf_counter() - started, 3)
        audio_s = len(pcm) / 2 / ASR_SAMPLE_RATE
        logger.info(f"asr audio={audio_s:.2f}s took={seconds}s")
        return web.json_response({"text": text, "seconds": seconds})

    async def tts(request: web.Request) -> web.StreamResponse:
        try:
            body = await request.json()
            text, voice = str(body["text"]), str(body.get("voice") or "")
        except (ValueError, KeyError, TypeError) as exc:
            raise web.HTTPBadRequest(text='JSON {"text", "voice"} required') from exc
        voice = voice or settings.tts_voices[0]
        if voice not in settings.tts_voices:
            raise web.HTTPBadRequest(text="voice not available")
        if not text.strip() or len(text) > settings.max_tts_chars:
            raise web.HTTPBadRequest(text="text empty or too long")
        piper = _CachedPiperVoice.load(settings.tts_model_dir / f"{voice}.onnx")
        response = web.StreamResponse(headers={"Content-Type": "audio/wav"})
        await response.prepare(request)
        await response.write(_wav_stream_header(piper.config.sample_rate))
        started, first = time.perf_counter(), None
        async with tts_lock:
            chunks = iter(piper.synthesize(text))
            while True:
                chunk = await asyncio.to_thread(next, chunks, None)
                if chunk is None:
                    break
                first = first or time.perf_counter() - started
                await response.write(chunk.audio_int16_bytes)
        await response.write_eof()
        logger.info(f"tts chars={len(text)} first_audio={first or 0:.3f}s")
        return response

    async def llm_proxy(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        session: aiohttp.ClientSession = state["session"]  # type: ignore[assignment]
        headers = {
            "Content-Type": request.headers.get("Content-Type", "application/json")
        }
        try:
            async with session.request(
                request.method,
                f"{settings.llm_upstream}{request.path}",
                data=body or None,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=settings.llm_timeout_seconds),
            ) as upstream:
                response = web.StreamResponse(
                    status=upstream.status,
                    headers={
                        "Content-Type": upstream.headers.get(
                            "Content-Type", "application/json"
                        )
                    },
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (aiohttp.ClientError, TimeoutError) as exc:
            logger.warning(f"llm upstream error: {exc}")
            raise web.HTTPBadGateway(text="llm unavailable") from exc

    async def on_startup(_: web.Application) -> None:
        state["session"] = aiohttp.ClientSession()
        # Load weights before accepting work so the first call is not slow.
        state["whisper"] = await asyncio.to_thread(
            shared_whisper_model,
            settings.asr_model,
            "cpu",
            settings.asr_compute_type,
            settings.asr_model_dir,
            settings.asr_cpu_threads,
        )
        for voice in settings.tts_voices:
            await asyncio.to_thread(
                _CachedPiperVoice.load, settings.tts_model_dir / f"{voice}.onnx"
            )
        logger.info("inference API models loaded")

    async def on_cleanup(_: web.Application) -> None:
        await state["session"].close()  # type: ignore[union-attr]

    max_asr_bytes = int(settings.max_asr_seconds * ASR_SAMPLE_RATE * 2)
    app = web.Application(
        middlewares=[guard],
        client_max_size=max(max_asr_bytes, settings.max_llm_body_bytes),
    )
    app.router.add_get("/health", health)
    app.router.add_post("/v1/asr", asr)
    app.router.add_post("/v1/tts", tts)
    app.router.add_post("/v1/chat/completions", llm_proxy)
    app.router.add_get("/v1/models", llm_proxy)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/local/vm-api.toml"))
    args = parser.parse_args()
    settings = load_settings(args.config)
    token = os.environ.get(TOKEN_ENV, "")
    app = build_app(settings, token)
    web.run_app(
        app,
        host=settings.bind_host,
        port=settings.port,
        access_log=None,
        shutdown_timeout=5,
    )


if __name__ == "__main__":
    main()
