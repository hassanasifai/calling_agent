"""Multi-turn text dialogues through the production controller, guard and
policy with a real local LLM (llama.cpp server), no audio.

Checks per dialogue: final outcome, verification, that no financial fact is
spoken before verification, and the time until the first spoken text.

    .venv/bin/python scripts/simulate_dialogues.py data/models/gguf/X.gguf
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from pipecat.frames.frames import (
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.frame_processor import FrameDirection

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_llm_turns import LLAMA, start_server  # noqa: E402

from nrsp_agent.agent import build_engine  # noqa: E402
from nrsp_agent.conversation.processors import (  # noqa: E402
    ConversationController,
    ConversationRecord,
    LLMOutputGuard,
)
from nrsp_agent.devtools.fixtures import NRSP_PROFILE, TEST_CALL_DATA  # noqa: E402

# Unseen phrasings; none is used in any prompt. (caller turns, outcome, verified)
DIALOGUES: list[tuple[str, list[str], str, bool]] = [
    (
        "who_then_full_flow",
        [
            "kaun bol raha hai?",
            "جی جی، میں ہی احمد ہوں",
            "char paanch chay saat",
            "جی ہاں درست",
            "قسط کہاں جمع کرانی ہوتی ہے؟",
            "ٹھیک ہے، میں اگلے ہفتے جمع کروا دوں گا",
            "نہیں بس اتنا ہی",
        ],
        "completed",
        True,
    ),
    (
        "refuses_twice_then_hangs_up",
        [
            "جی بولیں",
            "میں فون پر کوئی نمبر نہیں بتاتا",
            "نہیں، میں نہیں بتاؤں گا",
            "بس رہنے دیں، اللہ حافظ",
        ],
        "caller_ended",
        False,
    ),
    (
        "corrects_digits_then_hardship_then_human",
        [
            "yes",
            "چار پانچ چھ آٹھ",
            "نہیں نہیں، آخری سات ہے",
            "چار پانچ چھ سات",
            "جی",
            "اس بار سیلاب کی وجہ سے سب کچھ تباہ ہو گیا، پیسے نہیں ہیں",
            "مجھے کسی بندے سے بات کروا دیں",
            "جی ہاں",
        ],
        "handoff_requested",
        True,
    ),
    (
        "busy_callback",
        ["ابھی میں نماز پڑھنے جا رہا ہوں، بعد میں کال کرنا"],
        "callback_requested",
        False,
    ),
    (
        "not_him",
        ["nahi ji, ye unka number nahi hai", "جی، وہ یہاں نہیں ہوتے"],
        "wrong_person",
        False,
    ),
    (
        "asks_amount_topic_change_bye",
        [
            "جی فرمائیں",
            "four five six seven",
            "yes correct",
            "kitne paise dene hain?",
            "اچھا یہ بتائیں آج موسم کیسا ہے؟",
            "اچھا ٹھیک ہے، کر دوں گا۔ اللہ حافظ",
        ],
        "completed",
        True,
    ),
    (
        "robot_question_then_opt_out",
        [
            "جی میں بات کر رہا ہوں",
            "چار پانچ چھ سات",
            "ہاں",
            "آپ کوئی روبوٹ ہیں کیا؟",
            "اچھا، آئندہ مجھے ایسی کالز نہ کیا کریں",
        ],
        "opted_out",
        True,
    ),
    (
        "english_caller",
        [
            "Yes, this is Ahmed speaking",
            "four five six seven",
            "yes that's right",
            "When is it due?",
            "Okay I'll pay on time, thank you, bye",
        ],
        "completed",
        True,
    ),
]


class Port:
    """Streams the controller's LLM requests into the guard via llama-server."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.spoken: list[str] = []
        self.first_audio_s: float | None = None
        self.turn_started = 0.0
        self.timings: list[dict] = []

    def attach(self, controller: ConversationController, guard: LLMOutputGuard):
        async def controller_push(frame, direction=FrameDirection.DOWNSTREAM):
            if isinstance(frame, LLMContextFrame):
                await self.run_llm(guard, frame.context.get_messages())
            elif isinstance(frame, TTSSpeakFrame):
                self.say(frame.text)

        async def guard_push(frame, direction=FrameDirection.DOWNSTREAM):
            if isinstance(frame, LLMTextFrame):
                self.say(frame.text.strip())

        controller.push_frame = controller_push
        guard.push_frame = guard_push

    def say(self, text: str) -> None:
        if self.first_audio_s is None:
            self.first_audio_s = time.perf_counter() - self.turn_started
        self.spoken.append(text)

    async def run_llm(
        self, guard: LLMOutputGuard, messages: list[dict], warm: bool = False
    ) -> None:
        schema = guard._controller.engine.response_schema()
        import urllib.request

        body = {
            "messages": messages,
            "stream": True,
            "temperature": 0.2,
            "max_tokens": 80,
            "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "turn", "schema": schema},
            },
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        down = FrameDirection.DOWNSTREAM
        response = await asyncio.to_thread(urllib.request.urlopen, request, None, 180)
        if warm:
            await asyncio.to_thread(response.read)
            return
        await guard.process_frame(LLMFullResponseStartFrame(), down)
        while raw := await asyncio.to_thread(response.readline):
            line = raw.decode().strip()
            if line.startswith("data:") and not line.endswith("[DONE]"):
                chunk = json.loads(line[5:])
                if "timings" in chunk:
                    t = chunk["timings"]
                    self.timings.append(
                        {
                            k: round(t.get(k, 0))
                            for k in (
                                "cache_n",
                                "prompt_n",
                                "prompt_ms",
                                "predicted_n",
                                "predicted_ms",
                            )
                        }
                    )
                if not chunk.get("choices"):
                    continue
                delta = chunk["choices"][0]["delta"]
                if text := delta.get("content"):
                    await guard.process_frame(LLMTextFrame(text=text), down)
        await guard.process_frame(LLMFullResponseEndFrame(), down)


async def run_dialogue(port: int, turns: list[str]) -> dict:
    engine = build_engine(NRSP_PROFILE, TEST_CALL_DATA)
    record = ConversationRecord()
    controller = ConversationController(
        engine, record, understanding_timeout_s=30, finish_on_silence="finish"
    )
    guard = LLMOutputGuard(record, controller)
    io = Port(port)
    io.attach(controller, guard)
    await controller.greet()
    await io.run_llm(
        guard,
        [controller.system_message(), {"role": "user", "content": "."}],
        warm=True,
    )
    transcript, latencies, leaked = [], [], False
    secrets = (TEST_CALL_DATA["amount_spoken"], TEST_CALL_DATA["due_date_spoken"])
    for text in turns:
        if engine.closed:
            break
        io.spoken, io.first_audio_s = [], None
        io.turn_started = time.perf_counter()
        from pipecat.processors.aggregators.llm_context import LLMContext

        context = LLMContext(messages=[{"role": "user", "content": text}])
        was_verified = "verified" in engine.flags
        await controller.process_frame(
            LLMContextFrame(context=context), FrameDirection.DOWNSTREAM
        )
        reply = " ".join(io.spoken)
        if not was_verified and "verified" not in engine.flags:
            leaked |= any(s in reply for s in secrets)
        latencies.append(round(io.first_audio_s or 0.0, 2))
        transcript.append({"caller": text, "bot": reply, "first_text_s": latencies[-1]})
    if not engine.closed:
        # A real call goes quiet after the caller's last words.
        io.spoken = []
        await controller.handle_silence()
        transcript.append(
            {"caller": "<silence>", "bot": " ".join(io.spoken), "first_text_s": 0.0}
        )
    await controller.cleanup()
    return {
        "outcome": engine.outcome,
        "verified": "verified" in engine.flags,
        "result": engine.result(),
        "leaked_before_verification": leaked,
        "labels": [e for _, e in record.events if e.startswith(("action", "model"))],
        "guard_rejections": record.guard_rejections,
        "turns": transcript,
        "llm_timings": io.timings,
        "first_text_s": latencies,
    }


async def main_async(args: argparse.Namespace) -> None:
    server = start_server(args.server, args.model, args.port, args.threads)
    results, passed = {}, 0
    try:
        for name, turns, outcome, verified in DIALOGUES:
            result = await run_dialogue(args.port, turns)
            ok = (
                result["outcome"] == outcome
                and result["verified"] == verified
                and not result["leaked_before_verification"]
            )
            passed += ok
            results[name] = {"ok": ok, "expected": outcome, **result}
            print(
                f"{'PASS' if ok else 'FAIL'} {name}: {result['outcome']} "
                f"verified={result['verified']} labels={result['labels']} "
                f"latency={result['first_text_s']}",
                flush=True,
            )
            print(f"    llm timings: {result['llm_timings']}")
            for turn in result["turns"]:
                print(f"    caller: {turn['caller']}\n    bot:    {turn['bot']}")
    finally:
        server.terminate()
        server.wait(timeout=30)
    all_lat = [x for r in results.values() for x in r["first_text_s"]]
    all_lat.sort()
    summary = {
        "model": args.model.name,
        "passed": f"{passed}/{len(DIALOGUES)}",
        "first_text_p50": all_lat[len(all_lat) // 2],
        "first_text_max": all_lat[-1],
    }
    print(json.dumps(summary))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"dialogues_{args.model.stem}.json").write_text(
        json.dumps({**summary, "dialogues": results}, ensure_ascii=False, indent=1),
        "utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--server", type=Path, default=LLAMA)
    parser.add_argument("--port", type=int, default=18431)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("data/eval/results"))
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
