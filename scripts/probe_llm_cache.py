"""Measure where per-turn LLM prompt time goes on the local llama-server.

Replays the production request shape (system prompt, history, per-turn
schema) for a short scripted call and reports, per request, the server's own
prompt-token count and timings plus client time to first token. Used to tell
a fixed per-request cost (cache rewind) from a per-token cost.

    scripts/run_llm_server.sh &   # loopback :18431
    python scripts/probe_llm_cache.py
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

from nrsp_agent.agent import build_engine
from nrsp_agent.devtools.fixtures import NRSP_PROFILE, TEST_CALL_DATA
from nrsp_agent.voice.local_pipeline import LLM_EXTRA_BODY

CALLER = [
    "جی ہاں میں احمد علی بات کر رہا ہوں",
    "میرے شناختی کارڈ کے آخری چار ہندسے چار پانچ چھ سات",
    "جی درست ہے",
    "جی میں وقت پر ادائیگی کر دوں گا",
]


def stream(url: str, body: dict) -> tuple[str, float, float, dict]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first, text, timings = None, "", {}
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            timings = chunk.get("timings") or timings
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {}).get("content")
                if delta:
                    first = first or time.perf_counter()
                    text += delta
    end = time.perf_counter()
    return text, (first or end) - started, end - started, timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18431/v1")
    parser.add_argument("--model", default="gemma-4-E2B")
    parser.add_argument(
        "--prefill", action="store_true", help="prefill the known turn prefix first"
    )
    args = parser.parse_args()
    url = f"{args.url}/chat/completions"
    engine = build_engine(NRSP_PROFILE, TEST_CALL_DATA)
    greeting = engine.greeting()
    system = {"role": "system", "content": engine.system_prompt()}
    history = [
        {
            "role": "assistant",
            "content": json.dumps(
                {"action": "greeting", "value": "", "say": greeting.speak},
                ensure_ascii=False,
            ),
        }
    ]

    def body(messages: list[dict], max_tokens: int = 80) -> dict:
        return {
            "model": args.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": True,
            "reasoning_effort": "none",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "turn",
                    "schema": engine.response_schema(),
                    "strict": True,
                },
            },
            **LLM_EXTRA_BODY,
        }

    text, ttft, total, t = stream(
        url, body([system, {"role": "user", "content": "."}], 1)
    )
    print(f"warm      ttft={ttft:5.2f} total={total:5.2f} prompt_n={t.get('prompt_n')}")
    for caller in CALLER:
        if args.prefill:
            prefix = {"role": "user", "content": engine.turn_prefix(lambda s: s)}
            _, _, took, t = stream(url, body([system, *history, prefix], 1))
            print(f"prefill   total={took:5.2f} prompt_n={t.get('prompt_n')}")
        turn = {"role": "user", "content": engine.turn_message(caller, lambda s: s)}
        messages = [system, *history, turn]
        text, ttft, total, t = stream(url, body(messages))
        print(
            f"turn      ttft={ttft:5.2f} total={total:5.2f} "
            f"prompt_n={t.get('prompt_n')} "
            f"prompt_ms={t.get('prompt_ms', 0):.0f} gen_n={t.get('predicted_n')} "
            f"gen_ms={t.get('predicted_ms', 0):.0f} out={text[:70]}"
        )
        history += [turn, {"role": "assistant", "content": text}]
        try:
            out = json.loads(text)
            engine.on_model(out["action"], out.get("value", ""), caller)
        except (ValueError, KeyError):
            pass


if __name__ == "__main__":
    main()
