"""Measure one local Ollama Urdu response without a telephone call."""

from __future__ import annotations

import argparse
import json
import sys
import time
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:1.7b")
    parser.add_argument("--url", default="http://127.0.0.1:11434/api/chat")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    payload = {
        "model": args.model,
        "stream": False,
        "think": False,
        "messages": [
            {
                "role": "system",
                "content": "صرف مختصر اور شائستہ اردو میں جواب دیں۔",
            },
            {
                "role": "user",
                "content": (
                    "اپنا تعارف ایک جملے میں کروائیں اور پوچھیں "
                    "کیا میں بات کر سکتا ہوں۔"
                ),
            },
        ],
        "options": {"num_predict": 80, "temperature": 0.2},
    }
    request = Request(
        args.url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urlopen(request, timeout=180) as response:  # noqa: S310 - loopback URL
        result = json.load(response)
    elapsed = time.perf_counter() - started
    generated_tokens = int(result.get("eval_count", 0))
    generation_ns = int(result.get("eval_duration", 0))
    tokens_per_second = (
        generated_tokens / (generation_ns / 1_000_000_000) if generation_ns else 0
    )
    print(
        json.dumps(
            {
                "model": result.get("model", args.model),
                "response": result["message"]["content"],
                "wall_seconds": round(elapsed, 3),
                "load_seconds": round(int(result.get("load_duration", 0)) / 1e9, 3),
                "prompt_tokens": int(result.get("prompt_eval_count", 0)),
                "generated_tokens": generated_tokens,
                "tokens_per_second": round(tokens_per_second, 2),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
