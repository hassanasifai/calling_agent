"""Benchmark LLM turn understanding + reply with the production engine.

Starts a private llama.cpp server (loopback, one model, capped threads), puts
the NRSP profile's engine in the stage each labelled turn needs, sends the
exact production messages with the profile's JSON schema, and scores the
chosen action, reply safety (production guard) and latency.

    .venv/bin/python scripts/benchmark_llm_turns.py data/models/gguf/X.gguf \
        --set tests/fixtures/intent/dev.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np

from nrsp_agent.agent import build_engine
from nrsp_agent.agent.engine import AFFIRM, CallEngine
from nrsp_agent.conversation.processors import screen_sentence
from nrsp_agent.devtools.loopback import NRSP_PROFILE, TEST_CALL_DATA

LLAMA = Path(".cache/llama.cpp/b11516/llama-b11516/llama-server")


def engine_at(stage: str) -> CallEngine:
    engine = build_engine(NRSP_PROFILE, TEST_CALL_DATA)
    engine.greeting()
    if stage in ("verify", "discuss"):
        engine.on_model("confirm_identity", "", "جی")
    if stage == "discuss":
        engine.on_model("verify_cnic", "", "چار پانچ چھ سات")
        engine.on_model(AFFIRM, "", "جی")
    return engine


def start_server(binary: Path, model: Path, port: int, threads: int):
    proc = subprocess.Popen(
        [
            str(binary),
            "-m",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-t",
            str(threads),
            "-tb",
            str(threads),
            "-c",
            "4096",
            "-np",
            "1",
            "--jinja",
            "--no-webui",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(240):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                return proc
        except OSError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError("llama-server did not start")


def stream(port: int, messages: list[dict], schema: dict, max_tokens: int = 120):
    body = {
        "messages": messages,
        "stream": True,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "turn", "schema": schema},
        },
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    text, first, action_at = "", None, None
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            delta = json.loads(line[5:])["choices"][0]["delta"].get("content") or ""
            if delta and first is None:
                first = time.perf_counter() - started
            text += delta
            if action_at is None and '"value"' in text:
                action_at = time.perf_counter() - started
    total = time.perf_counter() - started
    return text, first or total, action_at or total, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "--set", type=Path, default=Path("tests/fixtures/intent/dev.jsonl")
    )
    parser.add_argument("--server", type=Path, default=LLAMA)
    parser.add_argument("--port", type=int, default=18431)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("data/eval/results"))
    args = parser.parse_args()
    items = [json.loads(x) for x in args.set.read_text("utf-8").splitlines() if x]
    t0 = time.perf_counter()
    server = start_server(args.server, args.model, args.port, args.threads)
    load_s = time.perf_counter() - t0
    rows = []
    try:
        for item in items:
            engine = engine_at(item["stage"])
            system = {"role": "system", "content": engine.system_prompt()}
            schema = engine.response_schema()
            # Warm the prefix cache as the live call does while ringing.
            stream(args.port, [system, {"role": "user", "content": "."}], schema, 1)
            messages = [
                system,
                {"role": "user", "content": engine.turn_message(item["text"], str)},
            ]
            text, ttft, action_s, total = stream(args.port, messages, schema)
            try:
                out = json.loads(text)
            except ValueError:
                out = {"action": "INVALID", "value": "", "say": ""}
            say = out.get("say", "").strip()
            reply_needed = out["action"] in (
                "reply",
                "record_promise",
                "record_difficulty",
            )
            profile = engine.profile
            rows.append(
                {
                    **item,
                    "got": out["action"],
                    "value": out.get("value", ""),
                    "say": say,
                    "ttft": ttft,
                    "action_s": action_s,
                    "total": total,
                    "ok": out["action"] == item["action"],
                    "reply_ok": (not reply_needed)
                    or screen_sentence(
                        say, profile.reply_forbidden_terms, profile.reply_allow_digits
                    )
                    is not None,
                }
            )
            r = rows[-1]
            print(
                f"{'OK ' if r['ok'] else 'BAD'} {item['action']:>17} -> "
                f"{r['got']:<17} {action_s:.2f}s {total:.2f}s | {item['text']} | "
                f"{say[:90]}",
                flush=True,
            )
    finally:
        server.terminate()
        server.wait(timeout=30)
    terminal = ("opt_out", "human_handoff", "callback", "wrong_person", "finish")
    summary = {
        "model": args.model.name,
        "set": args.set.name,
        "threads": args.threads,
        "load_s": round(load_s, 2),
        "n": len(rows),
        "action_acc": round(sum(r["ok"] for r in rows) / len(rows), 3),
        "reply_guard_pass": f"{sum(r['reply_ok'] for r in rows)}/{len(rows)}",
        "false_terminal": sum(r["got"] in terminal and not r["ok"] for r in rows),
        "missed_terminal": sum(r["action"] in terminal and not r["ok"] for r in rows),
    }
    for key in ("ttft", "action_s", "total"):
        values = [r[key] for r in rows]
        summary[f"{key}_p50"] = round(float(np.percentile(values, 50)), 3)
        summary[f"{key}_p95"] = round(float(np.percentile(values, 95)), 3)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"llm_{args.model.stem}_{args.set.stem}.json").write_text(
        json.dumps({**summary, "rows": rows}, ensure_ascii=False, indent=1), "utf-8"
    )
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
