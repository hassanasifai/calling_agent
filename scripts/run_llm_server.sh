#!/usr/bin/env bash
# Start the local LLM (llama.cpp, loopback only) for Ubuntu dev/benchmarks.
# Uses the official CPU build in .cache/llama.cpp (runtime AVX-512/AVX2
# kernels); the shared ~/.local/bin build is a generic OpenBLAS build that
# was measured 4.3x slower. Threads default to 6 (this VM's cgroup quota).
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=${LLAMA_SERVER:-.cache/llama.cpp/b11516/llama-b11516/llama-server}
MODEL=${LLM_GGUF:-data/models/gguf/gemma-4-E2B_q4_0-it.gguf}
exec nice -n "${LLM_NICE:-5}" "$BIN" -m "$MODEL" --alias gemma-4-E2B \
  --host 127.0.0.1 --port "${LLM_PORT:-18431}" -t "${LLM_THREADS:-6}" \
  -tb "${LLM_THREADS:-6}" -c 8192 -np 1 --jinja --no-webui
