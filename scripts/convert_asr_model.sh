#!/usr/bin/env bash
# Reproduce the selected ASR model: download khawajaaliarshad/whisper-small-urdu
# and convert it to CTranslate2 int8 for faster-whisper. Its repo has no
# tokenizer.json, so the unchanged openai/whisper-small tokenizer files are
# used (fine-tuning does not alter Whisper's vocabulary). Needs the bench venv
# (transformers + torch CPU): see PROGRESS.md.
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$PWD/.cache/huggingface}
PY=.cache/venv-bench/bin
SRC=data/models/hf/whisper-small-urdu
$PY/hf download khawajaaliarshad/whisper-small-urdu --local-dir "$SRC"
$PY/hf download openai/whisper-small tokenizer.json tokenizer_config.json vocab.json \
  merges.txt special_tokens_map.json added_tokens.json normalizer.json \
  preprocessor_config.json --local-dir "$SRC"
$PY/ct2-transformers-converter --model "$SRC" --output_dir data/models/ct2/whisper-small-urdu \
  --quantization int8 --copy_files tokenizer.json preprocessor_config.json --force
