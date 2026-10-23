#!/usr/bin/env bash
# One-time, idempotent setup of the calling agent's inference API on this VM.
# User-level only: no sudo, no system services, no reboot. Re-running keeps the
# existing token and local config.
#
#   scripts/vm/setup.sh [CLIENT_CIDR]     e.g. scripts/vm/setup.sh 25.1.0.114/32
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO=$(pwd)
LOCAL=config/local
UNIT_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
mkdir -p "$LOCAL" "$UNIT_DIR"
chmod 700 "$LOCAL"

# Prerequisites that already exist from development (nothing is downloaded).
for path in .venv/bin/python data/models/ct2/whisper-small-urdu \
    data/models/piper/ur_PK-aegis_female-medium.onnx \
    data/models/gguf/gemma-4-E2B_q4_0-it.gguf \
    .cache/llama.cpp/b11516/llama-b11516/llama-server; do
  [[ -e $path ]] || { echo "missing: $path" >&2; exit 1; }
done

if [[ ! -f $LOCAL/vm-api.toml ]]; then
  client=${1:-}
  [[ -n $client ]] || { echo "first run: pass the laptop's VPN address/CIDR" >&2; exit 1; }
  sed "s#__CLIENT__#${client}#" config/vm-api.example.toml > "$LOCAL/vm-api.toml"
  echo "created $LOCAL/vm-api.toml (allowed client: $client)"
fi
if [[ ! -f $LOCAL/vm-api.env ]]; then
  umask 077
  echo "NRSP_INFERENCE_TOKEN=$(head -c 32 /dev/urandom | base64 | tr -d '/+=' | head -c 40)" \
    > "$LOCAL/vm-api.env"
  echo "created $LOCAL/vm-api.env (token; copy it to the laptop's config\\local)"
fi
chmod 600 "$LOCAL/vm-api.env"

for unit in calling-ai.slice calling-llm.service calling-api.service; do
  sed "s#__REPO__#${REPO}#g" "deploy/systemd/$unit" > "$UNIT_DIR/$unit"
done
systemctl --user daemon-reload
systemctl --user enable calling-llm.service calling-api.service >/dev/null
systemctl --user restart calling-llm.service calling-api.service

if [[ $(loginctl show-user "$USER" -p Linger --value) != yes ]]; then
  echo "WARNING: linger is off; services stop at logout and do not start at boot." >&2
  echo "         'loginctl enable-linger $USER' fixes it (no reboot needed)." >&2
fi
echo "waiting for the API (model loading takes ~30 s on first start)..."
scripts/vm/ctl.sh wait
scripts/vm/ctl.sh status
