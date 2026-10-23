#!/usr/bin/env bash
# start | stop | restart | status | logs | health | wait  (user services only)
set -euo pipefail
cd "$(dirname "$0")/../.."
UNITS=(calling-llm.service calling-api.service)
conf=config/local/vm-api.toml
host=$(sed -n 's/^bind_host *= *"\(.*\)"/\1/p' "$conf" 2>/dev/null)
port=$(sed -n 's/^port *= *\([0-9]*\)/\1/p' "$conf" 2>/dev/null)
health() {
  # shellcheck disable=SC1091
  set -a; . config/local/vm-api.env; set +a
  # The allowlist normally excludes the VM itself, so this checks the LLM
  # directly and the API's listening socket; the laptop checks /health.
  curl -sf -m 3 http://127.0.0.1:18431/health >/dev/null && echo "llm: ok" || echo "llm: DOWN"
  ss -ltn | grep -q "${host}:${port} " && echo "api: listening on ${host}:${port}" || echo "api: NOT listening"
  # Authenticated end-to-end check (works when this VM is an allowed client).
  curl -s -m 5 -H "Authorization: Bearer ${NRSP_INFERENCE_TOKEN}" \
    "http://${host}:${port}/health" && echo
}
case ${1:-status} in
  start) systemctl --user start "${UNITS[@]}" ;;
  stop) systemctl --user stop calling-api.service calling-llm.service ;;
  restart) systemctl --user restart "${UNITS[@]}" ;;
  status)
    for u in "${UNITS[@]}"; do
      printf '%-22s %s / %s\n' "$u" "$(systemctl --user is-active "$u")" "$(systemctl --user is-enabled "$u")"
    done
    systemctl --user show calling-ai.slice -p MemoryCurrent -p CPUUsageNSec | sed 's/^/  /'
    health ;;
  logs) journalctl --user -u calling-api.service -u calling-llm.service -n "${2:-50}" --no-pager ;;
  health) health ;;
  wait)
    for _ in $(seq 120); do
      if ss -ltn | grep -q "${host}:${port} " && curl -sf -m 2 http://127.0.0.1:18431/health >/dev/null; then
        echo "ready"; exit 0
      fi
      sleep 2
    done
    echo "not ready after 240 s; see: scripts/vm/ctl.sh logs" >&2; exit 1 ;;
  *) echo "usage: $0 start|stop|restart|status|logs [n]|health|wait" >&2; exit 2 ;;
esac
