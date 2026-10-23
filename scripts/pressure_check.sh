#!/usr/bin/env bash
# Refuse heavy work when the shared VM is under pressure (Financial AI first).
# Exit 1 if MemAvailable < MIN_AVAIL_GB, swap is actively paging, or memory PSI
# is elevated. Prints a one-line summary either way.
set -u
MIN_AVAIL_GB=${MIN_AVAIL_GB:-12}
avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
avail_gb=$((avail_kb / 1024 / 1024))
psi_mem=$(awk '/^some/{split($2,a,"=");print a[2]}' /proc/pressure/memory)
psi_cpu=$(awk '/^some/{split($2,a,"=");print a[2]}' /proc/pressure/cpu)
read -r si so < <(vmstat 1 2 | tail -1 | awk '{print $7, $8}')
load=$(cut -d' ' -f1 /proc/loadavg)
echo "avail=${avail_gb}G psi_mem10=${psi_mem} psi_cpu10=${psi_cpu} swap_si=${si} swap_so=${so} load1=${load}"
if (( avail_gb < MIN_AVAIL_GB )) || (( si > 100 || so > 100 )) \
   || awk -v p="$psi_mem" -v c="$psi_cpu" -v mc="${MAX_PSI_CPU:-35}" 'BEGIN{exit !(p > 5 || c > mc)}'; then
  echo "PRESSURE: unsafe for heavy work" >&2
  exit 1
fi
