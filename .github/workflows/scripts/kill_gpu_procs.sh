#!/bin/bash
# Kill leftover compute processes holding the GPUs.
#
# CI jobs that time out or get cancelled can leave python processes behind which
# keep GPU memory allocated and make the next run OOM. This script is meant to be
# run before the test steps to make sure the GPUs are free.
#
# Only PIDs reported by `nvidia-smi --query-compute-apps` are targeted, so
# system daemons like nvidia-persistenced are left alone.
#
# The container in .github/workflows/test.yaml runs with --privileged --pid=host,
# so processes started by previous jobs are visible and killable from here.
#
# Usage:
#   bash .github/workflows/scripts/kill_gpu_procs.sh [--dry-run] [--timeout SECONDS]

set -uo pipefail

DRY_RUN=0
TERM_TIMEOUT=10

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --timeout)
      TERM_TIMEOUT="$2"
      shift 2
      ;;
    -h|--help)
      grep '^#' "$0" | cut -c 3-
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

log() {
  echo "[kill_gpu_procs] $*"
}

if ! command -v nvidia-smi > /dev/null 2>&1; then
  log "nvidia-smi not found, nothing to do"
  exit 0
fi

# Never kill this script or its parent chain (shell / CI step / runner).
is_protected() {
  local target=$1 pid=$$
  while [ -n "$pid" ] && [ "$pid" != "0" ]; do
    [ "$pid" = "$target" ] && return 0
    pid=$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null)
  done
  return 1
}

proc_cmdline() {
  tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | head -c 120
}

PIDS=$(
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | tr -d ' ' \
    | grep -E '^[0-9]+$' \
    | sort -un \
    || true
)

TARGETS=()
for pid in $PIDS; do
  [ -d "/proc/$pid" ] || continue
  if is_protected "$pid"; then
    log "skipping protected pid $pid ($(proc_cmdline "$pid"))"
    continue
  fi
  TARGETS+=("$pid")
done

if [ ${#TARGETS[@]} -eq 0 ]; then
  log "no GPU compute processes found"
  nvidia-smi || true
  exit 0
fi

log "found ${#TARGETS[@]} GPU process(es):"
for pid in "${TARGETS[@]}"; do
  log "  pid=$pid cmd=$(proc_cmdline "$pid")"
done

if [ "$DRY_RUN" -eq 1 ]; then
  log "dry run, not killing anything"
  exit 0
fi

log "sending SIGTERM"
for pid in "${TARGETS[@]}"; do
  kill -TERM "$pid" 2>/dev/null || true
done

waited=0
while [ "$waited" -lt "$TERM_TIMEOUT" ]; do
  alive=0
  for pid in "${TARGETS[@]}"; do
    [ -d "/proc/$pid" ] && alive=$((alive + 1))
  done
  [ "$alive" -eq 0 ] && break
  sleep 1
  waited=$((waited + 1))
done

for pid in "${TARGETS[@]}"; do
  if [ -d "/proc/$pid" ]; then
    log "pid $pid still alive after ${TERM_TIMEOUT}s, sending SIGKILL"
    kill -KILL "$pid" 2>/dev/null || true
  fi
done

log "done, current GPU state:"
nvidia-smi || true
