#!/bin/bash
# Kill leftover processes holding the GPUs.
#
# CI jobs that time out or get cancelled can leave python processes behind which
# keep GPU memory allocated and make the next run OOM. This script is meant to be
# run before the test steps to make sure the GPUs are free.
#
# The container in .github/workflows/test.yaml runs with --privileged --pid=host,
# so processes started by previous jobs are visible and killable from here.
#
# Usage:
#   bash .github/workflows/scripts/kill_gpu_procs.sh [--dry-run] [--timeout SECONDS]

set -uo pipefail

DRY_RUN=0
# how long to wait for a SIGTERM'd process to exit before SIGKILL
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

# PIDs we must never kill: this script, and everything up its parent chain
# (the shell, the CI step, the runner itself).
protected_pids() {
  local pid=$$
  while [ -n "$pid" ] && [ "$pid" != "0" ]; do
    echo "$pid"
    pid=$(awk '{print $4}' "/proc/$pid/stat" 2>/dev/null)
  done
}

# Collect GPU PIDs from nvidia-smi. This is the authoritative source but it only
# reports compute apps, and inside some containers it reports nothing.
gpu_pids_from_nvidia_smi() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -d ' '
  # the process table of the plain output also lists graphics ("G") apps, which
  # --query-compute-apps does not report. Rows look like:
  #   |    0   N/A  N/A     12345      C   python                        51525MiB |
  nvidia-smi 2>/dev/null | awk '$1 == "|" && $3 == "N/A" && $5 ~ /^[0-9]+$/ && $6 ~ /^[CG]$/ {print $5}'
}

# Fallback: anything with an open file descriptor on an nvidia device node.
# Catches processes nvidia-smi misses (e.g. hung ones, or PID-namespace issues).
gpu_pids_from_fds() {
  local fd target pid
  for fd in /proc/[0-9]*/fd/*; do
    target=$(readlink "$fd" 2>/dev/null) || continue
    case "$target" in
      /dev/nvidia*|/dev/nvidiactl|/dev/nvidia-uvm*)
        pid=${fd#/proc/}
        echo "${pid%%/*}"
        ;;
    esac
  done
}

mapfile -t PROTECTED < <(protected_pids)

is_protected() {
  local pid=$1 p
  for p in "${PROTECTED[@]}"; do
    [ "$pid" = "$p" ] && return 0
  done
  return 1
}

proc_cmdline() {
  tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null | head -c 120
}

CANDIDATES=$( { gpu_pids_from_nvidia_smi; gpu_pids_from_fds; } | grep -E '^[0-9]+$' | sort -un )

TARGETS=()
for pid in $CANDIDATES; do
  # process may have exited between listing and now
  [ -d "/proc/$pid" ] || continue
  if is_protected "$pid"; then
    log "skipping protected pid $pid ($(proc_cmdline "$pid"))"
    continue
  fi
  TARGETS+=("$pid")
done

if [ ${#TARGETS[@]} -eq 0 ]; then
  log "no GPU processes found"
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

sleep 2

# Report what is left; do not fail the job on leftovers, the test step will
# surface the real problem if the GPUs are still busy.
REMAINING=$( { gpu_pids_from_nvidia_smi; gpu_pids_from_fds; } | grep -E '^[0-9]+$' | sort -un )
for pid in $REMAINING; do
  is_protected "$pid" && continue
  [ -d "/proc/$pid" ] && log "WARNING: pid $pid survived ($(proc_cmdline "$pid"))"
done

log "done, current GPU state:"
nvidia-smi || true
