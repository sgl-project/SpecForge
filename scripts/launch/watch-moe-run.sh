#!/usr/bin/env bash
# Watcher for the restart run: milestones, checkpoints, errors, exit, plus a DISK GUARD that removes the
# restart run's OWN previous checkpoint before the next save when /personal free space < 320 GB
# (each save needs 292 GB and rotation deletes only after writing). Never touches other runs' checkpoints.
RUN=qwen3.8-27b-dspark-moe-regen-mixture-v1-cont2ep-restart-step5500-b
OUT=/personal/SpecForge-qwen38-moe/outputs/$RUN
LOG=$OUT/launch.log
TOTAL=4416; SAVE=500
last_step=0; err_lines=0; ckpts_seen=""
ts() { date -u +%m-%d\ %H:%M; }
freegb() { df -BG /personal | awk 'NR==2 {gsub("G","",$4); print $4}'; }
milestone() {
  line=$(grep -m1 -E "^step $1: " "$LOG")
  acc=$(echo "$line" | grep -oE "'train/acc': [0-9.e-]+" | grep -oE "[0-9.e-]+$")
  loss=$(echo "$line" | grep -oE "'train/loss': [0-9.e-]+" | grep -oE "[0-9.e-]+$")
  wall=$(echo "$line" | grep -oE "'train/perf/optimizer_step_time_s': [0-9.]+" | grep -oE "[0-9.]+$")
  peak=$(echo "$line" | grep -oE "'train/perf/mem_peak_alloc_gib': [0-9.]+" | grep -oE "[0-9.]+$")
  retr=$(echo "$line" | grep -oE "'train/perf/mem_alloc_retries': [0-9.]+" | grep -oE "[0-9.]+$")
  left=$(( (TOTAL - $1) * ${wall%.*} / 3600 ))
  printf "%s step %s/%s (overall %s/9916) acc=%.3f loss=%.3f wall=%.1fs peak=%.0fGB retries=%s disk_free=%sGB node0_free=%sGB eta~%sh\n" "$(ts)" "$1" "$TOTAL" "$((5500+$1))" "$acc" "$loss" "$wall" "$peak" "$retr" "$(freegb)" "$(awk '/MemFree/ {printf "%.0f", $4/1048576}' /sys/devices/system/node/node0/meminfo)" "$left"
}
while true; do
  if [ -f "$LOG" ]; then
    n=$(grep -c -E "Traceback|OutOfMemory|out of memory|Killed|NCCL (WARN|error)|Watchdog caught|timed out|status -800|port .* unavailable|Disk quota|No space left" "$LOG")
    if [ "$n" -gt "$err_lines" ]; then
      echo "$(ts) ERROR lines: $n (+$((n-err_lines))): $(grep -E "Traceback|OutOfMemory|out of memory|Killed|NCCL (WARN|error)|Watchdog caught|timed out|status -800|port .* unavailable|Disk quota|No space left" "$LOG" | tail -1 | cut -c1-200)"; err_lines=$n
    fi
    cur=$(grep -oE "^step [0-9]+:" "$LOG" | grep -oE "[0-9]+" | sort -n | tail -1); cur=${cur:-0}
    if [ "$cur" -ge 10 ] && [ "$last_step" -lt 10 ]; then milestone 10; last_step=10; fi
    next=$(( (last_step / SAVE + 1) * SAVE )); while [ "$cur" -ge "$next" ] && [ "$next" -le "$TOTAL" ]; do milestone "$next"; last_step=$next; next=$((next + SAVE)); done
    for d in "$OUT"/${RUN}-step*/; do
      [ -d "$d" ] || continue; b=$(basename "$d")
      case " $ckpts_seen " in *" $b "*) ;; *)
        if [ "$(ls "$d" | grep -c '\.tmp$')" = 0 ] && [ "$(ls "$d" | wc -l)" -ge 5 ]; then echo "$(ts) checkpoint $b complete ($(du -sh "$d" | cut -f1)); disk_free=$(freegb)GB"; ckpts_seen="$ckpts_seen $b"; fi;;
      esac
    done
    # DISK GUARD: within 60 steps of the next save and < 320 GB free -> remove this run's own older complete checkpoints
    nextsave=$(( (cur / SAVE + 1) * SAVE ))
    if [ $((nextsave - cur)) -le 60 ] && [ "$(freegb)" -lt 320 ]; then
      for d in $(ls -d "$OUT"/${RUN}-step*/ 2>/dev/null | sort -t p -k3 -n); do
        s=$(basename "$d" | grep -oE "[0-9]+$"); [ "$s" -lt "$nextsave" ] || continue
        [ "$(ls "$d" | grep -c '\.tmp$')" = 0 ] || continue
        rm -rf "$d" && echo "$(ts) DISK GUARD removed this run's checkpoint step$s before the step$nextsave save (free was <320 GB; now $(freegb)GB). Fallback remains cont2ep-from-1ep-v3-step5500."
      done
    fi
    if [ -f "$OUT/control/refs.jsonl.consumer_done" ] || [ "$cur" -ge "$TOTAL" ]; then echo "$(ts) RUN COMPLETE: last step $cur (overall $((5500+cur))); latest -> $(readlink "$OUT/${RUN}-latest" 2>/dev/null)"; exit 0; fi
  fi
  if [ -f "$LOG" ] && ! pgrep -f "specforge.cli tr[a]in -c .*$RUN" > /dev/null; then echo "$(ts) RUN EXITED (no supervisor). last step $cur. $(grep -E "Error|quota|space" "$LOG" | grep -v -i warn | tail -1 | cut -c1-200)"; exit 1; fi
  sleep 120
done
