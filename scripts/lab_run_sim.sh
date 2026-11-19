#!/usr/bin/env bash
# Start a full simulated session in the background on the lab machine and log to $OUT/launch.log.
# Usage: scripts/lab_run_sim.sh [OUT_DIR] [extra launch args...]
set -e
OUT="${1:-/tmp/n5run}"; shift || true
cd "$(dirname "$0")/.."
source /opt/ros/humble/setup.bash
source install/setup.bash
rm -rf "$OUT"; mkdir -p "$OUT"
NAME="${SIM_NAME-Sim Tester}"   # SIM_NAME= (empty) starts no session automatically
NAME_ARG=(); [ -n "$NAME" ] && NAME_ARG=("name:=$NAME")
setsid nohup ros2 launch portrait_bringup portrait.launch.py headless:=true record:="$OUT" \
  "${NAME_ARG[@]}" voice:=none isaac_args:="--exit-after-session" "$@" > "$OUT/launch.log" 2>&1 < /dev/null &
echo "launched, log: $OUT/launch.log"
