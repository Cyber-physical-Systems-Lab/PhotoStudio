#!/usr/bin/env bash
# Start the post-processing service (NOT a ROS node). It watches the portraits folder for finished
# captures (capture_complete.json) and writes <name>/<stamp>.jpg next to the raw shots.
# Usage: scripts/run_postprocess.sh [--once] [--root DIR] [--size 1200x1500]
cd "$(dirname "$0")/.."
# MediaPipe / TensorFlow Lite print harmless start-up notices; keep the log to our own lines.
export GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS="ignore::UserWarning"
PY="${PORTRAIT_POST_PYTHON:-$PWD/.venv/bin/python}"
"$PY" -c "import portrait_post" 2>/dev/null || "$PY" -m pip install -q -e postprocess --no-deps
exec "$PY" -m portrait_post.watch --root "${PORTRAIT_ROOT:-$HOME/Nova5Photo/portraits}" "$@"
