#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-}"
CAM="${2:-}"
WIDTH="${3:-1280}"
HEIGHT="${4:-720}"

if [[ -z "$PROFILE" ]]; then
  echo "Usage: $0 <profile> [camera_index] [width] [height]"
  echo "Examples:"
  echo "  $0 edwin 5"
  echo "  $0 edwin       # interactive pick"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV_DIR=".venv"
REQ_FILE="requirements.txt"
PY="${PY_BIN:-python3}"

say() { echo "[run_live] $*"; }

# 1) Create venv if missing
if [[ ! -d "$VENV_DIR" ]]; then
  say "Creating venv: $VENV_DIR"
  "$PY" -m venv "$VENV_DIR"
fi

# 2) Install deps if needed
if ! "$VENV_DIR/bin/python" -c "import cv2, mediapipe" >/dev/null 2>&1; then
  say "Installing requirements..."
  "$VENV_DIR/bin/python" -m pip install -U pip
  "$VENV_DIR/bin/python" -m pip install -r "$REQ_FILE"
fi

# 3) If camera index not given, show available ones and ask
if [[ -z "$CAM" ]]; then
  say "Probing cameras (0-12). Pick an index that shows OK:"
  "$VENV_DIR/bin/python" - <<'PY'
import cv2
for i in range(0, 13):
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    ok = cap.isOpened()
    if ok:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  {i}: OK ({w}x{h})")
    else:
        print(f"  {i}: no")
    cap.release()
print("\nTip: 0 is often OBS virtual camera; external USB cams are often 5/6.")
PY
  if command -v v4l2-ctl >/dev/null 2>&1; then
    echo
    say "v4l2 device list:"
    v4l2-ctl --list-devices || true
  fi
  echo
  read -r -p "Choose camera index [default 5]: " CAM_IN
  CAM="${CAM_IN:-5}"
fi

say "Starting live coach: profile=$PROFILE camera=$CAM ${WIDTH}x${HEIGHT}"
exec "$VENV_DIR/bin/python" -m bbcoach live --profile "$PROFILE" --camera-index "$CAM" --width "$WIDTH" --height "$HEIGHT"
