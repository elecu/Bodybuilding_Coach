#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-}"
MODE="desktop"
WIDTH="1280"
HEIGHT="720"

if [[ -n "${2:-}" ]]; then
  case "${2:-}" in
    live|desktop)
      MODE="${2}"
      WIDTH="${3:-1280}"
      HEIGHT="${4:-720}"
      ;;
    *)
      WIDTH="${2:-1280}"
      HEIGHT="${3:-720}"
      ;;
  esac
fi

if [[ -z "$PROFILE" ]]; then
  echo "Usage: $0 <profile> [live|desktop] [width] [height]"
  echo "Examples:"
  echo "  $0 edwin"
  echo "  $0 edwin 1600 900"
  echo "  $0 edwin desktop"
  echo "  $0 edwin live"
  echo "  $0 edwin desktop 1600 900"
  echo "  $0 edwin live 1280 720"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VENV_DIR=".venv"
REQ_FILE="requirements.txt"
PY="${PY_BIN:-python3}"

say() { echo "[run_live] $*"; }

# Kinect stability default (can be overridden externally):
#   BBCOACH_KINECT_PIPELINE=opengl ./run_live.sh <profile>
: "${BBCOACH_KINECT_PIPELINE:=auto}"
export BBCOACH_KINECT_PIPELINE
# Legacy Kinect behavior (same style as previous working flow):
# depth stream ON by default.
: "${BBCOACH_KINECT_ENABLE_DEPTH:=1}"
export BBCOACH_KINECT_ENABLE_DEPTH
if [[ "$BBCOACH_KINECT_ENABLE_DEPTH" == "1" || "$BBCOACH_KINECT_ENABLE_DEPTH" == "true" || "$BBCOACH_KINECT_ENABLE_DEPTH" == "yes" || "$BBCOACH_KINECT_ENABLE_DEPTH" == "on" ]]; then
  say "Kinect stream mode: RGB+Depth"
else
  say "Kinect stream mode: RGB-only safe mode (depth off)"
fi
# Keep isolation OFF by default to preserve legacy in-process Kinect logic.
: "${BBCOACH_KINECT_ISOLATE:=0}"
export BBCOACH_KINECT_ISOLATE
if [[ "$BBCOACH_KINECT_ISOLATE" == "1" || "$BBCOACH_KINECT_ISOLATE" == "true" || "$BBCOACH_KINECT_ISOLATE" == "yes" || "$BBCOACH_KINECT_ISOLATE" == "on" ]]; then
  say "Kinect isolation mode: ON (worker process)"
else
  say "Kinect isolation mode: OFF (legacy in-process)"
fi

# 1) Create venv if missing.
if [[ ! -d "$VENV_DIR" ]]; then
  say "Creating venv: $VENV_DIR"
  "$PY" -m venv "$VENV_DIR"
fi

# 2) Install base deps if needed.
# Desktop mode starts offline and does not need mediapipe at boot.
if [[ "$MODE" == "desktop" ]]; then
  if ! "$VENV_DIR/bin/python" -c "import cv2, tkinter" >/dev/null 2>&1; then
    say "Installing requirements..."
    "$VENV_DIR/bin/python" -m pip install -U pip
    "$VENV_DIR/bin/python" -m pip install -r "$REQ_FILE"
  fi
else
  if ! "$VENV_DIR/bin/python" -c "import cv2, mediapipe, tkinter" >/dev/null 2>&1; then
    say "Installing requirements..."
    "$VENV_DIR/bin/python" -m pip install -U pip
    "$VENV_DIR/bin/python" -m pip install -r "$REQ_FILE"
  fi
fi

# 3) Desktop mode uses AthleteProfile store and keeps active profile in sync.
if [[ "$MODE" == "desktop" ]]; then
set +e
ACTIVE_PROFILE="$(
"$VENV_DIR/bin/python" - "$PROFILE" <<'PY'
import sys
import json
from pathlib import Path
from bbcoach.athlete_profiles import AthleteProfileStore

requested = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
store = AthleteProfileStore.default()
profiles = store.list_profiles()

match = None
for p in profiles:
    if p.profile_name.strip().lower() == requested:
        match = p
        break

if match is None:
    repo_root = Path.cwd()
    legacy_path = repo_root / "config" / "profiles" / f"{sys.argv[1]}.json"
    legacy_name = sys.argv[1]
    competition_date = None
    height_cm = 175.0
    weight_kg = 80.0
    sex_category = "Male"
    if legacy_path.exists():
        try:
            payload = json.loads(legacy_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                legacy_name = str(payload.get("name", legacy_name)).strip() or legacy_name
                plan = payload.get("plan") if isinstance(payload.get("plan"), dict) else {}
                comp_raw = str(plan.get("competition_date", "")).strip()
                if comp_raw:
                    competition_date = comp_raw
                bodyweight_log = payload.get("bodyweight_log")
                if isinstance(bodyweight_log, list) and bodyweight_log:
                    last = bodyweight_log[-1]
                    if isinstance(last, dict) and last.get("weight_kg") is not None:
                        try:
                            weight_cm_raw = float(last.get("weight_kg"))
                            if weight_cm_raw > 0:
                                weight_kg = weight_cm_raw
                        except Exception:
                            pass
        except Exception:
            pass
    # Default DOB chosen only to satisfy required athlete profile field.
    # User can edit it from Profile menu in desktop UI.
    match = store.create(
        profile_name=legacy_name,
        date_of_birth="1990-01-01",
        sex_category=sex_category,
        height_cm=height_cm,
        weight_kg=weight_kg,
        competition_date=competition_date,
    )

store.set_active_profile_id(match.profile_id)
print(match.profile_name)
PY
)"
PROFILE_SET_STATUS=$?
set -e

if [[ "$PROFILE_SET_STATUS" -ne 0 ]]; then
  exit "$PROFILE_SET_STATUS"
fi

say "Active profile: $ACTIVE_PROFILE"
say "Opening desktop app in OFFLINE mode (camera/Kinect power is controlled from top menu)."

# 4) Launch desktop UI. Always starts with source OFF.
# Default: foreground (recommended so window lifecycle is explicit).
# Optional background mode: DESKTOP_BG=1 ./run_live.sh <profile>
if [[ "${DESKTOP_BG:-0}" != "1" ]]; then
  exec "$VENV_DIR/bin/python" -m bbcoach desktop --source none --width "$WIDTH" --height "$HEIGHT"
fi

LOG_DIR="sessions/logs"
mkdir -p "$LOG_DIR"
DESKTOP_LOG="$LOG_DIR/desktop.log"
nohup "$VENV_DIR/bin/python" -m bbcoach desktop --source none --width "$WIDTH" --height "$HEIGHT" >"$DESKTOP_LOG" 2>&1 &
DESKTOP_PID=$!
for _ in 1 2 3; do
  sleep 1
  if ! kill -0 "$DESKTOP_PID" >/dev/null 2>&1; then
    say "Desktop app failed to start. Showing log tail:"
    tail -n 80 "$DESKTOP_LOG" || true
    exit 1
  fi
done

say "Desktop app started (pid=$DESKTOP_PID)."
say "Logs: $DESKTOP_LOG"
exit 0
fi

# 3) Live mode keeps the feature-complete coaching window:
# pose-by-pose flow, top captures per pose, voice controls, scan, metrics, PDF, PCD.
SOURCE_KIND="${LIVE_SOURCE:-v4l2}"
CAM_INDEX="${CAM_INDEX:-}"

if [[ "$SOURCE_KIND" == "v4l2" && -z "$CAM_INDEX" ]]; then
  AUTO_CAM="$(
    "$VENV_DIR/bin/python" - <<'PY'
import glob
import time
import cv2

for node in sorted(glob.glob("/dev/video*")):
    suffix = node.replace("/dev/video", "")
    if not suffix.isdigit():
        continue
    idx = int(suffix)
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        continue
    ok = False
    for _ in range(5):
        ret, frame = cap.read()
        if ret and frame is not None:
            ok = True
            break
        time.sleep(0.05)
    cap.release()
    if ok:
        print(idx)
        raise SystemExit(0)
print("")
PY
  )"
  if [[ -n "$AUTO_CAM" ]]; then
    CAM_INDEX="$AUTO_CAM"
    say "Auto-selected webcam index: $CAM_INDEX"
  else
    CAM_INDEX="0"
    say "Could not auto-detect a working webcam index. Falling back to CAM_INDEX=$CAM_INDEX"
  fi
fi

if [[ -z "$CAM_INDEX" ]]; then
  CAM_INDEX="0"
fi

say "Starting live coach with full training features: profile=$PROFILE source=$SOURCE_KIND ${WIDTH}x${HEIGHT}"
exec "$VENV_DIR/bin/python" -m bbcoach live \
  --profile "$PROFILE" \
  --source "$SOURCE_KIND" \
  --cam-index "$CAM_INDEX" \
  --width "$WIDTH" \
  --height "$HEIGHT"
