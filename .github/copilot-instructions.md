# BB Coach — Copilot Instructions

## Project Overview
**BB Coach** is a Python-based webcam bodybuilding assistant that uses MediaPipe pose detection and body segmentation to score posing form, track progress, and provide live feedback. The app supports federation rulesets (WNBF UK, PCA), multiple pose categories, and persistent session/profile management.

**Tech Stack:** Python 3, OpenCV, MediaPipe Solutions (<0.10.30), Pydantic, NumPy, Rich CLI.

## Architecture & Data Flow

### Core Modules
- **`app.py`** — Main live loop ([`run_live()`](bbcoach/app.py#L182)): frame capture → pose detection → scoring → UI overlay → auto-capture
- **`profile.py`** — User data models (Pydantic): `UserProfile`, `CompetitionPlan`, `PrepPhase`, `ContestResult`; persisted as JSON
- **`poses/library.py`** — Pose definitions: target angles + tolerances per pose (e.g., `mp_front`, `mp_back`); category→routine mappings
- **`metrics/pose_features.py`** — Geometry computation: shoulder level, torso upright, elbow symmetry from normalized landmarks
- **`poses/scoring.py`** — Scoring logic: compares computed features against pose targets with tolerance bands; supports user-captured templates
- **`vision/pose.py`** — MediaPipe wrapper: landmark extraction + segmentation mask
- **`storage/session.py`** — Capture storage: JSON snapshots + JPEG frames + auto-indexed directories by date/profile
- **`federations/library.py`** — Federation rules and category displays

### Critical Data Flow
```
Frame (BGR) → PoseBackend.process_bgr()
  ├→ Landmarks (dict: name→(x,y)) + Mask (segmentation)
  ├→ compute_features(landmarks) → feature dict (shoulder_level, torso_upright, elbow_sym, etc.)
  ├→ score_pose(features, target, tolerance) → PoseScore(0-100, per_feature, ok_flags)
  └→ compute_from_mask(mask) → Proportions (silhouette ratios)
  
Auto-capture (if stable + high score):
  └→ SessionStore saves: full frame + cutout + story frame → index.json
```

### Coordinate Systems
- **MediaPipe landmarks:** Normalized [0, 1] coords (x=horizontal, y=vertical downward in image space)
- **Feature angles:** Degrees; torso upright uses vertical axis (0, -1)
- **UI overlay:** Raw pixel coordinates on display frame

## Key Patterns & Conventions

### Template Personalization
When user presses **Space**, current pose landmarks + features are captured as a "template" in `profile.templates[pose_key]`. On next scoring of that pose, template features override generic pose targets—enables personalized baselines.

### Pose Definitions
- Each pose in `POSES` dict specifies: `target` (ideal feature values), `tolerance` (allowed deviation), `guidance` (on-screen tips)
- Routines (`ROUTINES` dict) group poses by category; pose sequence cycles with N/P keys
- Generic pose guides available in `POSE_GUIDES` (visual reference overlays)

### Auto-Capture Thresholds
Configured in `AutoCaptureConfig`:
- `stable_frames`: motion buffer size (check avg motion < threshold)
- `min_score`: minimum pose score to capture
- `cooldown_frames`: prevent capture spam
- `settle_frames`: wait after pose switch before capturing
- `top_k`: keep best N captures per pose, discard older ones

### Profile Lifecycle
1. Create: `python -m bbcoach profile new --name <name>`
2. Edit interactively: `python -m bbcoach profile edit --name <name>`
3. Load in live mode: `python -m bbcoach live --profile <name> --camera <idx>`
4. Modifications during live session saved on quit
5. View: `python -m bbcoach profile show --name <name>`

## Common Tasks

### Adding a New Pose
1. Add `PoseDef` entry to `POSES` dict in [poses/library.py](bbcoach/poses/library.py)
   - `key`: internal identifier
   - `target`: dict of feature names → target angles/values
   - `tolerance`: dict of feature names → acceptable deviation
   - `guidance`: list of on-screen tips
2. Add pose to a `ROUTINES[category]` list
3. Optionally add visual guide to `POSE_GUIDES` for overlay

### Adjusting Scoring Logic
- **Feature computation:** Modify angle calculations in [metrics/pose_features.py](bbcoach/metrics/pose_features.py)
- **Scoring formula:** Edit score calculation in [poses/scoring.py](bbcoach/poses/scoring.py) (currently: linear 0-100 based on feature error vs tolerance)
- **Override templates:** Scoring automatically uses captured templates; no code change needed

### UI Overlays & Rendering
- Pose skeleton drawn by `draw_pose_overlay()` in [vision/overlay.py](bbcoach/vision/overlay.py)
- Pose guides (reference postures) drawn by `draw_pose_guide()`
- Body segmentation outline added by `draw_mask_outline()`
- Auto-gen story frames for social sharing (resized, annotated frames)

### MediaPipe Dependency Warning
The project pins `mediapipe<0.10.30` because newer versions removed the Solutions API (`mp.solutions.pose.*`). If dependency errors occur, [vision/pose.py](bbcoach/vision/pose.py#L18) includes error messaging—recommend reinstall per instructions.

## Testing & Debugging

### Quick Validation
- Run live with test profile: `python -m bbcoach live --profile edwin --camera 0 --width 1280 --height 720`
- Press **I** (show info toggle) to debug: display frame index, motion buffer, capture status
- Check session captures: `sessions/<profile>_YYYYMMDD.json` (snapshots) and `sessions/captures/<profile>/YYYYMMDD/` (images + index.json)

### Camera Setup
- Use [run_live.sh](run_live.sh) to auto-probe available cameras (0–12)
- Pass Linux device path or numeric index: `--camera /dev/video2` or `--camera 5`

## File Organization Reference
- **Config:** `config/profiles/*.json` (user profiles)
- **Data:** `sessions/*.json` (snapshots), `sessions/captures/<profile>/<date>/*.jpg` (auto-captures)
- **Pose templates:** Embedded in profile JSON under `templates` key
- **Docs:** [docs/SOURCES.md](docs/SOURCES.md) (references for pose definitions, body composition data)
