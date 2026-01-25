# BB Coach (Webcam) — Fedora 40

A local, privacy-friendly webcam-based bodybuilding assistant:

- **Progress tracking** (repeatable proportions + silhouette ratios)
- **Pose coach (live)** with on-screen **red/green** guidance
- Optional **federation rulesets**: **WNBF UK** or **PCA**
- Optional **First Timers** mode
- Optional **competition countdown**, and a simple evidence-based prep planner

> This is a *training aid*. It is not medical advice.

## Quick start (Fedora 40)

```bash
cd bb_webcam_coach
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Start live coach
python -m bbcoach live --profile edwin --camera-index 0 --width 1280 --height 720
```

## Key controls (live window)

- **Q**: quit
- **Space**: capture/update **template** for the current pose
- **N**: next pose
- **P**: previous pose
- **F**: cycle federation (WNBF UK / PCA)
- **T**: toggle First Timers
- **C**: cycle categories (supports multiple selections)
- **S**: save a session snapshot (pose + proportions)

## Profiles

Profiles are stored in `config/profiles/` as JSON.

```bash
python -m bbcoach profile new --name edwin
python -m bbcoach profile edit --name edwin
python -m bbcoach profile show --name edwin
```

## Reports

```bash
python -m bbcoach report --profile edwin
```

