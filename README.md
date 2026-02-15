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
# (includes Open3D for 4-view AUTO MERGE / MANUAL ALIGN)

# Install offline voice-command model once (recommended)
scripts/vendor_vosk_model_small_en_us.sh

# Start live coach
python -m bbcoach live --profile edwin --camera-index 0 --width 1280 --height 720
```

## Voice commands (microphone, offline)

The app no longer auto-downloads Vosk at runtime. Install the model once:

```bash
scripts/vendor_vosk_model_small_en_us.sh
```

The model is stored in `models/vosk` and reused on every run.

## Open3D (4-view merge/align)

`open3d` is part of the base dependencies (`requirements.txt`), so it installs with the normal setup and is reused from the same virtualenv on next runs.

## Spoken coaching (TTS)

Spoken coaching works out-of-the-box on Linux x86_64 using the vendored Piper binary and the **Alan (UK)** voice model.

- Piper binary: `vendor/piper/linux_x86_64/piper`
- Voice model: `data/tts/en_GB-alan-medium/voice.onnx` (+ `voice.onnx.json`)
- If no audio player is available, the app falls back to **espeak** (if installed).

Enable spoken coaching:

```bash
python -m bbcoach live --profile edwin --coach-voice --tts-backend piper_bin
```

If you need to (re)download the vendored assets:

```bash
scripts/vendor_piper_linux_x86_64.sh
scripts/vendor_voice_alan.sh
```

## Device connection flow (Offline/Webcam/Kinect)

The desktop app now uses a **connect-on-demand** flow:

- The app always starts in a safe **Offline** state (no hardware required).
- Use the **Device** panel to pick an input source:
  - `None (Offline)`
  - `Webcam`
  - `Kinect`
- Press **Connect / Start** only when you want to open a device.
- If connection fails, the app stays up and shows an actionable error with **Retry**.
- Press **Disconnect / Stop** to release the active source before switching.
- If a stream is lost, the app transitions to **ERROR** and asks you to reconnect + retry.
- **Use sample input** is available in Offline/Error for hardware-free UI testing.

Persistence/logging:

- Last selected source is saved in `config/device_connection.json`.
- Device connection attempts and failures are logged to `sessions/logs/device.log`.

## Key controls (live window)

- **Q**: quit
- **Space**: capture/update **template** for the current pose
- **N**: next pose
- **P**: previous pose
- **F**: cycle federation (WNBF UK / UKBFF / PCA)
- **X**: toggle coach voice (TTS)
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
