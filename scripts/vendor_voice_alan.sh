#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$ROOT_DIR/data/tts/en_GB-alan-medium"

ONNX_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx"
JSON_URL="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json"

mkdir -p "$DEST_DIR"

if command -v curl >/dev/null 2>&1; then
  curl -L "$ONNX_URL" -o "$DEST_DIR/voice.onnx"
  curl -L "$JSON_URL" -o "$DEST_DIR/voice.onnx.json"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$DEST_DIR/voice.onnx" "$ONNX_URL"
  wget -O "$DEST_DIR/voice.onnx.json" "$JSON_URL"
else
  echo "Error: curl or wget is required to download the voice model." >&2
  exit 1
fi

echo "Voice model installed to $DEST_DIR"
