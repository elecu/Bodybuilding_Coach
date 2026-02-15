#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$ROOT_DIR/vendor/piper/linux_x86_64"
TMP_DIR="$(mktemp -d)"
TARBALL_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$DEST_DIR"

if command -v curl >/dev/null 2>&1; then
  curl -L "$TARBALL_URL" -o "$TMP_DIR/piper.tar.gz"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$TMP_DIR/piper.tar.gz" "$TARBALL_URL"
else
  echo "Error: curl or wget is required to download Piper." >&2
  exit 1
fi

tar -xzf "$TMP_DIR/piper.tar.gz" -C "$TMP_DIR"

PIPER_PATH="$(find "$TMP_DIR" -type f -name piper | head -n 1)"
if [[ -z "$PIPER_PATH" ]]; then
  echo "Error: Could not find piper binary in the extracted archive." >&2
  exit 1
fi

PIPER_DIR="$(dirname "$PIPER_PATH")"
cp -a "$PIPER_DIR/." "$DEST_DIR/"
chmod +x "$DEST_DIR/piper"

echo "Piper installed to $DEST_DIR"
