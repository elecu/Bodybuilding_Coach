#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$ROOT_DIR/models/vosk"
MODEL_URL="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
PY="${PY_BIN:-python3}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
ZIP_PATH="$TMP_DIR/vosk.zip"

if command -v curl >/dev/null 2>&1; then
  curl -L "$MODEL_URL" -o "$ZIP_PATH"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$ZIP_PATH" "$MODEL_URL"
else
  echo "Error: curl or wget is required to download the Vosk model." >&2
  exit 1
fi

ZIP_PATH="$ZIP_PATH" DEST_DIR="$DEST_DIR" "$PY" - <<'PY'
import os
import shutil
import sys
import zipfile
from pathlib import Path

zip_path = Path(os.environ["ZIP_PATH"])
dest = Path(os.environ["DEST_DIR"])
extract_dir = zip_path.parent / "extract"
extract_dir.mkdir(parents=True, exist_ok=True)

def is_valid_model(path: Path) -> bool:
    has_conf = (path / "conf" / "model.conf").exists() or (path / "model.conf").exists()
    return has_conf and (path / "am" / "final.mdl").exists()

with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(extract_dir)

candidates = [p.parent for p in extract_dir.rglob("model.conf")]
model_root = None
for cand in candidates:
    if is_valid_model(cand):
        model_root = cand
        break

if model_root is None:
    print("Error: downloaded zip does not contain a valid Vosk model.", file=sys.stderr)
    sys.exit(1)

if dest.exists():
    shutil.rmtree(dest)
dest.parent.mkdir(parents=True, exist_ok=True)
shutil.move(str(model_root), str(dest))

# Legacy compatibility: keep root model.conf if only conf/model.conf exists.
conf_model = dest / "conf" / "model.conf"
root_model = dest / "model.conf"
if conf_model.exists() and not root_model.exists():
    shutil.copy2(conf_model, root_model)

if not is_valid_model(dest):
    print("Error: Vosk model install failed (missing required files).", file=sys.stderr)
    sys.exit(1)
PY

echo "Vosk model installed to $DEST_DIR"
