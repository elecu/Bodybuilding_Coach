#!/usr/bin/env bash
set -euo pipefail

echo "=== BB Coach Fedora 40 installer ==="

dnf -y install python3 python3-pip python3-virtualenv gcc gcc-c++ make cmake \
  mesa-libGL ffmpeg || true

echo "Done. Create a venv and install requirements:" 
cat <<'TXT'
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
TXT
