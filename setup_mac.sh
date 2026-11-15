#!/bin/bash
# Clipper one-time setup for macOS (Apple Silicon or Intel). Run:  bash setup_mac.sh
set -e
cd "$(dirname "$0")"
echo "=== Clipper one-time setup (macOS) ==="
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is needed first. Install it from https://brew.sh, then run this again."
  exit 1
fi
echo "[1/3] Installing FFmpeg, Deno (yt-dlp needs it for YouTube) and Python 3.11..."
brew install ffmpeg deno python@3.11
echo "[2/3] Creating Python environment..."
[ -d .venv ] || "$(brew --prefix)/bin/python3.11" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
echo "[3/3] Installing Python packages (this takes a few minutes)..."
pip install -U -r requirements.txt
chmod +x start_app.sh run_clipper.sh
echo
echo "Setup done. Start Clipper with:  ./start_app.sh   (or ./run_clipper.sh for the terminal version)"
