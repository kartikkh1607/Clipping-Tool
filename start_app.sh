#!/bin/bash
# Clipper web app (macOS / Linux). Keep this terminal open while clips are being made.
cd "$(dirname "$0")"
URL="http://127.0.0.1:7860"
if curl -s -o /dev/null -m 2 "$URL/"; then   # already running: just open the page
  (command -v open >/dev/null && open "$URL") || xdg-open "$URL" >/dev/null 2>&1
  exit 0
fi
source .venv/bin/activate
export PYTHONUTF8=1 GRADIO_ANALYTICS_ENABLED=False
echo "Updating yt-dlp..."
python -m pip install -q -U "yt-dlp[default]" >/dev/null 2>&1
( for i in $(seq 1 180); do
    if curl -s -o /dev/null -m 1 "$URL/"; then
      (command -v open >/dev/null && open "$URL") || xdg-open "$URL" >/dev/null 2>&1; break
    fi; sleep 0.7
  done ) &
echo "Starting Clipper at $URL ..."
python app.py
