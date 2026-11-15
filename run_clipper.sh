#!/bin/bash
# Clipper terminal version (macOS / Linux): paste links, get clips.
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONUTF8=1
python -m pip install -q -U "yt-dlp[default]" >/dev/null 2>&1
while true; do
  echo
  read -r -p "Paste YouTube or Twitch link (or press Enter to quit): " URL
  [ -z "$URL" ] && exit 0
  read -r -p "Campaign (skiimasktimmy / none) [none]: " CAMP
  ARGS=()
  if [ -n "$CAMP" ] && [ "$CAMP" != "none" ]; then
    ARGS+=(--campaign "$CAMP")
  else
    read -r -p "Language (en/hi) [en]: " LANG_
    [ "$LANG_" = "hi" ] || LANG_=en
    ARGS+=(--lang "$LANG_")
  fi
  if [[ "$URL" == *twitch.tv/videos* ]]; then
    read -r -p "Section (e.g. 0:30:00-2:00:00, Enter = whole VOD): " SECTION
    [ -n "$SECTION" ] && ARGS+=(--section "$SECTION")
  fi
  python clipper.py "$URL" "${ARGS[@]}"
done
