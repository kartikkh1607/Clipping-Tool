#!/usr/bin/env python3
"""
Clipper - paste a YouTube link, get the best moments as vertical, captioned clips.

Pipeline
  1. Download an analysis copy (yt-dlp) + read YouTube's "Most replayed" heatmap
  2. Transcribe with word timestamps (faster-whisper, GPU if available)
  3. Score every possible window: most-replayed heat + loudness/hype + speech
  4. (Optional) Gemini watches the best candidates, picks the final clips and writes titles
  5. Snap every cut to word boundaries, keeping the payoff inside the clip
  6. Re-download only the chosen moments at the best quality available (SDR, up to 4K / 60 fps)
  7. Render 1080x1920 (or 1440p / 4K) clips at the source frame rate: speaker-tracked crop, blurred-background or
     facecam layout, word-highlight captions, hook banner, two-pass loudness normalisation
  8. Check every clip (length, frame rate, audio, black frames, loudness) before calling it done
  9. Save to  <output_root>/DD-MM-YYYY/NN_<channel>_<video>/

Usage
  python clipper.py "https://www.youtube.com/watch?v=..."
  python clipper.py URL1 URL2 ...            (several videos)
  python clipper.py --list links.txt         (one link per line)
  python clipper.py --file video.mp4 --credit "CreatorName"   (local file)
  python clipper.py --live "https://www.twitch.tv/<channel>"  (clip a stream while it's live)
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:   # framing.py sits next to this file, whatever folder we're started from
    sys.path.insert(0, str(HERE))
import framing  # noqa: E402
import reactfx  # noqa: E402
CONFIG_PATH = HERE / "config.json"

IS_WINDOWS = os.name == "nt"
IS_MAC = sys.platform == "darwin"
# Caption fonts that exist out of the box on each system (libass falls back to a default font otherwise)
if IS_WINDOWS:
    _FONT, _HINDI_FONT = "Arial Black", "Nirmala UI"
elif IS_MAC:
    _FONT, _HINDI_FONT = "Arial Black", "Kohinoor Devanagari"
else:
    _FONT, _HINDI_FONT = "DejaVu Sans", "Noto Sans Devanagari"

DEFAULTS = {
    "output_root": "",
    "num_clips": 5,
    "min_clip_seconds": 20,
    "max_clip_seconds": 40,
    "layout": "auto",            # "auto" = speaker-tracked crop for talking heads, blur otherwise
                                 # "blur" = full frame on blurred bg, "crop" = 9:16 crop that follows faces,
                                 # "facecam" = facecam on top + gameplay below
    "zoom": 1.25,                # blur layout: how much to enlarge the gameplay (1.0 = fit width)
    "show_title_banner": True,   # hook text box at the top of the clip
    "title_banner_seconds": 3,   # how long the hook box stays up (0 = the whole clip)
    "censor_captions": True,     # mask slurs and hard swears in the captions (N****); the audio is untouched
    "auto_facecam": True,        # "auto" layout: streams with a webcam in a corner get the split facecam layout
    "react_zoom": True,          # facecam layout: cut to the streamer's face full-screen on the biggest reactions
    "react_max": 2,              # face zooms per clip
    "react_seconds": 2.0,        # length of each face zoom
    "react_min_score": 0.35,     # how sudden a sound jump must be to count as a reaction (0-1)
    "punch_word": True,          # one big word BEHIND the streamer's head during the first face zoom
    "punch_color": "#FFFFFF",
    "words_per_caption": 3,
    "caption_font": _FONT,
    "caption_size": 84,
    "whisper_model": "large-v3-turbo",   # GPU model
    "whisper_cpu_model": "small",        # used if the GPU fails (and always on a Mac)
    "whisper_device": "auto",            # auto | cuda | cpu
    "whisper_batch_size": 8,             # GPU batched transcription (1 = off); lower it if the GPU runs out of memory
    "render_workers": 2,                 # clips rendered at the same time
    "language": "en",                    # en | hi | "" (auto-detect)
    "hindi_caption_font": _HINDI_FONT,   # supports Devanagari
    "hinglish_roman_captions": True,     # hi: transliterate captions to Roman Hinglish via Gemini
    "gemini_api_key": "",                # free key: https://aistudio.google.com/apikey
    "gemini_model": "gemini-3.5-flash-lite",                           # free tier: 500/day, 15/min
    "gemini_fallback_models": ["gemini-3.6-flash", "gemini-3.8-flash"],  # free tier: 20/day, 5/min each
    "max_download_height": 1080,         # analysis copy (whole video); the final clips are re-downloaded in HQ
    "hq_refetch": True,                  # re-download just the chosen clips at the best quality available
    "hq_max_height": 2160,               # YouTube: up to 4K for the final clips (downscaled to 1080x1920 = sharp)
    "twitch_hq_max_height": 1080,        # Twitch: "source" quality is usually 1080p60
    "max_fps": 60,                       # keep the source frame rate up to this (YouTube Shorts plays 60 fps)
    "output_height": 1920,               # 1920 = 1080x1920, 2560 = 1440x2560, 3840 = 2160x3840 (4K)
    "output_fps": "source",              # "source" (capped at max_fps) | 60 | 30
    "file_size": "balanced",             # smaller | balanced | best: sets nvenc_cq / x264_crf / max_bitrate
    "encoder": "auto",                   # auto (NVIDIA GPU if it works, else CPU) | nvenc | x264
    "nvenc_cq": 22,                      # GPU encoder quality (lower = better, bigger files)
    "x264_crf": 20,                      # CPU encoder quality (lower = better, bigger files)
    "max_bitrate": "20M",                # cap at 1080p (x1.7 at 1440p, x3.5 at 4K): YouTube re-encodes everything
    "x264_preset": "slow",               # CPU encoder effort: slower = better quality per MB
    "audio_bitrate": "192k",
    "target_lufs": -14,                  # loudness target (YouTube/TikTok normalise around -14)
    "qa_check": True,                    # check every rendered clip before calling it done
    "face_tracking": True,               # crop/auto layouts follow faces; facecam layout finds a moved cam
    "cookies_from_browser": "",          # e.g. "chrome" for age-restricted videos (not used with pot_provider)
    "pot_provider": True,                # start/check the bgutil PO Token server before YouTube downloads
    "pot_provider_dir": "~/bgutil-ytdlp-pot-provider",
    "pot_provider_port": 4416,
    "skip_start_seconds": 10,            # avoid intros
    "skip_end_seconds": 20,              # avoid outros / "subscribe"
    "keep_source": False,                # keep the downloaded full video
    "twitch_max_height": 720,            # Twitch VODs are hours long: cap the download
    "overlay_image": "",                 # PNG burned on every clip (transparency kept)
    "overlay_width_pct": 85,             # overlay width as % of 1080
    "facecam_box": None,                 # [x, y, w, h] in facecam_ref pixels, for layout "facecam"
    "facecam_ref": [1280, 720],          # source resolution the facecam box was measured on
    "ai_pool_size": 24,                  # candidates Gemini reviews (was 12)
    "vision_review": True,               # Gemini WATCHES each candidate (low-res preview) before picking
    "vision_batch": 6,                   # candidate previews per Gemini video request
    "vision_preview_height": 360,        # preview resolution sent to Gemini (keeps tokens low)
    "live_clips_per_window": 2,          # live mode: clips from each 10-minute window
    "live_ai_pool_size": 12,             # live mode: candidates Gemini reviews per window (keeps a long stream
                                         # inside the free daily quota)
}
CAMPAIGNS_PATH = HERE / "campaigns.json"

# Every layout is designed on a 1080x1920 canvas and scaled by k = output_height / 1920 at render time,
# so 1440p and 4K are rendered at their real size (not upscaled at the end). Captions stay on a 1080x1920
# ASS canvas: libass scales them, sharply, to the real frame.
BASE_W, BASE_H = 1080, 1920
RESOLUTIONS = {"1080p": 1920, "1440p": 2560, "4K": 3840}      # label -> output_height
BITRATE_FACTOR = {1920: 1.0, 2560: 1.7, 3840: 3.5}           # max_bitrate is set for 1080p
FILE_SIZES = {   # Settings "File size" -> encoder quality (lower cq/crf = better, bigger)
    "smaller": {"nvenc_cq": 25, "x264_crf": 23, "max_bitrate": "12M"},
    "balanced": {"nvenc_cq": 22, "x264_crf": 20, "max_bitrate": "20M"},
    "best": {"nvenc_cq": 19, "x264_crf": 18, "max_bitrate": "35M"},
}

# facecam layout (on the 1080x1920 canvas): facecam on top, overlay band, gameplay below
FC_CAM_H = 720
FC_BAND_H = 160
FC_GAME_Y = FC_CAM_H + FC_BAND_H
FC_BAND_COLOR = "0x0E0B16"


def ev(v: float) -> int:
    """Nearest even integer (video sizes and offsets must be even for yuv420p)."""
    return int(round(v / 2.0)) * 2


def out_height(cfg: dict) -> int:
    try:
        h = int(cfg.get("output_height") or BASE_H)
    except (TypeError, ValueError):
        h = BASE_H
    return h if h in BITRATE_FACTOR else BASE_H


def out_size(cfg: dict) -> tuple[int, int]:
    """(width, height) of the finished clip: 1080x1920, 1440x2560 or 2160x3840."""
    h = out_height(cfg)
    return ev(BASE_W * h / BASE_H), h


def fc_geometry(cfg: dict) -> tuple[int, int, int]:
    """(cam height, band height, gameplay top) of the facecam layout at the output size."""
    k = out_height(cfg) / BASE_H
    cam, band = ev(FC_CAM_H * k), ev(FC_BAND_H * k)
    return cam, band, cam + band


def _rate(text: str) -> tuple[float, str] | None:
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([kKmM]?)\s*$", str(text or ""))
    return (float(m.group(1)), m.group(2).upper()) if m else None


def scaled_bitrate(cfg: dict) -> str:
    """max_bitrate (set for 1080p) scaled for the output resolution: x1.7 at 1440p, x3.5 at 4K."""
    mx = str(cfg.get("max_bitrate") or "20M")
    r = _rate(mx)
    f = BITRATE_FACTOR[out_height(cfg)]
    if not r or f == 1.0:
        return mx
    return f"{r[0] * f:g}{r[1]}"


def mbps(rate: str) -> float:
    r = _rate(rate)
    if not r:
        return 20.0
    return r[0] / 1000 if r[1] == "K" else r[0] if r[1] == "M" else r[0] / 1e6


def estimate_clip_mb(cfg: dict, seconds: float = 30.0) -> tuple[float, float]:
    """(typical, worst case) MB for a clip of this length at the chosen resolution/fps/file size.
    Quality-targeted encoding uses only what the picture needs: busy gameplay runs near the bitrate cap,
    talking heads far below it. Typical = ~55% of the cap at 60 fps, ~40% at 30 fps; worst = the cap."""
    cap = mbps(scaled_bitrate(cfg))
    fps = str(cfg.get("output_fps") or "source")
    fill = 0.40 if fps == "30" else 0.55
    audio = mbps(str(cfg.get("audio_bitrate") or "192k"))
    to_mb = lambda v: (v + audio) * seconds / 8
    return round(to_mb(cap * fill)), round(to_mb(cap))


def render_fps(cfg: dict, src_fps: float) -> tuple[str, float]:
    """Output frame rate: "source" = the source's own rate capped at max_fps; 60 or 30 = exactly that."""
    want = str(cfg.get("output_fps") or "source").strip().lower()
    if want in ("60", "30"):
        return want, float(want)
    return output_fps(src_fps, float(cfg.get("max_fps") or 60))

PROMO_WORDS = [
    "sponsor", "sponsored", "use code", "promo code", "link in the description",
    "link in description", "subscribe", "patreon", "merch", "today's video is brought",
    "check out", "discount", "affiliate", "like and subscribe",
]
# Only words that signal a real moment. Everyday words ("what", "yes", "run", "win", "bro", "kya"...) used
# to be here too, which made ordinary chatter score as hype. Question marks no longer count either.
HYPE_WORDS = [
    "oh my god", "no way", "let's go", "lets go", "holy", "insane", "crazy", "wow", "clutch",
    "no no no", "oh no", "omg", "rage", "impossible", "unbelievable", "hahaha", "what the",
    "are you kidding", "are you serious", "no shot", "oh my", "bruh", "yooo", "lmao", "screaming",
    # Hindi / Hinglish (Roman + Devanagari, since Whisper writes Hindi in Devanagari)
    "arey", "arre", "oye", "pagal", "bachao", "bhaag", "maaro", "khatam", "mar gaya", "kya baat hai",
    "oh bhai", "bhai bhai", "kya scene", "arey yaar",
    "अरे", "ओए", "पागल", "बचाओ", "भाग", "मारो", "खत्म", "मर गया", "क्या बात है", "भाई भाई",
]
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


# ----------------------------------------------------------------------------- utils
LOG_HOOK = None  # optional callable(str): the web app (app.py) streams progress messages through this


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode(), flush=True)
    if LOG_HOOK:
        LOG_HOOK(line)


def slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^\w\sऀ-ॿ-]", "", text or "", flags=re.UNICODE)
    text = re.sub(r"[।॥]", "", text)  # danda punctuation
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    return (text[:max_len].rstrip("_")) or "video"


def fmt_ts(sec: float) -> str:
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_time(t: str) -> float:
    parts = [float(p) for p in t.strip().split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def parse_section(text: str) -> tuple[float, float]:
    """"0:30:00-2:00:00" -> (1800.0, 7200.0)"""
    m = re.fullmatch(r"\s*([\d:.]+)\s*-\s*([\d:.]+)\s*", text or "")
    if not m:
        sys.exit(f"ERROR: --section must look like 0:30:00-2:00:00, got '{text}'")
    a, b = parse_time(m.group(1)), parse_time(m.group(2))
    if b <= a:
        sys.exit("ERROR: --section end must be after its start")
    return a, b


def is_youtube(url: str) -> bool:
    return "youtu" in url


def is_twitch(url: str) -> bool:
    return "twitch.tv" in url


def run(cmd: list[str], cwd: Path | None = None, quiet: bool = True) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if res.returncode != 0:
        tail = (res.stderr or res.stdout or "")[-2500:]
        raise RuntimeError(f"Command failed: {' '.join(cmd[:4])} ...\n{tail}")
    return res


def load_config(args) -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception as e:
            log(f"WARNING: could not read config.json ({e}); using defaults")
    if not cfg.get("gemini_api_key"):
        cfg["gemini_api_key"] = os.environ.get("GEMINI_API_KEY", "")
    cfg["campaign"] = ""
    if args.campaign and args.campaign.lower() != "none":
        presets = json.loads(CAMPAIGNS_PATH.read_text(encoding="utf-8")) if CAMPAIGNS_PATH.exists() else {}
        name = args.campaign.lower()
        if name not in presets:
            sys.exit(f"ERROR: campaign '{args.campaign}' not found in campaigns.json ({', '.join(presets) or 'none defined'})")
        cfg.update(presets[name])
        cfg["campaign"] = name
        log(f"Campaign preset: {name}")
    if args.overlay_image:
        cfg["overlay_image"] = args.overlay_image
    if cfg.get("overlay_image"):
        p = Path(cfg["overlay_image"])
        cfg["overlay_image"] = str(p if p.is_absolute() else HERE / p)
        if not Path(cfg["overlay_image"]).exists():
            sys.exit(f"ERROR: overlay image not found: {cfg['overlay_image']}")
    cfg["section"] = parse_section(args.section) if args.section else None
    if args.out:
        cfg["output_root"] = args.out
    if args.clips:
        cfg["num_clips"] = args.clips
    if args.lang:
        cfg["language"] = "" if args.lang == "auto" else args.lang
    cfg["verbose"] = args.verbose
    cfg["_lang"] = cfg.get("language") or ""
    if not cfg["output_root"]:
        root = input("Where should clips be saved? (folder path): ").strip().strip('"')
        cfg["output_root"] = root
        saved = {}
        if CONFIG_PATH.exists():
            try:
                saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                saved = {}
        saved["output_root"] = root
        CONFIG_PATH.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        log(f"Saved output folder to config.json: {root}")
    return cfg


def check_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            sys.exit(f"ERROR: {tool} not found. Run setup_windows.bat (or install FFmpeg) and reopen the terminal.")


# ----------------------------------------------------------------------------- download
def ensure_pot_provider(cfg: dict) -> None:
    """Make sure the bgutil PO Token HTTP server is running (YouTube's bot check needs PO tokens)."""
    import urllib.request

    port = int(cfg["pot_provider_port"])

    def alive() -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    if alive():
        log(f"PO Token provider active on port {port}")
        return
    server = Path(os.path.expanduser(cfg["pot_provider_dir"])) / "server"
    if (server / "build" / "main.js").exists() and shutil.which("node"):
        cmd = ["node", "build/main.js", "--port", str(port)]
    elif (server / "node_modules").exists() and shutil.which("deno"):
        cmd = ["deno", "run", "--allow-env", "--allow-net", "--allow-ffi=.", "--allow-read=.",
               "../src/main.ts", "--port", str(port)]
    else:
        raise RuntimeError(
            f"PO Token provider not found in {server}.\n"
            "Set it up once (needs git + Node.js 22+):\n"
            "  git clone --branch 2.0.2 https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git %USERPROFILE%\\bgutil-ytdlp-pot-provider\n"
            "  cd %USERPROFILE%\\bgutil-ytdlp-pot-provider\\server && npm ci && npx tsc\n"
            "or set \"pot_provider\": false in config.json.")
    log("Starting PO Token provider...")
    flags = 0x00000008 | 0x00000200 | 0x08000000 if os.name == "nt" else 0  # detached, new group, no window
    with open(server / "provider.log", "ab") as lf:
        subprocess.Popen(cmd, cwd=str(server), stdout=lf, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, creationflags=flags)
    for _ in range(60):
        if alive():
            log(f"PO Token provider active on port {port}")
            return
        time.sleep(0.5)
    raise RuntimeError(f"PO Token provider did not start on port {port}. See {server / 'provider.log'}")


def format_spec(url: str, height: int) -> tuple[str, list[str]]:
    """yt-dlp format string + sort for the best SDR video up to `height` (HDR would look washed out
    after the 8-bit conversion, and YouTube always offers an SDR version too)."""
    sort = ["hdr:sdr", f"res:{height}", "fps"]
    if is_twitch(url):  # Twitch HLS formats already contain audio
        return f"b[height<={height}]/b", sort
    return f"bv*[height<={height}]+ba/b[height<={height}]/b", sort


def ydl_base_opts(url: str, cfg: dict) -> dict:
    """yt-dlp options shared by the analysis download and the HQ re-download."""
    opts = {
        "quiet": True,
        "no_warnings": False,
        "noplaylist": True,
        "noprogress": False,
        "socket_timeout": 30,
        "retries": 15,
        "fragment_retries": 15,
        "continuedl": True,
    }
    if cfg.get("verbose"):
        opts.update({"verbose": True, "quiet": False})
    if is_youtube(url):  # YouTube-only: bot check workarounds
        if cfg.get("pot_provider"):
            ensure_pot_provider(cfg)
            opts["extractor_args"] = {"youtubepot-bgutilhttp": {"base_url": [f"http://127.0.0.1:{int(cfg['pot_provider_port'])}"]}}
            if cfg.get("cookies_from_browser"):
                log("Note: cookies_from_browser is ignored while pot_provider is on")
        elif cfg.get("cookies_from_browser"):
            opts["cookiesfrombrowser"] = (cfg["cookies_from_browser"],)
    return opts


def best_available(info: dict) -> tuple[int, float]:
    """Highest SDR video height and frame rate this source offers."""
    hs, fps = [0], [0.0]
    for f in info.get("formats") or []:
        if f.get("vcodec") in (None, "none") or not f.get("height"):
            continue
        if str(f.get("dynamic_range") or "SDR").upper() != "SDR":
            continue
        hs.append(int(f["height"]))
        fps.append(float(f.get("fps") or 0))
    return max(hs), max(fps)


def download(url: str, work: Path, cfg: dict) -> tuple[Path, dict]:
    import yt_dlp

    h = int(cfg["max_download_height"])
    if is_twitch(url):
        h = min(h, int(cfg["twitch_max_height"]))
    fmt, sort = format_spec(url, h)
    opts = {
        **ydl_base_opts(url, cfg),
        "format": fmt,
        "format_sort": sort,
        "merge_output_format": "mp4",
        "outtmpl": str(work / "source.%(ext)s"),
    }
    if cfg.get("_progress_hook"):  # web app: download progress bar
        opts["progress_hooks"] = [cfg["_progress_hook"]]
    section = cfg.get("section")
    pre = {}
    if cfg.get("campaign") or section:  # metadata first: check the source and the VOD length before downloading
        with yt_dlp.YoutubeDL({**opts, "quiet": True}) as ydl:
            pre = ydl.extract_info(url, download=False)
        check_campaign_source(url, pre, cfg)
    full = float(pre.get("duration") or 0)
    if section and full and section[0] >= full:
        raise RuntimeError(f"--section starts at {fmt_ts(section[0])} but the video is only {fmt_ts(full)} long")
    if section and full and (section[0] > 0 or section[1] < full):
        a, b = section[0], min(section[1], full)
        section = (a, b)
        log(f"Downloading only {fmt_ts(a)}-{fmt_ts(b)}")
        opts["download_ranges"] = yt_dlp.utils.download_range_func(None, [(a, b)])
    else:
        section = None
    for attempt in range(1, 4):  # flaky connections: resume the .part file a few times
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except yt_dlp.utils.DownloadError as e:
            if attempt == 3 or "confirm you" in str(e):
                raise
            log(f"Download interrupted ({str(e).strip()[:100]}) - resuming, attempt {attempt + 1}/3")
            time.sleep(5)
    path = None
    for d in info.get("requested_downloads") or []:
        if d.get("filepath") and Path(d["filepath"]).exists():
            path = Path(d["filepath"])
    if path is None:
        found = sorted(work.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
        if not found:
            raise RuntimeError("Download finished but no video file was found.")
        path = found[0]
    meta = {
        "title": info.get("title") or "video",
        "channel": info.get("channel") or info.get("uploader") or "creator",
        "channel_url": info.get("channel_url") or info.get("uploader_url") or "",
        "url": info.get("webpage_url") or url,
        "id": info.get("id") or "",
        "duration": float(info.get("duration") or 0),
        "heatmap": info.get("heatmap") or [],
        "view_count": info.get("view_count"),
        "offset": section[0] if section else 0.0,  # clip times + offset = time in the original video
        "chapters": info.get("chapters") or [],
        "twitch": is_twitch(url),
        "source_url": url,
        "best_height": best_available(info)[0],
        "best_fps": best_available(info)[1],
    }
    if section:  # the file is only the section; its own length is what we process
        meta["duration"] = probe_duration(path)
        meta["heatmap"] = [{**p, "start_time": p["start_time"] - section[0], "end_time": p["end_time"] - section[0]}
                           for p in meta["heatmap"] if p.get("end_time", 0) > section[0] and p.get("start_time", 0) < section[1]]
    return path, meta


def check_campaign_source(url: str, info: dict, cfg: dict) -> None:
    """Campaign footage rules: refuse sources from the wrong platform or channel."""
    if not cfg.get("campaign"):
        return
    plat, chan = cfg.get("source_platform"), (cfg.get("source_channel") or "").lower()
    if plat == "twitch" and not is_twitch(url):
        raise RuntimeError(f"Campaign '{cfg['campaign']}' only allows Twitch footage, not: {url}")
    if chan:
        ids = {str(info.get(k) or "").lower() for k in ("uploader_id", "channel_id", "channel", "uploader")}
        if chan not in ids:
            raise RuntimeError(f"Campaign '{cfg['campaign']}' only allows footage from '{chan}', "
                               f"this video is from '{info.get('uploader') or info.get('channel')}'")


def probe_duration(path: Path) -> float:
    res = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=nw=1:nk=1", str(path)])
    return float(res.stdout.strip() or 0)


def probe_size(path: Path) -> tuple[int, int]:
    res = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=width,height", "-of", "csv=p=0:s=x", str(path)])
    w, h = res.stdout.strip().split("\n")[0].split("x")[:2]
    return int(w), int(h)


def has_audio(path: Path) -> bool:
    res = run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
               "stream=index", "-of", "csv=p=0", str(path)])
    return bool(res.stdout.strip())


def probe_fps(path: Path) -> float:
    """Frame rate of the first video stream (30 if it can't be read)."""
    res = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
               "stream=avg_frame_rate,r_frame_rate", "-of", "default=nw=1", str(path)])
    vals = dict(line.split("=", 1) for line in res.stdout.splitlines() if "=" in line)
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            num, den = vals.get(key, "0/0").split("/")
            fps = float(num) / float(den)
            if 1 <= fps <= 240:
                return fps
        except (ValueError, ZeroDivisionError):
            continue
    return 30.0


def output_fps(src_fps: float, max_fps: float = 60) -> tuple[str, float]:
    """Frame rate to render at: the source's own rate (NTSC rates kept exact), capped at max_fps."""
    if src_fps > max_fps + 0.5:
        src_fps = max_fps
    for num, den in ((24000, 1001), (30000, 1001), (60000, 1001)):
        if abs(src_fps - num / den) < 0.02:
            return f"{num}/{den}", num / den
    for std in (24, 25, 30, 48, 50, 60):
        if abs(src_fps - std) < 0.15:
            return str(std), float(std)
    return f"{src_fps:.3f}", src_fps


# ----------------------------------------------------------------------------- high-quality re-download
HQ_PAD = 4.0   # seconds of extra footage around each chosen clip (cuts may move a little after alignment)


def fetch_hq_segments(clips: list[dict], meta: dict, work: Path, cfg: dict) -> dict[int, Path]:
    """Download ONLY the chosen moments at the best quality available (one yt-dlp call for all of them).
    The analysis copy stays the fallback, so a failure here never loses a clip. Returns {clip index: file}."""
    url = meta.get("source_url")
    if not url or not cfg.get("hq_refetch", True) or not clips:
        return {}
    cap = int(cfg["twitch_hq_max_height"] if meta.get("twitch") else cfg["hq_max_height"])
    best_h = min(cap, int(meta.get("best_height") or 0))
    best_fps = float(meta.get("best_fps") or 0)
    if best_h <= meta.get("_analysis_h", 0) and best_fps <= meta.get("_analysis_fps", 0) + 1:
        log("The analysis copy is already the best quality available - rendering from it")
        return {}
    import yt_dlp

    off = meta.get("offset", 0.0)
    ranges = [(max(0.0, c["start"] + off - HQ_PAD), c["end"] + off + HQ_PAD) for c in clips]
    fmt, sort = format_spec(url, cap)
    hq = work / "hq"
    hq.mkdir(exist_ok=True)
    opts = {**ydl_base_opts(url, cfg), "format": fmt, "format_sort": sort, "merge_output_format": "mkv",
            "outtmpl": str(hq / "seg_%(section_start)d.%(ext)s"),
            "download_ranges": yt_dlp.utils.download_range_func(None, ranges)}
    log(f"Fetching the {len(clips)} chosen moments in high quality (best available: {best_h}p"
        + (f"{best_fps:.0f}" if best_fps else "") + ")...")
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as e:
        log(f"High-quality download failed ({str(e).strip()[:150]}) - rendering from the analysis copy")
        return {}
    out: dict[int, Path] = {}
    for d in info.get("requested_downloads") or []:
        p, ss = d.get("filepath"), d.get("section_start")
        if not p or ss is None or not Path(p).exists():
            continue
        for i, (a, _) in enumerate(ranges):
            if i not in out and abs(a - float(ss)) < 0.5:
                out[i] = Path(p)
                break
    if len(out) < len(clips):
        log(f"High-quality download returned {len(out)} of {len(clips)} moments - the rest use the analysis copy")
    return out


def read_wav(path: Path, start: float, dur: float) -> tuple[np.ndarray, int]:
    """Mono 16-bit WAV samples [start, start+dur) as float32."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        a = max(0, int(start * sr))
        if a >= w.getnframes():
            return np.zeros(0, dtype=np.float32), sr
        w.setpos(a)
        data = np.frombuffer(w.readframes(int(dur * sr)), dtype=np.int16).astype(np.float32)
    return data, sr


def best_lag(ref: np.ndarray, probe: np.ndarray) -> tuple[float, int]:
    """Normalised cross-correlation of probe slid along ref: (best correlation 0-1, lag in samples)."""
    n, m = len(ref), len(probe)
    probe = probe - probe.mean()
    size = 1 << int(np.ceil(np.log2(n + m)))
    raw = np.fft.irfft(np.fft.rfft(ref, size) * np.conj(np.fft.rfft(probe, size)), size)[: n - m + 1]
    c1 = np.concatenate([[0.0], np.cumsum(ref, dtype=np.float64)])
    c2 = np.concatenate([[0.0], np.cumsum(ref.astype(np.float64) ** 2)])
    win_sum, win_sq = c1[m:] - c1[:-m], c2[m:] - c2[:-m]
    energy = np.sqrt(np.maximum(win_sq - win_sum ** 2 / m, 1e-9)) * np.linalg.norm(probe)
    corr = raw / np.maximum(energy, 1e-9)
    lag = int(np.argmax(corr))
    return float(corr[lag]), lag


def align_segment(seg: Path, wav: Path, nominal: float, work: Path) -> float | None:
    """Where (in analysis-file seconds) the HQ segment really starts. Partial downloads begin on a keyframe,
    so the requested start is only approximate; matching the segment's audio against the analysis audio
    pins it down to the sample. None if the match isn't convincing (e.g. silence)."""
    seg_wav = work / f"{seg.stem}_align.wav"
    try:
        run(["ffmpeg", "-y", "-v", "error", "-i", str(seg), "-t", "40", "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(seg_wav)])
        s, sr = read_wav(seg_wav, 0, 40)
    except Exception:
        return None
    finally:
        seg_wav.unlink(missing_ok=True)
    for a0 in (2.0, 10.0, 18.0):                   # try a few 8 s probes in case one is silent
        probe = s[int(a0 * sr): int((a0 + 8) * sr)]
        if len(probe) < 4 * sr or probe.std() < 30:
            continue
        lo = max(0.0, nominal + a0 - 12.0)
        ref, _ = read_wav(wav, lo, 32.0)
        if len(ref) <= len(probe):
            continue
        corr, lag = best_lag(ref, probe)
        if corr >= 0.6:
            # the extracted WAV starts at the segment's first AUDIO sample; a stream-copied cut can start its
            # audio a little after its video (FFmpeg 9: one AAC frame, ~21 ms), and rendering seeks on the
            # video timeline, so return where the segment's VIDEO starts
            return lo + lag / sr - a0 - av_start_gap(seg)
    return None


def stream_start_times(path: Path) -> dict[str, float]:
    """{"video": start_time, "audio": start_time} of the first stream of each kind (missing ones left out)."""
    try:
        res = run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "json", str(path)])
        out: dict[str, float] = {}
        for s in json.loads(res.stdout).get("streams") or []:
            kind, st = s.get("codec_type"), s.get("start_time")
            if kind in ("video", "audio") and kind not in out and st not in (None, "N/A"):
                out[kind] = float(st)
        return out
    except Exception:
        return {}


def av_start_gap(path: Path) -> float:
    """How much later the audio stream starts than the video stream (seconds; 0 if unknown)."""
    st = stream_start_times(path)
    if "audio" in st and "video" in st:
        gap = st["audio"] - st["video"]
        if abs(gap) < 2.0:   # anything bigger is a broken file, not a priming offset
            return gap
    return 0.0


# ----------------------------------------------------------------------------- facecam
def find_facecam(url: str, cfg: dict, at: float = 600.0) -> Path:
    """Save a frame from ~10 min in with a labelled pixel grid, to read the facecam box from."""
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "format": f"b[height<={int(cfg['twitch_max_height'])}]/b"}) as ydl:
        info = ydl.extract_info(url, download=False)
    at = min(at, float(info.get("duration") or at * 2) / 2)
    out = HERE / "facecam_check.png"
    tmp = HERE / "_facecam_raw.png"
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{at:.0f}", "-i", info["url"], "-frames:v", "1", str(tmp)])
    w, h = probe_size(tmp)
    labels = []
    for x in range(0, w, 80):
        labels.append(f"Dialogue: 0,0:00:00.00,0:00:01.00,L,,0,0,0,,{{\\an7\\pos({x + 2},2)}}{x}")
    for y in range(80, h, 80):
        labels.append(f"Dialogue: 0,0:00:00.00,0:00:01.00,L,,0,0,0,,{{\\an7\\pos(2,{y + 2})}}{y}")
    (HERE / "_grid.ass").write_text(f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: L,Arial,{max(12, h // 50)},&H0000FFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + "\n".join(labels) + "\n", encoding="utf-8")
    run(["ffmpeg", "-y", "-v", "error", "-i", tmp.name, "-vf",
         "drawgrid=w=40:h=40:t=1:c=white@0.25,drawgrid=w=80:h=80:t=1:c=yellow@0.7,ass=_grid.ass",
         "-frames:v", "1", out.name], cwd=HERE)
    tmp.unlink(missing_ok=True)
    (HERE / "_grid.ass").unlink(missing_ok=True)
    log(f"Saved {out} ({w}x{h}, frame at {fmt_ts(at)} of '{info.get('title')}'). Read the facecam box "
        f"as x,y,w,h in these pixels and put it in campaigns.json -> facecam_box (facecam_ref [{w}, {h}]).")
    return out


def facecam_box_px(cfg: dict, src_w: int, src_h: int) -> tuple[float, float, float, float] | None:
    """The configured facecam_box scaled to this source's size (None if not set or set to "auto")."""
    box = cfg.get("facecam_box")
    if not box or isinstance(box, str):
        return None
    rw, rh = cfg.get("facecam_ref") or [src_w, src_h]
    sx, sy = src_w / rw, src_h / rh
    return box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy


def facecam_crop(cfg: dict, src_w: int, src_h: int) -> tuple[int, int, int, int] | None:
    """Facecam box scaled to the source and trimmed (centered) to the 1080x720 slot's aspect."""
    box = facecam_box_px(cfg, src_w, src_h)
    return fit_cam_crop(box, src_w, src_h) if box else None


def fit_cam_crop(box, src_w: int, src_h: int) -> tuple[int, int, int, int] | None:
    """Trim a cam box (source px) to the 1080x720 slot's aspect, centered.
    Trimming instead of widening keeps stream overlay borders/widgets around the cam out of the shot."""
    x, y, w, h = box
    target = 1080 / FC_CAM_H
    if w / h > target:
        nw = h * target
        x, w = x + (w - nw) / 2, nw
    else:
        nh = w / target
        y, h = y + (h - nh) / 2, nh
    if x < 0 or y < 0 or x + w > src_w or y + h > src_h:
        return None
    even = lambda v: int(v) // 2 * 2
    return even(x), even(y), even(w), even(h)


def facecam_visible(src: Path, crop: tuple, times: list[float]) -> bool:
    """True if the facecam area shows a live picture (not black/frozen) in at least one sample."""
    x, y, w, h = crop
    for t in times:
        res = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.1f}", "-i", str(src), "-frames:v", "1",
                              "-vf", f"crop={w}:{h}:{x}:{y},scale=64:64,format=gray", "-f", "rawvideo", "-"],
                             capture_output=True)
        px = np.frombuffer(res.stdout, dtype=np.uint8)
        if px.size == 64 * 64 and px.std() > 12 and 15 < px.mean() < 245:
            return True
    return False


# ----------------------------------------------------------------------------- transcription
def _add_nvidia_dll_dirs() -> None:
    """On Windows, make the pip-installed CUDA libraries (cuBLAS/cuDNN) findable."""
    if os.name != "nt":
        return
    try:
        import nvidia  # namespace package from nvidia-cublas-cu12 / nvidia-cudnn-cu12
        for base in list(nvidia.__path__):
            for sub in Path(base).iterdir():
                b = sub / "bin"
                if b.is_dir():
                    os.add_dll_directory(str(b))
                    os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def _transcribe_child(wav: str, out_json: str, device: str, model_name: str, language: str,
                      batch_size: str = "1") -> None:
    """Runs in a separate process so a CUDA crash can't kill the main pipeline."""
    _add_nvidia_dll_dirs()
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute)
    opts = dict(language=language or None, word_timestamps=True, vad_filter=True, beam_size=5,
                condition_on_previous_text=False)
    if int(batch_size) > 1:  # GPU: decode several speech chunks at once (much faster on long videos)
        segments, info = BatchedInferencePipeline(model).transcribe(wav, batch_size=int(batch_size), **opts)
    else:
        segments, info = model.transcribe(wav, **opts)
    out = {"language": info.language, "segments": []}
    for seg in segments:
        words = [[float(w.start), float(w.end), w.word] for w in (seg.words or [])]
        out["segments"].append({"start": float(seg.start), "end": float(seg.end),
                                "text": seg.text.strip(), "words": words})
        print(f"  {fmt_ts(seg.start)}  {seg.text.strip()[:80]}", flush=True)
    Path(out_json).write_text(json.dumps(out), encoding="utf-8")


def transcribe(wav: Path, work: Path, cfg: dict) -> dict:
    out_json = work / "transcript.json"
    attempts = []
    dev = cfg["whisper_device"]
    if dev == "auto" and IS_MAC:  # no CUDA on a Mac: go straight to the CPU model
        dev = "cpu"
    bs = int(cfg.get("whisper_batch_size") or 1)
    if dev in ("auto", "cuda"):
        attempts.append(("cuda", cfg["whisper_model"], bs))
        if bs > 1:  # e.g. out of GPU memory: retry unbatched on the GPU before the slow CPU
            attempts.append(("cuda", cfg["whisper_model"], 1))
    if dev in ("auto", "cpu"):
        attempts.append(("cpu", cfg["whisper_cpu_model"], 1))
    for n, (device, model, b) in enumerate(attempts):
        log(f"Transcribing on {device.upper()} with '{model}'" + (f", batch {b}" if b > 1 else "")
            + " (first run downloads the model)...")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--_transcribe", str(wav),
               str(out_json), device, model, cfg.get("language") or "", str(b)]
        # stream the child's transcript lines (progress for the web app); UTF-8 so Hindi can't crash it
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env, text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            line = line.rstrip("\n")
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                print(line.encode("ascii", "replace").decode(), flush=True)
            if LOG_HOOK:
                LOG_HOOK(line)
        proc.wait()
        res = proc
        if res.returncode == 0 and out_json.exists():
            return json.loads(out_json.read_text(encoding="utf-8"))
        nxt = attempts[n + 1] if n + 1 < len(attempts) else None
        log(f"Transcription on {device} failed" + (f" - retrying on {nxt[0].upper()}" if nxt else ""))
    log("WARNING: no transcript - clips will be picked by audio/heatmap only, without captions.")
    return {"language": "", "segments": []}


# ----------------------------------------------------------------------------- signals
def extract_wav(src: Path, wav: Path) -> None:
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(wav)])


def loudness_per_second(wav: Path, n_sec: int) -> np.ndarray:
    """0-1 loudness per second. Reads the WAV in 5-minute chunks so a 4-hour VOD needs ~40 MB, not ~2 GB."""
    rms_parts = []
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        hop = sr // 2  # 0.5 s
        chunk = hop * 600
        while True:
            data = np.frombuffer(w.readframes(chunk), dtype=np.int16).astype(np.float32)
            n = len(data) // hop
            if n == 0:
                break
            frames = data[: n * hop].reshape(n, hop)
            rms_parts.append(np.sqrt((frames ** 2).mean(axis=1)))
    if not rms_parts:
        return np.zeros(n_sec)
    rms = np.concatenate(rms_parts) + 1e-6
    db = 20 * np.log10(rms)
    lo, hi = np.percentile(db, 10), np.percentile(db, 99)
    norm = np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1)
    per_sec = norm[: (len(norm) // 2) * 2].reshape(-1, 2).mean(axis=1)
    out = np.zeros(n_sec)
    out[: min(n_sec, len(per_sec))] = per_sec[:n_sec]
    return out


def spike_per_second(loud: np.ndarray, window: int = 30) -> np.ndarray:
    """How much louder each second is than the ~30 s before it (0-1).
    Game audio (gunfire, music) is loud all the time, so absolute loudness can't tell a peak from
    normal play. A peak is a JUMP: a scream after quiet, a sudden explosion, chat going off."""
    n = len(loud)
    if n == 0:
        return loud
    c = np.concatenate([[0.0], np.cumsum(loud)])
    idx = np.arange(n)
    a = np.maximum(0, idx - window)
    base = np.where(idx > a, (c[idx] - c[a]) / np.maximum(idx - a, 1), loud)
    # the video's typical level around here (5-min rolling median): coming back to normal volume
    # after a quiet stretch is NOT a peak, so the jump must also beat this
    step = 10
    typ_pts = [np.median(loud[max(0, i - 150):i + 150]) for i in range(0, n, step)]
    typical = np.interp(idx, np.arange(0, n, step), typ_pts)
    # strongest second in a short window, so a 1-2 s burst still counts
    peak = np.array([loud[i:i + 3].max() for i in range(n)])
    jump = np.clip((peak - np.maximum(base, typical)) / 0.30, 0, 1)
    return np.convolve(jump, np.ones(3) / 3, mode="same")


def heat_per_second(heatmap: list, n_sec: int) -> np.ndarray | None:
    if not heatmap:
        return None
    arr = np.zeros(n_sec)
    for p in heatmap:
        s, e, v = p.get("start_time", 0), p.get("end_time", 0), p.get("value", 0)
        arr[int(s): max(int(s) + 1, int(np.ceil(e)))] = v
    mx = arr.max()
    return arr / mx if mx > 0 else None


def hype_per_second(segments: list, n_sec: int) -> tuple[np.ndarray, np.ndarray]:
    hype = np.zeros(n_sec)
    speech = np.zeros(n_sec)
    for seg in segments:
        for s, e, word in seg["words"]:
            for t in range(int(s), min(n_sec, int(e) + 1)):
                speech[t] = 1
            w = re.sub(r"[^\wऀ-ॿ' ]", "", word.lower()).strip()
            if not w:
                continue
            if w in HYPE_WORDS or "!" in word:
                if int(s) < n_sec:
                    hype[int(s)] += 1
        text = " " + seg["text"].lower() + " "
        for phrase in HYPE_WORDS:
            if " " in phrase and phrase in text and int(seg["start"]) < n_sec:
                hype[int(seg["start"])] += 1
    if hype.max() > 0:
        hype = np.convolve(hype, np.ones(5) / 5, mode="same")
        hype = np.clip(hype / max(np.percentile(hype[hype > 0], 90), 1e-6), 0, 1)
    return hype, speech


_WORD_INDEX: dict = {"id": None, "words": [], "starts": []}


def _word_index(segments: list):
    """Flat, start-sorted word list, rebuilt only when a new transcript comes in (fast lookups on long VODs)."""
    if _WORD_INDEX["id"] != id(segments) or _WORD_INDEX.get("n") != len(segments):
        words = sorted((w for seg in segments for w in seg["words"]), key=lambda w: w[0])
        _WORD_INDEX.update(id=id(segments), n=len(segments), words=words, starts=[w[0] for w in words])
    return _WORD_INDEX["words"], _WORD_INDEX["starts"]


def words_in(segments: list, start: float, end: float) -> list:
    words, starts = _word_index(segments)
    i = bisect.bisect_left(starts, start - 0.05)
    out = []
    for w in words[i:]:
        if w[0] > end + 0.05:
            break
        if w[1] <= end + 0.05:
            out.append(w)
    return out


def text_in(segments: list, start: float, end: float) -> str:
    return "".join(w[2] for w in words_in(segments, start, end)).strip()


# ----------------------------------------------------------------------------- selection
def find_candidates(duration: float, segments: list, heat, loud, hype, speech, cfg: dict, spike=None) -> list[dict]:
    n = len(loud)
    lo, hi = float(cfg["min_clip_seconds"]), float(cfg["max_clip_seconds"])
    skip_a = min(cfg["skip_start_seconds"], duration * 0.05)
    skip_b = min(cfg["skip_end_seconds"], duration * 0.05)
    if duration < lo + 2:
        return [{"start": 0.0, "end": duration, "score": 1.0, "heat": 0, "loud": 0, "hype": 0}]

    seg_starts = [s["start"] for s in segments]
    seg_ends = [s["end"] for s in segments]

    def snap_start(t):
        for s, e in zip(seg_starts, seg_ends):
            if s <= t < e:
                return s
        return t

    def snap_end(start, target):
        best = None
        for e in seg_ends:
            if start + lo <= e <= start + hi:
                if best is None or abs(e - target) < abs(best - target):
                    best = e
        return best if best is not None else target

    has_heat = heat is not None
    if spike is None:
        spike = spike_per_second(loud)
    # loudness now = 40% absolute level + 60% sudden jump over the previous 30 s
    w_heat, w_loud, w_hype, w_speech = (0.45, 0.30, 0.15, 0.10) if has_heat else (0.0, 0.55, 0.30, 0.15)
    cs = lambda a: np.concatenate([[0], np.cumsum(a)])
    c_spike = cs(spike)
    c_heat = cs(heat) if has_heat else None
    c_loud, c_hype, c_speech = cs(loud), cs(hype), cs(speech)

    def mean(c, a, b):
        a, b = max(0, int(a)), min(n, int(np.ceil(b)))
        return (c[b] - c[a]) / max(b - a, 1)

    starts = sorted(set([round(s, 2) for s in seg_starts] + list(np.arange(0, duration, 3.0))))
    cands = []
    for s0 in starts:
        s = snap_start(s0)
        if s < skip_a or s + lo > duration - skip_b:
            continue
        for target in (lo, (lo + hi) / 2, hi):
            e = snap_end(s, s + target)
            e = min(e, duration - skip_b)
            if e - s < lo - 0.5 or e - s > hi + 0.5:
                continue
            a, b = int(s), int(np.ceil(e))
            seg_loud, seg_spike = loud[a:b], spike[a:b]
            abs_l = 0.5 * seg_loud.mean() + 0.5 * np.percentile(seg_loud, 90) if len(seg_loud) else 0
            # a clip needs ONE real peak, not loud filler: take the window's top spike moments
            spk = 0.7 * np.percentile(seg_spike, 95) + 0.3 * seg_spike.mean() if len(seg_spike) else 0
            l_score = 0.4 * abs_l + 0.6 * spk
            h = mean(c_heat, s, e) if has_heat else 0
            hy = mean(c_hype, s, e)
            sp = mean(c_speech, s, e)
            # strong opening: viewers decide in the first seconds
            hook_end = min(e, s + 5)
            hook = (0.25 * mean(c_loud, s, hook_end) + 0.35 * mean(c_spike, s, hook_end)
                    + 0.40 * (mean(c_heat, s, hook_end) if has_heat else mean(c_hype, s, hook_end)))
            score = w_heat * h + w_loud * l_score + w_hype * hy + w_speech * sp + 0.15 * hook
            txt = text_in(segments, s, e).lower() if segments else ""
            if any(p in txt for p in PROMO_WORDS):
                score -= 0.35
            cands.append({"start": round(s, 2), "end": round(e, 2), "score": float(score),
                          "heat": float(h), "loud": float(l_score), "hype": float(hy), "spike": float(spk)})
    cands.sort(key=lambda c: c["score"], reverse=True)
    return cands


def pick_non_overlapping(cands: list[dict], k: int, gap: float = 5.0) -> list[dict]:
    chosen = []
    for c in cands:
        if all(c["end"] + gap <= x["start"] or c["start"] >= x["end"] + gap for x in chosen):
            chosen.append(c)
        if len(chosen) >= k:
            break
    return chosen


_GEMINI_LOCK = threading.Lock()
_GEMINI_LAST = [0.0]
GEMINI_MIN_INTERVAL = 60 / 14   # free tier allows 15 requests/min: stay just under, even from parallel threads
GEMINI_USED: set[str] = set()   # models that answered during this run


def gemini_json(client, cfg: dict, prompt, temperature: float, low_media: bool = False):
    """generate_content with JSON output. 429 (quota) -> next model at once; 503/timeout -> wait and retry."""
    from google.genai import types

    models = [cfg["gemini_model"]] + [m for m in cfg.get("gemini_fallback_models", []) if m != cfg["gemini_model"]]
    err = None
    for model in models:
        for attempt in range(3):
            with _GEMINI_LOCK:  # one call at a time, spaced out: calls are queued, never burst
                wait = _GEMINI_LAST[0] + GEMINI_MIN_INTERVAL - time.time()
                if wait > 0:
                    time.sleep(wait)
                _GEMINI_LAST[0] = time.time()
            try:
                gc = dict(response_mime_type="application/json", temperature=temperature)
                if low_media:  # video: ~100 tokens/s instead of ~300, plenty to see a play
                    gc["media_resolution"] = types.MediaResolution.MEDIA_RESOLUTION_LOW
                resp = client.models.generate_content(
                    model=model, contents=prompt, config=types.GenerateContentConfig(**gc),
                )
                data = json.loads(resp.text)
                GEMINI_USED.add(model)
                log(f"  Gemini answered with {model}")
                return data
            except Exception as e:
                err = e
                msg = str(e)
                log(f"  Gemini {model} attempt {attempt + 1}: {msg[:70]}")
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    break  # quota used up for this model: retrying won't help, try the next one
                busy = any(c in msg for c in ("503", "UNAVAILABLE", "timed out", "Timeout"))
                if busy and attempt < 2:
                    time.sleep(5 if attempt == 0 else 20)
                    continue
                break
    raise err or RuntimeError("no Gemini model configured")


VISION_PAD = 3.0  # seconds of context before/after each candidate in its preview


def make_preview(src: Path, c: dict, out: Path, cfg: dict, duration: float) -> float:
    """Small, cheap copy of one candidate for Gemini to watch. Returns where the preview starts."""
    s = max(0.0, c["start"] - VISION_PAD)
    e = min(duration, c["end"] + VISION_PAD)
    h = int(cfg.get("vision_preview_height") or 360)
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{s:.2f}", "-i", str(src), "-t", f"{e - s:.2f}",
         "-vf", f"scale=-2:{h},fps=2", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "32",
         "-c:a", "aac", "-b:a", "48k", "-ac", "1", "-ar", "22050", "-movflags", "+faststart", str(out)])
    return s


def gemini_watch(pool: list[dict], src: Path, segments: list, meta: dict, cfg: dict, work: Path) -> int:
    """Gemini WATCHES every candidate (picture + sound) and scores it 0-10 as a standalone Short.
    Adds watched_score / what_happens / peak_at to each candidate. Returns how many were scored.
    Any failure just leaves candidates unscored - the text-only pick still runs."""
    key = cfg.get("gemini_api_key")
    if not key or not cfg.get("vision_review", True) or not pool:
        return 0
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return 0
    client = genai.Client(api_key=key, http_options={"timeout": 180_000})
    pdir = work / "previews"
    pdir.mkdir(exist_ok=True)
    camp = ""
    if cfg.get("campaign"):
        camp = "Approved formats (anything else scores 3 or less): " + "; ".join(cfg.get("approved_formats", [])) + "\n"
    batch = max(1, int(cfg.get("vision_batch") or 6))
    scored = 0
    log(f"Gemini is watching {len(pool)} candidate moments...")
    for b0 in range(0, len(pool), batch):
        group = list(enumerate(pool))[b0:b0 + batch]
        uploaded, parts, starts = [], [], {}
        try:
            for i, c in group:
                f = pdir / f"cand_{i:02d}.mp4"
                starts[i] = make_preview(src, c, f, cfg, meta["duration"])
                up = client.files.upload(file=str(f), config={"mime_type": "video/mp4"})
                uploaded.append(up)
            # uploaded videos must finish processing before they can be used
            for _ in range(60):
                states = [client.files.get(name=u.name).state.name for u in uploaded]
                if all(st == "ACTIVE" for st in states):
                    break
                if any(st == "FAILED" for st in states):
                    raise RuntimeError("a preview failed processing")
                time.sleep(2)
            else:
                raise RuntimeError("previews still processing after 2 min")
            for (i, c), up in zip(group, uploaded):
                lead = c["start"] - starts[i]
                parts.append(types.Part.from_text(text=(
                    f"CANDIDATE id={i}: the clip itself starts {lead:.0f} s into this preview and lasts "
                    f"{c['end'] - c['start']:.0f} s. Transcript: "
                    + text_in(segments, c['start'], c['end'])[:600])))
                parts.append(types.Part.from_uri(file_uri=up.uri, mime_type="video/mp4"))
            parts.append(types.Part.from_text(text=f"""You are the toughest editor of a gaming Shorts channel called
"Only the peak moments". You just watched {len(group)} candidate moments from "{meta['title']}" by {meta['channel']}.
Judge what you SEE and HEAR, not just the words: kills, clutches, fails, jump scares, rage, big reactions.
{camp}Score each 0-10 as a standalone YouTube Short:
 9-10: a stranger would stop scrolling and rewatch it; clear payoff, no context needed
 6-8: good moment, slightly slow or needs a little context
 3-5: ordinary gameplay or talking
 0-2: menus, loading, idle, sponsor/giveaway talk, nothing happens
Be strict: most candidates are NOT a 9.
Return JSON: {{"clips": [{{"id": int, "score": int, "what_happens": str (max 20 words, what a viewer sees),
"peak_at": float (seconds from the start of the CLIP where the payoff happens, -1 if none),
"needs_context": bool}}]}}"""))
            data = gemini_json(client, cfg, parts, 0.2, low_media=True)
            rows = data["clips"] if isinstance(data, dict) else data
            ids = {i for i, _ in group}
            for r in rows:
                try:
                    i = int(r["id"])
                    if i not in ids:
                        continue
                    pool[i]["watched_score"] = max(0, min(10, int(round(float(r.get("score", 0))))))
                    pool[i]["what_happens"] = str(r.get("what_happens", ""))[:160]
                    pool[i]["peak_at"] = float(r.get("peak_at", -1))
                    p = pool[i]
                    if 0 <= p["peak_at"] <= p["end"] - p["start"] + VISION_PAD:
                        p["payoff_abs"] = p["start"] + p["peak_at"]  # source time of the payoff
                    pool[i]["needs_context"] = bool(r.get("needs_context", False))
                    scored += 1
                except Exception:
                    continue
            log(f"  watched {min(b0 + batch, len(pool))}/{len(pool)}: " + ", ".join(
                f"#{i}={pool[i].get('watched_score', '?')}" for i, _ in group))
        except Exception as e:
            log(f"  watching batch {b0 // batch + 1} failed ({str(e)[:120]}) - those stay text-only")
        finally:
            for u in uploaded:
                try:
                    client.files.delete(name=u.name)
                except Exception:
                    pass
    shutil.rmtree(pdir, ignore_errors=True)
    return scored


def rerank_by_watch(pool: list[dict]) -> list[dict]:
    """Blend the watched score into the automatic score, so even the no-Gemini-pick fallback uses it."""
    for c in pool:
        if "watched_score" in c:
            c["score"] = 0.35 * c["score"] + 0.65 * (c["watched_score"] / 10) - (0.1 if c.get("needs_context") else 0)
    return sorted(pool, key=lambda c: c["score"], reverse=True)


def gemini_pick(pool: list[dict], segments: list, meta: dict, cfg: dict) -> list[dict] | None:
    key = cfg.get("gemini_api_key")
    if not key:
        return None
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        log("google-genai not installed - skipping AI selection")
        return None
    lo, hi, k = cfg["min_clip_seconds"], cfg["max_clip_seconds"], cfg["num_clips"]
    lang_note = ""
    if cfg.get("_lang") == "hi":
        lang_note = ("\nThe video is in Hindi/Hinglish. Write title, banner and description in Hinglish using "
                     "Roman script only (e.g. \"bhai ye kya scene hai\"), never Devanagari. Keep English words in English.\n")
    camp_note, camp_fields = "", ""
    if cfg.get("campaign"):
        formats = "\n".join(f"  - {f}" for f in cfg.get("approved_formats", []))
        camp_note = f"""
CAMPAIGN RULES ({cfg.get('campaign_name', cfg['campaign'])}) - clips outside these rules get rejected:
- Only pick moments that clearly fit one of these approved formats, nothing else:
{formats}
- Skip menus, loading screens, giveaway/key/sponsor talk, idle chatting and anything that needs context.
- Clutch plays: start right before the decisive action so the payoff comes in the first seconds.
- First impressions/hot takes: the banner states his hot take in a few words.
- banner: a short, punchy gaming hook of at most 6 words, like "HE DID NOT SEE THAT COMING" or
  "WORST TEAMMATE EVER". Never a sentence, explanation or description.
- format: which approved format the clip is. game: the game being played (from the transcript,
  the video title or game_hint), e.g. "Bodycam".
"""
        camp_fields = ', "format": str, "game": str'
    watch_note = ("" if not any("watched_score" in c for c in pool) else
                  '\n"watched_score" (0-10) and "what_happens" come from an editor who WATCHED each moment. '
                  'They are the strongest signal: never pick a watched_score below 6 if better ones exist, and '
                  'prefer the highest. "payoff_at" is when the payoff happens (seconds in the source): the clip '
                  'must contain it, and for clutch plays start 3-6 s before it so it lands in the first seconds.')
    heat_note = ("" if any(c["heat"] for c in pool) else
                 '\n"most_replayed" is not available for this source (always 0); rely on loudness and the transcript.')
    items = []
    for i, c in enumerate(pool):
        item = {
            "id": i, "start": c["start"], "end": c["end"],
            "most_replayed": round(c["heat"], 2), "loudness": round(c["loud"], 2),
            "transcript": text_in(segments, max(0, c["start"] - 10), c["end"] + 10)[:1500],
        }
        if "watched_score" in c:  # Gemini already watched this one
            item["watched_score"] = c["watched_score"]
            item["what_happens"] = c.get("what_happens", "")
            if c.get("peak_at", -1) >= 0:
                item["payoff_at"] = round(c["start"] + c["peak_at"], 1)
            if c.get("needs_context"):
                item["needs_context"] = True
        if chapter_at(meta, c["start"]):  # Twitch only
            item["game_hint"] = chapter_at(meta, c["start"])
        items.append(item)
    prompt = f"""You are an expert short-form video editor for a gaming clips channel.
Source video: "{meta['title']}" by {meta['channel']}.
Below are candidate moments (seconds). "most_replayed" (0-1) is YouTube's real replay heatmap;
"loudness" (0-1) is how loud/hype the audio is. Transcript covers 10 s before and after each candidate.{heat_note}{watch_note}
{camp_note}

Pick the {k} BEST moments that work as standalone Shorts: a genuine funny, hype, clutch, rage,
or shocking moment that makes sense without context. Prefer high most_replayed. Skip sponsor reads,
intros, outros and slow talking. You may move start/end by up to 10 s so the clip opens on a strong
hook (no dead air) and ends right after the payoff. Every clip must be {lo}-{hi} seconds long and
clips must not overlap.

For each clip write:
- title: <= 70 chars, curiosity/hype, true to what happens, no fake claims
- banner: <= 6 words, shown on screen as the hook (e.g. "HE DID NOT SEE THIS COMING")
- description: 1-2 short sentences about the moment
- hashtags: 5 relevant hashtags without spaces
- reason: why this moment is strong
- punch_word: ONE word (max 12 letters, English, no swearing) that sums up the reaction, shown huge
  behind the streamer's head when he reacts, e.g. "SPIDER", "CLUTCH", "BETRAYED", "JUMPSCARE"
{lang_note}
Return JSON: {{"clips": [{{"id": int, "start": float, "end": float, "title": str, "banner": str,
"description": str, "hashtags": [str], "reason": str, "punch_word": str{camp_fields}}}]}} ordered best first.

Candidates:
{json.dumps(items, ensure_ascii=False)}"""
    try:
        client = genai.Client(api_key=key, http_options={"timeout": 90_000})  # ms; never hang forever
        data = gemini_json(client, cfg, prompt, 0.4)
        clips = data["clips"] if isinstance(data, dict) else data
        log(f"  Gemini returned {len(clips)} picks: " + ", ".join(
            f"id {c.get('id')} {float(c.get('start', 0)):.0f}-{float(c.get('end', 0)):.0f}s" for c in clips))
    except Exception as e:
        log(f"Gemini step failed ({str(e)[:200]}) - using automatic picks")
        return None

    out = []
    dur = meta["duration"]
    for c in clips:
        try:
            base = pool[int(c["id"])]
            s = max(0.0, min(float(c.get("start", base["start"])), dur - lo))
            e = float(c.get("end", base["end"]))
            if abs(s - base["start"]) > 12:  # model drifted off its candidate - don't trust it
                log(f"  Gemini pick id {c['id']} dropped: start {s:.0f}s is {abs(s - base['start']):.0f}s from its candidate")
                continue
            e = min(max(e, s + lo), s + hi, dur)
            s, e = refine_cut(s, e, segments, lo, hi, dur, base.get("payoff_abs"))
            if e - s < min(lo, dur) - 0.5:
                log(f"  Gemini pick id {c['id']} dropped: too short ({e - s:.0f}s)")
                continue
            if any(not (e + 1 <= x["start"] or s >= x["end"] + 1) for x in out):
                log(f"  Gemini pick id {c['id']} dropped: overlaps an earlier pick ({s:.0f}-{e:.0f}s)")
                continue
            out.append({**base, "start": round(s, 2), "end": round(e, 2),
                        "title": str(c.get("title", ""))[:100], "banner": str(c.get("banner", ""))[:60],
                        "description": str(c.get("description", "")),
                        "hashtags": [str(h).lstrip("#").replace(" ", "") for h in c.get("hashtags", [])][:8],
                        "reason": str(c.get("reason", "")), "punch_word": str(c.get("punch_word", ""))[:20],
                        "format": str(c.get("format", "")), "game": str(c.get("game", ""))})
            if cfg.get("campaign"):  # hard cap: banners are hooks, not sentences
                out[-1]["banner"] = " ".join(out[-1]["banner"].split()[:6])
        except Exception as ex:
            log(f"  Gemini pick dropped: unreadable ({str(ex)[:60]})")
            continue
        if len(out) >= k:
            break
    return out or None


SENTENCE_END = re.compile(r"[.!?।…]['\")\]]*$")


def refine_cut(s: float, e: float, segments: list, lo: float, hi: float, dur: float,
               payoff: float | None = None) -> tuple[float, float]:
    """Move a cut so it never splits a word, ends on a finished sentence when one is close, and keeps the
    payoff (if known) inside the clip; length stays within lo..hi. Clips without speech are left alone,
    apart from the payoff and length rules."""
    words, starts = _word_index(segments) if segments else ([], [])
    ends_seg = {round(seg["end"], 2) for seg in segments} if segments else set()

    def word_at(t):  # the word being spoken at time t, if any
        i = bisect.bisect_right(starts, t) - 1
        return words[i] if i >= 0 and words[i][0] < t < words[i][1] else None

    def sentence_final(w):
        return bool(SENTENCE_END.search(w[2].strip())) or round(w[1], 2) in ends_seg

    # 1. the payoff must be inside, with a beat after it to land
    if payoff is not None and 0 <= payoff <= dur:
        if payoff > e - 1.5:
            e = min(dur, payoff + 2.5)
            s = max(s, e - hi)
        if payoff < s + 0.3:
            s = max(0.0, payoff - 3.0)
            e = min(e, s + hi)
    # 2. start: never mid-word (step back to the word's start, leaving a 0.1 s breath)
    w = word_at(s)
    if w:
        i = bisect.bisect_left(starts, w[0])
        prev_end = words[i - 1][1] if i > 0 else 0.0
        s = max(prev_end, w[0] - 0.1, 0.0)
    def after(i):  # cut point just after word i: a short tail, but never into the next word
        nxt = words[i + 1][0] if i + 1 < len(words) else dur
        return min(words[i][1] + 0.15, max(words[i][1], nxt - 0.02))

    def last_word_before(t):
        return bisect.bisect_right(starts, t) - 1

    # 3. end: never mid-word; run on to the end of the sentence when it finishes within 3 s and the cap
    if words:
        i = last_word_before(e)
        if i >= 0 and words[i][1] > e - 0.02:          # cut falls inside (or right at the end of) word i
            e = after(i)
        if i >= 0 and not sentence_final(words[i]):
            j = i + 1
            while j < len(words) and words[j][0] - e < 3.0 and after(j) <= s + hi:
                if sentence_final(words[j]):
                    e = after(j)
                    break
                j += 1
    # 4. too long: end on the last sentence that fits (else the last whole word); too short: run on
    if e - s > hi:
        e = s + hi
        if words:
            fits = [i for i in range(max(0, last_word_before(s)), last_word_before(e) + 1)
                    if words[i][0] >= s and after(i) <= s + hi]
            final = [i for i in fits if sentence_final(words[i]) and after(i) >= s + lo]
            if final or fits:
                e = max(s + lo, after((final or fits)[-1]))
    if e - s < lo:
        e = s + lo
        w = word_at(e)
        if w:
            i = bisect.bisect_left(starts, w[0])
            if after(i) - s <= hi:
                e = after(i)
    return round(max(0.0, s), 2), round(min(dur, e), 2)


def romanize_words(words: list, cfg: dict, batch: int = 120) -> list:
    """Transliterate Devanagari words to Roman Hinglish via Gemini, keeping count and timings."""
    key = cfg.get("gemini_api_key")
    if not key or not any(DEVANAGARI.search(w[2]) for w in words):
        return words
    try:
        from google import genai
        client = genai.Client(api_key=key, http_options={"timeout": 90_000})  # ms; never hang forever
    except Exception as e:
        log(f"Hinglish captions unavailable ({str(e)[:120]}) - keeping Devanagari")
        return words
    out = [list(w) for w in words]
    for i in range(0, len(words), batch):
        part = [w[2].strip() for w in words[i:i + batch]]
        prompt = f"""Transliterate these Hindi/Hinglish caption words into Roman-script Hinglish, the way
Indian gamers type in chats (e.g. "भाई ये क्या सीन है" -> "bhai ye kya scene hai").
Rules: return exactly {len(part)} items in the same order, one output item per input item, never merge
or split items. English words, and English words written in Devanagari (e.g. "गेम" -> "game"), become the
normal English spelling. Words already in Roman script stay unchanged. Keep any punctuation attached.
Return JSON: {{"words": [str]}}

Input: {json.dumps(part, ensure_ascii=False)}"""
        try:
            data = gemini_json(client, cfg, prompt, 0.0)
            roman = data["words"] if isinstance(data, dict) else data
            if len(roman) != len(part):
                raise ValueError(f"got {len(roman)} words for {len(part)}")
        except Exception as e:
            log(f"Hinglish transliteration failed ({str(e)[:120]}) - keeping Devanagari for these words")
            continue
        for j, r in enumerate(roman):
            orig = words[i + j][2]
            r = str(r).strip()
            if r and DEVANAGARI.search(orig):  # only touch Devanagari words; English stays as spoken
                lead = orig[: len(orig) - len(orig.lstrip())]
                out[i + j][2] = lead + r
    return out


def chapter_at(meta: dict, t: float) -> str:
    """Twitch VOD chapters are the game being played; t is in clip/file time."""
    if not meta.get("twitch"):
        return ""
    t += meta.get("offset", 0)
    for ch in meta.get("chapters") or []:
        if ch.get("start_time", 0) <= t < ch.get("end_time", float("inf")):
            return ch.get("title") or ""
    return ""


def src_time(meta: dict, t: float) -> float:
    """File time -> time in the original video (the --section offset added; live mode: the stream time)."""
    if callable(meta.get("_abs")):
        return float(meta["_abs"](t))
    return t + meta.get("offset", 0)


def source_link(meta: dict, t: float) -> str:
    """Link to the original video at time t (file time; the --section offset is added)."""
    if callable(meta.get("_link")):   # live mode: VOD link when known, else the stream time
        return meta["_link"](t)
    t = int(t + meta.get("offset", 0))
    link = meta["url"]
    if meta.get("id") and "youtu" in link:
        return f"https://www.youtube.com/watch?v={meta['id']}&t={t}s"
    m = re.search(r"twitch\.tv/videos/(\d+)", link)
    if m:
        return f"https://www.twitch.tv/videos/{m.group(1)}?t={t // 3600}h{t % 3600 // 60}m{t % 60}s"
    return link


def fallback_text(clip: dict, segments: list, meta: dict) -> None:
    txt = ("".join(w[2] for w in clip["words"]).strip() if clip.get("words")
           else text_in(segments, clip["start"], clip["end"]))
    first = re.split(r"(?<=[.!?])\s+", txt)[0] if txt else ""
    first = first.strip().strip('"')
    if 8 <= len(first) <= 70:
        title = first
    elif first:
        title = first[:67].rsplit(" ", 1)[0] + "..."
    else:
        title = f"{meta['channel']} moment"
    clip.setdefault("title", title)
    clip.setdefault("banner", "")
    clip.setdefault("description", f"Best moment from {meta['channel']}'s video.")
    clip.setdefault("hashtags", ["gaming", "shorts", "gamingclips", slugify(meta["channel"], 25).lower(), "viral"])
    clip.setdefault("reason", f"auto score {clip['score']:.2f} (replay {clip['heat']:.2f}, loudness {clip['loud']:.2f})")


# ----------------------------------------------------------------------------- captions
def ass_time(t: float) -> str:
    t = max(0.0, t)
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def clean_word(w: str, upper: bool = True) -> str:
    w = w.strip().replace("{", "").replace("}", "").replace("\\", "")
    w = re.sub(r"[\"“”,.;:।॥]", "", w)
    return w.upper() if upper else w


def chunk_words(words: list, per: int) -> list[list]:
    chunks, cur = [], []
    for w in words:
        if cur and (len(cur) >= per or w[0] - cur[-1][1] > 0.6 or re.search(r"[.!?।]$", cur[-1][2].strip())):
            chunks.append(cur)
            cur = []
        if clean_word(w[2]):
            cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def write_captions(clip: dict, segments: list, cfg: dict, ass_path: Path, srt_path: Path) -> None:
    s0, s1 = clip["start"], clip["end"]
    dur = s1 - s0
    layout = cfg.get("_layout") or cfg["layout"]
    blur = layout == "blur"
    cap_margin = 420 if blur else 560
    banner_margin = 300 if blur else 180
    if layout == "facecam":  # banner at the top of the gameplay, captions over its lower part
        cap_margin, banner_margin = 330, FC_GAME_Y + 25
    hindi = cfg.get("_lang") == "hi"
    font = cfg["hindi_caption_font"] if hindi else cfg["caption_font"]
    size = int(cfg["caption_size"])
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{font},{size},&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,7,3,2,60,60,{cap_margin},1
Style: Banner,{font},60,&H00000000,&H000000FF,&H00FFFFFF,&H00FFFFFF,-1,0,0,0,100,100,0,0,3,16,0,8,80,80,{banner_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    # strip ASS control characters: a "\N" or "{...}" from Gemini would otherwise become formatting codes
    banner = re.sub(r"\\[Nnh]", " ", clip.get("banner") or "")
    banner = re.sub(r"\s+", " ", re.sub(r"[{}\\]", "", banner)).strip().upper()
    if cfg.get("show_title_banner") and banner:
        secs = float(cfg.get("title_banner_seconds") or 0)
        b_end = min(dur, secs) if secs > 0 else dur   # a hook, not a label: gone once the moment starts
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(b_end)},Banner,,0,0,0,,{{\\fad(0,150)}}{banner}")

    words = [[w[0] - s0, w[1] - s0, w[2]] for w in (clip.get("words") or words_in(segments, s0, s1))]
    words = [w for w in words if w[1] > 0 and w[0] < dur]
    srt = []
    hl, white = r"{\c&H00E5FF&}", r"{\c&HFFFFFF&}"
    chunks = chunk_words(words, int(cfg["words_per_caption"]))
    for i, ch in enumerate(chunks):
        c_end = min(dur, ch[-1][1] + 0.15)
        if i + 1 < len(chunks):
            c_end = min(c_end, chunks[i + 1][0][0])
        tokens = [clean_word(w[2], upper=not hindi) for w in ch]
        if cfg.get("censor_captions", True):
            tokens = [reactfx.mask_word(t) for t in tokens]
        for j, w in enumerate(ch):
            start = max(0.0, w[0])
            end = ch[j + 1][0] if j + 1 < len(ch) else c_end
            if end <= start:
                end = start + 0.1
            txt = " ".join((hl + t + white) if k == j else t for k, t in enumerate(tokens))
            pop = r"{\fscx108\fscy108\t(0,80,\fscx100\fscy100)}" if j == 0 else ""
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Caption,,0,0,0,,{pop}{txt}")
        srt.append(f"{i + 1}\n{srt_time(max(0, ch[0][0]))} --> {srt_time(c_end)}\n{' '.join(tokens)}\n")
    ass_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    srt_path.write_text("\n".join(srt), encoding="utf-8")


# ----------------------------------------------------------------------------- render
_ENCODER: dict = {}
_ENC_LOCK = threading.Lock()
SCALE = "flags=lanczos"   # sharpest standard resampler; used for every resize of the picture


def encoder_args(kind: str, cfg: dict, fps: float) -> list[str]:
    """Video encoder settings. YouTube re-encodes every upload, so the file we hand it should be a clean,
    high-bitrate master: quality-targeted rate control, adaptive quantisation, a keyframe every 2 s."""
    gop = str(max(1, int(round(fps * 2))))
    cq = str(cfg.get("nvenc_cq", 22))
    mx = scaled_bitrate(cfg)
    m = re.match(r"^(\d+(?:\.\d+)?)([kKmM]?)$", mx)
    buf = f"{float(m.group(1)) * 2:g}{m.group(2)}" if m else mx
    if kind == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr", "-cq", cq, "-b:v", "0",
                "-maxrate", mx, "-bufsize", buf, "-spatial-aq", "1", "-temporal-aq", "1",
                "-rc-lookahead", "32", "-bf", "3", "-profile:v", "high", "-g", gop]
    if kind == "nvenc_basic":  # older GPUs/drivers that reject lookahead or AQ
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", cq, "-b:v", "0",
                "-maxrate", mx, "-bufsize", buf, "-profile:v", "high", "-g", gop]
    return ["-c:v", "libx264", "-preset", str(cfg.get("x264_preset", "slow")), "-crf", str(cfg.get("x264_crf", 20)),
            "-maxrate", mx, "-bufsize", buf, "-profile:v", "high", "-x264-params", "aq-mode=3", "-g", gop]


def pick_encoder(cfg: dict | None = None) -> str:
    """'nvenc' (GPU, high-quality settings), 'nvenc_basic' (GPU, plain settings) or 'x264' (CPU).
    Tested once per output size with a real encode at that size (e.g. 2160x3840), so a GPU that can't do it
    never fails a clip."""
    cfg = cfg or DEFAULTS
    want = str(cfg.get("encoder", "auto")).lower()
    W, H = out_size(cfg)
    slot = _enc_slot(cfg)
    with _ENC_LOCK:
        if slot not in _ENCODER:
            choice = "x264"
            if want in ("auto", "nvenc"):
                for kind in ("nvenc", "nvenc_basic"):
                    try:
                        run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r=60:d=0.5",
                             *encoder_args(kind, cfg, 60), "-f", "null", "-"])
                        choice = kind
                        break
                    except Exception:
                        continue
            _ENCODER[slot] = choice
            if choice == "x264" and want in ("auto", "nvenc") and H != BASE_H:
                log(f"The GPU encoder can't do {W}x{H} here")
            log({"nvenc": "Using NVIDIA GPU encoder (h264_nvenc, high-quality settings)",
                 "nvenc_basic": "Using NVIDIA GPU encoder (h264_nvenc, basic settings)",
                 "x264": f"Using CPU encoder (libx264, preset {cfg.get('x264_preset', 'slow')}, "
                         f"CRF {cfg.get('x264_crf', 20)})"}[choice])
        return _ENCODER[slot]


def _enc_slot(cfg: dict) -> tuple[str, int]:
    return str(cfg.get("encoder", "auto")).lower(), out_height(cfg)


TRUE_PEAK = -1.5   # dBTP ceiling (streaming platforms' recommendation)


def loudnorm_filter(src: Path, start: float, dur: float, cfg: dict) -> str:
    """Two-pass loudness: measure this clip's audio (EBU R128), then apply ONE steady gain to reach the
    target. If that gain would push peaks past the ceiling, a fast peak limiter catches just the peaks.
    No pumping (single-pass loudnorm rides the volume up and down) and it lands on target. "" for silence."""
    target = float(cfg.get("target_lufs", -14))
    base = f"loudnorm=I={target}:TP={TRUE_PEAK}:LRA=11"
    measured_i, measured_tp = measure_loudness(src, start, dur)
    if measured_i is None:
        return base  # couldn't measure: single-pass normalisation, as before
    if not (np.isfinite(measured_i) and np.isfinite(measured_tp)) or measured_i <= -69.5:
        return ""    # silence: nothing to normalise
    gain = min(24.0, target - measured_i)   # never boost a near-silent clip by more than 24 dB
    if measured_tp + gain <= TRUE_PEAK:
        return f"volume={gain:.2f}dB"
    ceiling = 10 ** ((TRUE_PEAK - 0.5) / 20)   # sample-peak limit with headroom for inter-sample peaks
    limiter = f"alimiter=limit={ceiling:.4f}:attack=3:release=60:level=0"
    # limiting the peaks takes away some loudness: measure the result and make up for it (the more the
    # limiter works, the more it takes, so repeat until it lands within 0.3 LU; each pass is ~0.3 s)
    for _ in range(3):
        after, _ = measure_loudness(src, start, dur, f"volume={gain:.2f}dB,{limiter}")
        if after is None or not np.isfinite(after) or abs(target - after) <= 0.3:
            break
        gain = min(24.0, gain + max(-3.0, min(4.0, target - after)))
    return f"volume={gain:.2f}dB,{limiter}"


def measure_loudness(src: Path, start: float, dur: float, af: str = "") -> tuple[float | None, float | None]:
    """(integrated LUFS, true peak dBTP) of the source audio in [start, start+dur], after filter `af`.
    Uses FFmpeg's EBU R128 meter (ebur128), the standard BS.1770 meter the platforms and qa_check use;
    loudnorm's built-in meter reads up to ~1 LU differently on short clips."""
    chain = (af + "," if af else "") + "aresample=48000,ebur128=peak=true"
    try:
        res = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{start:.3f}", "-i", str(src),
                              "-t", f"{dur:.3f}", "-vn", "-map", "0:a:0", "-af", chain, "-f", "null", "-"],
                             capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
        summary = res.stderr[res.stderr.rindex("Summary:"):]
        i = re.search(r"I:\s+(-?[\d.]+|-inf) LUFS", summary).group(1)
        tp = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", summary).group(1)
        return float(i), float(tp)
    except Exception:
        return None, None


def overlay_geometry(cfg: dict, layout: str) -> tuple[int, int, int] | None:
    """(width, height, y) of the overlay image in the output frame, clear of cam, gameplay and captions."""
    if not cfg.get("overlay_image"):
        return None
    W, H = out_size(cfg)
    k = H / BASE_H
    iw, ih = probe_size(Path(cfg["overlay_image"]))
    w = ev(W * float(cfg["overlay_width_pct"]) / 100)
    h = ev(w * ih / iw)
    if layout == "facecam":   # in the band between facecam and gameplay
        cam_h, band_h, _ = fc_geometry(cfg)
        h = min(h, band_h - ev(6 * k))
        w = ev(h * iw / ih) if h < round(w * ih / iw) else w
        y = cam_h + (band_h - h) // 2
    elif layout == "crop":    # bottom edge, below the captions
        y = H - h - ev(80 * k)
    else:                     # blur: below the gameplay and captions (the banner sits above the gameplay)
        y = ev(1540 * k)
    return w, h, y


def video_chain(layout: str, cfg: dict, sw: int, sh: int, fps_str: str, punch_input: int | None = None) -> str:
    """filter_complex for the picture, ending in [v0] (output size from out_size(): 1080x1920, 1440x2560 or
    2160x3840; bt709, yuv420p, at fps_str). Every position is the 1080x1920 design scaled by k = H / 1920.
    facecam: cfg["_react"] [(a, b), ...] cuts to the face full-screen for those clip seconds; punch_input is
    the ffmpeg input index of the transparent "word behind the head" video for the first of them."""
    even = lambda v: int(v) // 2 * 2
    W, H = out_size(cfg)
    head = (f"[0:v]fps={fps_str},scale=in_color_matrix=auto:out_color_matrix=bt709:out_range=tv,"
            f"format=yuv420p")
    if layout == "facecam":
        cam_h, _, game_y = fc_geometry(cfg)
        n = cfg["_cam_crop_n"]                    # trimmed cam crop, 0-1 of the frame
        cx, cy, cw, ch = even(n[0] * sw), even(n[1] * sh), even(n[2] * sw), even(n[3] * sh)
        game_h = H - game_y
        gw = min(sw, even(sh * W / game_h))
        gx = (sw - gw) // 2
        b = cfg["_cam_box_n"]                     # the whole cam (with its frame), to keep it out of the gameplay
        bx, bw = b[0] * sw, b[2] * sw
        pad = 16 * sw / 1280
        if bx + bw / 2 > sw / 2:   # cam on the right: keep it (and its frame) out of the gameplay crop
            gx = max(0, min(gx, int(bx - pad) - gw))
        else:
            gx = min(sw - gw, max(gx, int(bx + bw + pad)))
        gx = even(gx)
        react = cfg.get("_react") or []
        rc = react_crop_px(cfg, sw, sh) if react else None
        split = (f"{head},split=3[cam][game][rx];" if rc else f"{head},split=2[cam][game];")
        chain = (f"{split}"
                 f"[cam]crop={cw}:{ch}:{cx}:{cy},scale={W}:{cam_h}:{SCALE},unsharp=5:5:0.5,"
                 f"pad={W}:{H}:0:0:color={FC_BAND_COLOR}[base];"
                 f"[game]crop={gw}:{sh}:{gx}:0,scale={W}:{game_h}:force_original_aspect_ratio=increase:{SCALE},"
                 f"crop={W}:{game_h}[g];"
                 f"[base][g]overlay=0:{game_y}" + ("[vs]" if rc else "[v0]"))
        if not rc:
            return chain
        on = "+".join(f"between(t\\,{a:.3f}\\,{b:.3f})" for a, b in react)
        x, y, w, h = rc
        chain += (f";[rx]crop={w}:{h}:{x}:{y},scale={W}:{H}:{SCALE},unsharp=5:5:0.7[rz];"
                  f"[vs][rz]overlay=0:0:enable={on}")
        if punch_input is None:
            return chain + "[v0]"
        a, b = react[0]
        return (chain + f"[vr];[{punch_input}:v]format=rgba,setpts=PTS-STARTPTS+{a:.3f}/TB[pw];"
                f"[vr][pw]overlay=0:0:eof_action=pass:enable=between(t\\,{a:.3f}\\,{b:.3f})[v0]")
    if layout == "crop":
        cw = even(sh * BASE_W / BASE_H)
        if cw >= sw:   # source already 9:16 or narrower: fill the frame
            return f"{head},scale={W}:{H}:force_original_aspect_ratio=increase:{SCALE},crop={W}:{H}[v0]"
        pts = cfg.get("_track")
        x = framing.crop_x_expr(pts, sw, cw) if pts else str(even((sw - cw) / 2))
        # crop first, then a single resize: sharper and faster than enlarging the whole frame first
        return f"{head},crop={cw}:{sh}:{x}:0,scale={W}:{H}:{SCALE}[v0]"
    # blur: the full frame over a blurred, darkened copy of itself (blurred at a small fixed size whatever the
    # output size: same look, far less work), then the sharp frame on top
    z = float(cfg.get("zoom", 1.0))
    fw = ev(W * z)
    fg = (f"scale={fw}:-2:{SCALE},crop=min(iw\\,{W}):min(ih\\,{H})" if sw >= sh else
          f"scale={W}:{H}:force_original_aspect_ratio=decrease:{SCALE}")   # vertical/square source: fit it
    return (f"{head},split=2[a][b];"
            f"[a]scale=270:480:force_original_aspect_ratio=increase:flags=bilinear,crop=270:480,boxblur=6:2,"
            f"scale={W}:{H}:flags=bicubic,eq=brightness=-0.10[bg];"
            f"[b]{fg}[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[v0]")


def render_clip(src: Path, clip: dict, folder: Path, base: str, cfg: dict, audio: bool, t0: float = 0.0) -> Path:
    """Render one clip. clip start/end are analysis-file seconds; `src` begins at t0 in that timeline
    (0 for the analysis copy, the aligned start for a high-quality segment)."""
    ass_name = f"{base}.ass"
    out_name = f"{base}.mp4"
    layout = cfg.get("_layout") or cfg["layout"]
    sw, sh = probe_size(src)
    W, H = out_size(cfg)
    fps_str, fps = render_fps(cfg, probe_fps(src))
    start = max(0.0, clip["start"] - t0)
    dur = clip["end"] - clip["start"]
    ov = overlay_geometry(cfg, layout)
    punch = None
    react = cfg.get("_react") if layout == "facecam" else None
    if react and cfg.get("_punch_word"):
        rc = react_crop_px(cfg, sw, sh)
        a, b = react[0]
        if rc:
            punch = reactfx.word_overlay(src, start + a, b - a, fps_str, fps, rc, cfg["_punch_word"],
                                         folder / f".{base}_punch.mov", str(cfg.get("punch_color") or "#FFFFFF"),
                                         W=W, H=H)
            if punch:
                clip["_punch"] = f"'{cfg['_punch_word']}' " + ("behind the head" if punch["behind"] else
                                                                "in front (no person-cutout model)")
    punch_idx = (2 if ov else 1) if punch else None
    chain = video_chain(layout, cfg, sw, sh, fps_str, punch_idx)
    if ov:
        chain += f";[1:v]scale={ov[0]}:{ov[1]}:{SCALE}[ov];[v0][ov]overlay=(W-w)/2:{ov[2]}[v1]"
    # tag the frames themselves as BT.709: FFmpeg 7.1+ takes the encoder's colour info from the frames and
    # ignores -color_primaries/-color_trc, so without this newer builds write "unknown" colour tags
    chain += (f";[{'v1' if ov else 'v0'}]ass={ass_name},"
              f"setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv[vout]")
    head = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src)]
    if ov:
        head += ["-i", cfg["overlay_image"]]
    if punch:
        head += ["-i", str(punch["path"])]
    tail = ["-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-color_range", "tv"]
    if audio:  # the source's own audio only, loudness-normalised, with tiny fades so cuts don't click
        af = ",".join(f for f in (loudnorm_filter(src, start, dur, cfg), "aresample=48000",
                                  "afade=t=in:st=0:d=0.03", f"afade=t=out:st={max(0.0, dur - 0.06):.3f}:d=0.06") if f)
        tail += ["-map", "0:a:0", "-af", af, "-c:a", "aac", "-b:a", str(cfg.get("audio_bitrate", "192k")),
                 "-ar", "48000", "-ac", "2"]
    else:
        tail += ["-an"]
    tail += ["-movflags", "+faststart", out_name]
    kind = cfg.get("_force_encoder") or pick_encoder(cfg)
    cmd = head + ["-t", f"{dur:.3f}", "-filter_complex", chain, "-map", "[vout]", *encoder_args(kind, cfg, fps)] + tail
    try:
        run(cmd, cwd=folder)
    except RuntimeError:
        if not kind.startswith("nvenc"):
            raise
        log("GPU encode failed, retrying on CPU")
        with _ENC_LOCK:
            _ENCODER[_enc_slot(cfg)] = "x264"
        cmd = head + ["-t", f"{dur:.3f}", "-filter_complex", chain, "-map", "[vout]",
                      *encoder_args("x264", cfg, fps)] + tail
        run(cmd, cwd=folder)
    finally:
        if punch:
            Path(punch["path"]).unlink(missing_ok=True)
    clip["_fps"] = fps
    clip["_size"] = (W, H)
    return folder / out_name


def qa_check(path: Path, dur: float, fps: float, audio: bool, target_lufs: float = -14,
             size: tuple[int, int] = (BASE_W, BASE_H)) -> tuple[list, list]:
    """Check a finished clip. Returns (problems, notes): problems mean the file is broken and worth one
    re-render; notes are worth a look but the clip is usable. size: the expected (width, height)."""
    try:
        res = run(["ffprobe", "-v", "error", "-show_entries",
                   "stream=codec_type,width,height,avg_frame_rate:format=duration", "-of", "json", str(path)])
        info = json.loads(res.stdout)
    except Exception as e:
        return [f"unreadable file ({str(e)[:80]})"], []
    streams = info.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not v:
        return ["no video stream"], []
    problems, notes = [], []
    if (v.get("width"), v.get("height")) != tuple(size):
        problems.append(f"size {v.get('width')}x{v.get('height')}")
    got = float((info.get("format") or {}).get("duration") or 0)
    if abs(got - dur) > 0.5:
        problems.append(f"length {got:.1f}s instead of {dur:.1f}s")
    try:
        num, den = str(v.get("avg_frame_rate", "0/1")).split("/")
        vf = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        vf = 0.0
    if abs(vf - fps) > 1:
        problems.append(f"{vf:.2f} fps instead of {fps:.2f}")
    if audio and not a:
        problems.append("no audio")
    # one decoding pass for black frames and loudness
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vf", "blackdetect=d=0.5:pix_th=0.08"]
    cmd += ["-af", "ebur128"] if a else ["-an"]
    out = subprocess.run(cmd + ["-f", "null", "-"], capture_output=True, text=True, encoding="utf-8",
                         errors="replace").stderr
    black = sum(float(x) for x in re.findall(r"black_duration:\s*([\d.]+)", out))
    if black > 0.5 * max(dur, 1):
        problems.append(f"{black:.0f}s of black picture")
    elif black >= 1.5:
        notes.append(f"{black:.1f}s of black picture")
    if a:
        lufs = re.findall(r"I:\s+(-?[\d.]+|-inf) LUFS", out)
        level = float(lufs[-1]) if lufs and lufs[-1] != "-inf" else float("-inf")
        if level < -45:
            notes.append("the audio is (almost) silent")
        elif abs(level - target_lufs) > 3:
            notes.append(f"loudness {level:.1f} LUFS (target {target_lufs:.0f})")
    return problems, notes


def game_tag(game: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (game or "").lower().replace("&", "and"))


def write_campaign_details(clip: dict, rank: int, meta: dict, cfg: dict, path: Path) -> None:
    tags = list(cfg.get("campaign_hashtags", []))
    game = clip.get("game") or chapter_at(meta, clip["start"])
    if game_tag(game) and game_tag(game) not in tags:
        tags.append(game_tag(game))
    t0, t1 = src_time(meta, clip["start"]), src_time(meta, clip["end"])
    body = f"""TITLE
{clip['title']}

CAPTION
{clip['title']} | {cfg['required_text']}
{' '.join('#' + h for h in tags)}

SOURCE
{source_link(meta, clip['start'])}
{meta.get('_time_label', 'VOD time')}: {fmt_ts(t0)} - {fmt_ts(t1)}  ({clip['end'] - clip['start']:.0f}s)

{cfg.get('post_note', '')}

--- info (not for upload) ---
Rank: {rank} (1 = strongest)
Format: {clip.get('format') or 'auto pick (no Gemini)'}   Game: {game or 'unknown'}
Banner: {clip.get('banner', '')}
Loudness score: {clip['loud']:.2f}
Why: {clip['reason']}
{render_info(clip)}"""
    path.write_text(body, encoding="utf-8")


def render_info(clip: dict) -> str:
    """Lines for the .txt info block: what the clip was rendered from and what the quality check found."""
    out = f"Rendered: {clip.get('_rendered', '?')}\n"
    qa = clip.get("_qa") or []
    out += ("Quality check: " + "; ".join(qa) if qa else "Quality check: passed") + "\n"
    return out


def write_details(clip: dict, rank: int, meta: dict, path: Path) -> None:
    tags = " ".join("#" + h for h in clip["hashtags"])
    link = source_link(meta, clip["start"])
    credit = f"Credit: {meta['channel']}" + (f" ({meta['channel_url']})" if meta.get("channel_url") else "")
    body = f"""TITLE
{clip['title']}

DESCRIPTION
{clip['description']}

{credit}
Full video: {link}

{tags}

HASHTAGS
{tags}

--- info (not for upload) ---
Rank: {rank} (1 = strongest)
{meta.get('_time_label', 'Source time')}: {fmt_ts(src_time(meta, clip['start']))} - {fmt_ts(src_time(meta, clip['end']))}  ({clip['end'] - clip['start']:.0f}s)
Most-replayed score: {clip['heat']:.2f}   Loudness score: {clip['loud']:.2f}
Why: {clip['reason']}
{render_info(clip)}"""
    path.write_text(body, encoding="utf-8")


# ----------------------------------------------------------------------------- main flow
def make_video_folder(cfg: dict, meta: dict) -> Path:
    day = Path(cfg["output_root"]) / dt.date.today().strftime("%d-%m-%Y")
    day.mkdir(parents=True, exist_ok=True)
    taken = [int(p.name[:2]) for p in day.iterdir() if p.is_dir() and re.match(r"^\d{2}_", p.name)]
    idx = max(taken, default=0) + 1   # never reuse a number, even after a folder was deleted
    folder = day / f"{idx:02d}_{slugify(meta['channel'], 20)}_{slugify(meta['title'], 40)}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def overlaps(c: dict, others: list[dict], gap: float = 1.0) -> bool:
    return any(not (c["end"] + gap <= x["start"] or c["start"] >= x["end"] + gap) for x in others)


def _face_in(seen: list, box: tuple, sw: int, sh: int) -> tuple[float, float, float] | None:
    """Median (cx, cy, h), normalised, of the faces seen inside box (source pixels)."""
    x, y, w, h = box
    pts = [f for faces in seen for f in faces
           if x <= f[0] * sw <= x + w and y <= f[1] * sh <= y + h]
    if not pts:
        return None
    return tuple(float(np.median([f[k] for f in pts])) for k in (0, 1, 3))


def react_crop_px(cfg: dict, sw: int, sh: int) -> tuple[int, int, int, int] | None:
    """9:16 crop (x, y, w, h source pixels) of the facecam, centred on the face: the full-screen face zoom."""
    n = cfg.get("_cam_crop_n")
    if not n:
        return None
    even = lambda v: int(v) // 2 * 2
    cx, cy, cw, ch = n[0] * sw, n[1] * sh, n[2] * sw, n[3] * sh
    w, h = ch * 9 / 16, ch
    if w > cw:                     # a tall cam: use its full width instead
        w, h = cw, cw * 16 / 9
    face = cfg.get("_face_n")
    fx = face[0] * sw if face else cx + cw / 2
    fy = face[1] * sh if face else cy + ch / 2
    x = min(max(fx - w / 2, cx), cx + cw - w)
    y = min(max(fy - h * 0.42, cy), cy + ch - h)
    w, h = max(2, even(w)), max(2, even(h))
    return even(x), even(y), w, h


def _cam_plan(rank: int, clip: dict, src: Path, sw: int, sh: int, cfg: dict, box, must_find: bool = False):
    """Facecam layout settings for one clip, or None. box: the configured cam (source px) or None.
    The configured box is checked for a face; if there is none, a moved cam is searched for.
    must_find: only accept a cam found in the footage (the auto layout has no configured box)."""
    s, e = clip["start"], clip["end"]
    use_faces = cfg.get("face_tracking", True) and framing.detector().ok
    found, seen = None, []
    if use_faces:
        times = [s + 1.5, s + (e - s) * 0.25, (s + e) / 2, s + (e - s) * 0.75, e - 1.5]
        det = framing.detector()
        frames = framing.frames_at(src, times, sw, sh)
        seen = [det.detect(img) for _, img in frames]
        if not (box and framing.face_in_box(seen, box, sw, sh)):
            loc = framing.locate_facecam(seen)
            if loc:
                cand = framing.cam_box_from_face(*loc, sw, sh)
                if framing.region_is_live(frames, cand, sw, sh):   # a webcam moves; a HUD portrait doesn't
                    found = cand
    if must_find and not found:
        return None
    if found:
        if box:
            log(f"  clip {rank}: no face in the configured facecam box - found the cam at "
                f"x={found[0]}, y={found[1]} ({sw}x{sh}) and used that")
        box = found
    elif box:
        crop = fit_cam_crop(box, sw, sh)
        mid = (s + e) / 2
        if not crop or not facecam_visible(src, crop, [s + 2, mid, e - 2]):
            log(f"  clip {rank}: facecam not found in its box - using the blur layout")
            box = None
    crop = fit_cam_crop(box, sw, sh) if box else None
    if not crop:
        return None
    return {"_layout": "facecam",
            "_cam_box_n": (box[0] / sw, box[1] / sh, box[2] / sw, box[3] / sh),
            "_cam_crop_n": (crop[0] / sw, crop[1] / sh, crop[2] / sw, crop[3] / sh),
            "_face_n": _face_in(seen, crop, sw, sh)}


def plan_framing(rank: int, clip: dict, src: Path, sw: int, sh: int, cfg: dict) -> dict:
    """Per-clip layout decision, made on the analysis copy. Returns the extra render settings:
    _layout, and _track (tracked crop) or _cam_box_n/_cam_crop_n/_face_n (facecam), all in 0-1 coordinates."""
    layout = cfg["layout"]
    s, e = clip["start"], clip["end"]
    use_faces = cfg.get("face_tracking", True) and framing.detector().ok
    if layout == "facecam":
        plan = _cam_plan(rank, clip, src, sw, sh, cfg, facecam_box_px(cfg, sw, sh))
        if not plan:
            if not facecam_box_px(cfg, sw, sh) and not cfg.get("_cam_warned"):
                log("No usable facecam box for this video - using the blur layout (full frame keeps the facecam visible)")
                cfg["_cam_warned"] = True
            return {"_layout": "blur"}
        return plan
    if layout in ("auto", "crop"):
        plan, track = None, []
        if use_faces:
            track = framing.face_track(src, s, e - s, sw, sh)
            plan = framing.plan_tracked_crop(track, sw, sh, auto=(layout == "auto"))
            if plan:
                plan["points"] = framing.snap_cuts_to_shots(src, s, plan["points"], sw, sh)
        if layout == "auto":
            if plan:
                moves = (len(plan["points"]) - 1) // 2
                log(f"  clip {rank}: talking head - speaker-tracked crop ({moves} camera move{'s' * (moves != 1)})")
                return {"_layout": "crop", "_track": plan["points"]}
            mains = [max(f, key=lambda a: a[2] * a[3]) for _, f in track if f]
            if cfg.get("auto_facecam", True) and framing._is_corner_cam(mains):
                cam = _cam_plan(rank, clip, src, sw, sh, cfg, None, must_find=True)
                if cam:
                    log(f"  clip {rank}: stream with a facecam - split layout (cam on top, gameplay below)")
                    return cam
            return {"_layout": "blur"}
        return {"_layout": "crop", "_track": plan["points"] if plan else None}
    return {"_layout": layout}


def process(source: str, cfg: dict, local: bool, credit: str = "", local_title: str = "") -> Path:
    t0 = time.time()
    GEMINI_USED.clear()
    tmp_root = Path(cfg["output_root"]) / "_work"
    work = tmp_root / (dt.datetime.now().strftime("%Y%m%d_%H%M%S_") + os.urandom(2).hex())
    work.mkdir(parents=True, exist_ok=True)
    try:   # the work folder (multi-GB downloads) is removed even when something fails
        return _process(source, cfg, local, credit, local_title, work, t0)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        try:
            tmp_root.rmdir()
        except OSError:
            pass


def _process(source: str, cfg: dict, local: bool, credit: str, local_title: str, work: Path, t0: float) -> Path:
    if local:
        src = Path(source).resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        meta = {"title": local_title or src.stem, "channel": credit or "creator", "channel_url": "",
                "url": "", "id": "", "duration": probe_duration(src), "heatmap": [], "view_count": None,
                "offset": 0.0, "chapters": [], "twitch": False}
        meta.update(cfg.get("_meta") or {})   # live mode: stream url, channel link, stream-time mapping
        log(f"Using local file: {src.name}")
    else:
        log(f"Downloading: {source}")
        src, meta = download(source, work, cfg)
        if not meta["duration"]:
            meta["duration"] = probe_duration(src)
        if cfg.get("creator_name"):
            meta["channel"] = cfg["creator_name"]
        log(f"Got '{meta['title']}' by {meta['channel']} ({fmt_ts(meta['duration'])})"
            + ("  [Most-replayed heatmap found]" if meta["heatmap"] else "  [no heatmap - using audio/speech only]"))
    sw, sh = probe_size(src)
    meta["_analysis_h"], meta["_analysis_fps"] = sh, probe_fps(src)

    duration = meta["duration"]
    n_sec = int(np.ceil(duration)) + 1
    audio = has_audio(src)

    wav = work / "audio.wav"
    if audio:
        log("Extracting audio...")
        extract_wav(src, wav)
        transcript = transcribe(wav, work, cfg)
        loud = loudness_per_second(wav, n_sec)
        spike = spike_per_second(loud)
    else:
        transcript, loud = {"segments": []}, np.zeros(n_sec)
        spike = np.zeros(n_sec)
    segments = transcript["segments"]
    if not cfg.get("language"):  # auto: follow what Whisper detected
        cfg["_lang"] = transcript.get("language") or ""
    heat = heat_per_second(meta["heatmap"], n_sec)
    hype, speech = hype_per_second(segments, n_sec)

    log("Scoring moments...")
    cands = find_candidates(duration, segments, heat, loud, hype, speech, cfg, spike)
    k = int(cfg["num_clips"])
    lo, hi = float(cfg["min_clip_seconds"]), float(cfg["max_clip_seconds"])
    # live mode: moments already clipped from an earlier (overlapping) window are off limits
    avoid = [{"start": float(a), "end": float(b)} for a, b in (cfg.get("_avoid") or [])]
    if avoid:
        cands = [c for c in cands if not overlaps(c, avoid)]
        log(f"Skipping {len(avoid)} moment(s) already clipped from the previous window")
    pool = pick_non_overlapping(cands, max(int(cfg.get("ai_pool_size") or 24), 2 * k))
    if gemini_watch(pool, src, segments, meta, cfg, work):
        pool = rerank_by_watch(pool)
    clips = gemini_pick(pool, segments, meta, cfg) if pool else None
    if clips and avoid:   # Gemini may move a cut by a few seconds: re-check against what was already clipped
        clips = [c for c in clips if not overlaps(c, avoid)] or None
    if clips:
        log(f"Gemini chose {len(clips)} clips")
    else:
        clips = []
        if not cfg.get("gemini_api_key"):
            log("No Gemini key - using automatic picks and simple titles")
    for c in pool:   # automatic picks (or top-up), with the same clean-cut rules as Gemini's picks
        if len(clips) >= k:
            break
        c = dict(c)
        c["start"], c["end"] = refine_cut(c["start"], c["end"], segments, lo, hi, duration, c.get("payoff_abs"))
        if not overlaps(c, clips) and not overlaps(c, avoid):
            clips.append(c)
    roman = cfg.get("_lang") == "hi" and cfg.get("hinglish_roman_captions")
    if roman and not cfg.get("gemini_api_key"):
        log("No Gemini key - Hindi captions stay in Devanagari")
    for c in clips:
        if roman:  # before fallback_text, so fallback titles are Hinglish too
            c["words"] = romanize_words(words_in(segments, c["start"], c["end"]), cfg)
        fallback_text(c, segments, meta)
        if not c.get("banner"):
            c["banner"] = ""
    if not clips:
        if cfg.get("_allow_empty"):   # live window with nothing new: normal, not an error
            log("No new moments in this part")
            return None
        raise RuntimeError("Video too short to make clips with the current length settings.")

    # high-quality footage for just these clips, aligned to the analysis timeline by their audio
    hq: dict[int, tuple[Path, float]] = {}
    if not local and audio:
        for i, seg in fetch_hq_segments(clips, meta, work, cfg).items():
            nominal = clips[i]["start"] - HQ_PAD
            at = align_segment(seg, wav, max(0.0, nominal), work)
            seg_dur = probe_duration(seg)
            if at is None or clips[i]["start"] - at < 0 or clips[i]["end"] - at > seg_dur - 0.05:
                log(f"  clip {i + 1}: couldn't line up the high-quality footage - using the analysis copy")
                continue
            hq[i] = (seg, at)
        if hq:
            log(f"High-quality footage ready for {len(hq)} of {len(clips)} clips")

    if cfg.get("_folder"):   # live mode: one folder for the whole stream, clips numbered on from earlier windows
        folder = Path(cfg["_folder"])
        folder.mkdir(parents=True, exist_ok=True)
    else:
        folder = make_video_folder(cfg, meta)
    first_no = int(cfg.get("_first_number") or 1)
    W, H = out_size(cfg)
    log(f"Rendering {len(clips)} clips -> {folder}  ({W}x{H}, "
        + (f"{cfg.get('output_fps')} fps" if str(cfg.get("output_fps") or "source") != "source" else "source fps")
        + f", max {scaled_bitrate(cfg)}bps)")
    summary = [f"{meta['title']}  —  {meta['channel']}", meta["url"], ""]
    pick_encoder(cfg)

    def render_one(rank: int, clip: dict) -> str:
        ccfg = {**cfg, **plan_framing(rank, clip, src, sw, sh, cfg)}  # per clip: renders run in parallel
        if ccfg["_layout"] == "facecam" and cfg.get("react_zoom", True):
            if cfg.get("overlay_image"):   # campaign overlay sits mid-frame: it would cover the face zoom
                if not cfg.get("_react_warned"):
                    log("Face zooms are off for clips with an overlay image (it would sit on the face)")
                    cfg["_react_warned"] = True
            else:
                ccfg["_react"] = reactfx.pick_reactions(
                    spike, loud, hype, clip["start"], clip["end"], int(cfg.get("react_max", 2)),
                    float(cfg.get("react_seconds", 2.0)), float(cfg.get("react_min_score", 0.35)))
                if ccfg["_react"] and cfg.get("punch_word", True):
                    a, b = ccfg["_react"][0]
                    rel = [[w[0] - clip["start"], w[1] - clip["start"], w[2]] for w in
                           (clip.get("words") or words_in(segments, clip["start"], clip["end"]))]
                    ccfg["_punch_word"] = reactfx.pick_punch_word(clip.get("punch_word", ""), rel, a, b)
                if ccfg["_react"]:
                    log(f"  clip {rank}: {len(ccfg['_react'])} face zoom(s) at "
                        + ", ".join(f"{a:.1f}s" for a, _ in ccfg["_react"])
                        + (f", word '{ccfg['_punch_word']}'" if ccfg.get("_punch_word") else ""))
        clip["_number"] = number = first_no + rank - 1
        base = f"{number:02d}_{slugify(clip['title'], 45)}"
        write_captions(clip, segments, ccfg, folder / f"{base}.ass", folder / f"{base}.srt")
        r_src, r_t0 = hq.get(rank - 1, (src, 0.0))
        r_audio = has_audio(r_src)
        try:
            out = render_clip(r_src, clip, folder, base, ccfg, r_audio, r_t0)
        except Exception as ex:   # e.g. an FFmpeg build that can't decode the HQ codec: never lose the clip
            if r_src == src:
                raise
            log(f"  clip {rank}: rendering from the high-quality footage failed ({str(ex).strip()[:120]}) "
                "- using the first download")
            r_src, r_t0, r_audio = src, 0.0, audio
            out = render_clip(r_src, clip, folder, base, ccfg, r_audio, r_t0)
        dur = clip["end"] - clip["start"]
        size = out_size(cfg)
        problems, notes = (qa_check(out, dur, clip["_fps"], r_audio, float(cfg.get("target_lufs", -14)), size)
                           if cfg.get("qa_check", True) else ([], []))
        if problems:   # broken file: one more try on the CPU encoder from the analysis copy
            log(f"  clip {rank}: quality check failed ({'; '.join(problems)}) - re-rendering")
            r_src, r_t0, r_audio = src, 0.0, audio
            out = render_clip(r_src, clip, folder, base, {**ccfg, "_force_encoder": "x264"}, r_audio, r_t0)
            problems, notes = qa_check(out, dur, clip["_fps"], r_audio, float(cfg.get("target_lufs", -14)), size)
        h = probe_size(r_src)[1]
        where = "high-quality download" if r_src != src else ("original file" if local else "first download")
        fps_mode = str(cfg.get("output_fps") or "source")
        clip["_rendered"] = (f"{size[0]}x{size[1]} at {clip['_fps']:.2f} fps"
                             f" ({'source frame rate' if fps_mode == 'source' else fps_mode + ' fps chosen'}); "
                             f"{ccfg['_layout']} layout, from the {where} ({h}p)")
        if ccfg.get("_react"):
            clip["_rendered"] += (f"; {len(ccfg['_react'])} face zoom(s)"
                                  + (f", punch word {clip['_punch']}" if clip.get("_punch") else ""))
        clip["_qa"] = problems + notes
        if problems or notes:
            log(f"  clip {rank}: quality check: {'; '.join(problems + notes)}")
        if cfg.get("campaign"):
            write_campaign_details(clip, rank, meta, ccfg, folder / f"{base}.txt")
        else:
            write_details(clip, rank, meta, folder / f"{base}.txt")
        (folder / f"{base}.ass").unlink(missing_ok=True)
        log(f"  {rank}/{len(clips)} done: {clip['title']}")
        flag = "  [CHECK: " + "; ".join(problems) + "]" if problems else ""
        return (f"{number:02d}. [{fmt_ts(src_time(meta, clip['start']))}-{fmt_ts(src_time(meta, clip['end']))}] "
                f"{clip['title']}{flag}")

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(1, int(cfg.get("render_workers") or 1))) as ex:
        summary += list(ex.map(lambda a: render_one(*a), enumerate(clips, 1)))
    mode = "a" if cfg.get("_folder") else "w"   # live: every window adds to the stream's summary/transcript
    with open(folder / "_summary.txt", mode, encoding="utf-8") as f:
        f.write("\n".join(summary) + "\n\n")
    if segments:
        with open(folder / "_transcript.txt", mode, encoding="utf-8") as f:
            f.write("\n".join(f"[{fmt_ts(src_time(meta, s['start']))}] {s['text']}" for s in segments) + "\n")
    if isinstance(cfg.get("_made"), list):   # live mode: which moments this window clipped (file time)
        cfg["_made"].extend({"start": c["start"], "end": c["end"], "title": c["title"],
                             "number": c.get("_number")} for c in clips)

    if cfg.get("keep_source") and not local:
        shutil.move(str(src), str(folder / f"_source{src.suffix}"))
    log("Gemini model used: " + (", ".join(sorted(GEMINI_USED)) if GEMINI_USED else "none (automatic picks)"))
    log(f"Finished in {(time.time() - t0) / 60:.1f} min -> {folder}")
    return folder


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--_transcribe":
        _transcribe_child(*sys.argv[2:8])
        return
    ap = argparse.ArgumentParser(description="YouTube/Twitch link -> best moments as vertical captioned clips")
    ap.add_argument("urls", nargs="*", help="YouTube or Twitch (VOD / clip) link(s)")
    ap.add_argument("--list", help="text file with one link per line")
    ap.add_argument("--file", help="use a local video file instead of a link")
    ap.add_argument("--credit", default="", help="creator name for a local file")
    ap.add_argument("--title", default="", help="video title for a local file")
    ap.add_argument("--out", help="output folder (overrides config.json)")
    ap.add_argument("--clips", type=int, help="number of clips (default 5)")
    ap.add_argument("--lang", choices=["en", "hi", "auto"],
                    help="spoken language (default: config 'language', normally en)")
    ap.add_argument("--verbose", action="store_true", help="show yt-dlp debug output")
    ap.add_argument("--campaign", help="campaign preset from campaigns.json (e.g. skiimasktimmy)")
    ap.add_argument("--overlay-image", help="PNG to burn on every clip (transparency kept)")
    ap.add_argument("--section", help="only download/process this part, e.g. 0:30:00-2:00:00")
    ap.add_argument("--find-facecam", metavar="URL", help="save facecam_check.png (frame ~10 min in + pixel grid)")
    ap.add_argument("--facecam-at", metavar="TIME", help="with --find-facecam: frame time instead of 0:10:00")
    ap.add_argument("--live", metavar="URL", help="clip a Twitch/YouTube stream while it's live (--clips = clips "
                                                  "per 10-minute window, default 2; Ctrl+C stops)")
    args = ap.parse_args()

    check_tools()
    cfg = load_config(args)
    if args.find_facecam:
        find_facecam(args.find_facecam, cfg, parse_time(args.facecam_at) if args.facecam_at else 600.0)
        return
    if args.live:
        import live
        try:
            live.run_terminal(args.live, cfg, args.clips or int(cfg.get("live_clips_per_window") or 2))
        except Exception as e:
            log(f"FAILED: {args.live}\n{e}")
            sys.exit(1)
        return
    urls = list(args.urls)
    if args.list:
        urls += [l.strip() for l in Path(args.list).read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")]
    if not urls and not args.file:
        u = input("Paste YouTube or Twitch link: ").strip()
        if u:
            urls.append(u)
    jobs = [(args.file, True)] if args.file else [(u, False) for u in urls]
    failed = []
    for i, (src, local) in enumerate(jobs, 1):
        log(f"=== Video {i}/{len(jobs)} ===")
        try:
            process(src, cfg, local, credit=args.credit, local_title=args.title)
        except Exception as e:
            log(f"FAILED: {src}\n{e}")
            failed.append(src)
    if failed:
        log(f"{len(failed)} video(s) failed: " + ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
