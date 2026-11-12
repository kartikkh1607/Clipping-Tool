#!/usr/bin/env python3
"""
Clipper - paste a YouTube link, get the best moments as vertical, captioned clips.

Pipeline
  1. Download the video (yt-dlp) + read YouTube's "Most replayed" heatmap
  2. Transcribe with word timestamps (faster-whisper, GPU if available)
  3. Score every possible window: most-replayed heat + loudness/hype + speech
  4. (Optional) Gemini picks the final clips, tightens the cut, writes titles
  5. Render 1080x1920 clips with word-highlight captions and a hook banner
  6. Save to  <output_root>/DD-MM-YYYY/NN_<channel>_<video>/

Usage
  python clipper.py "https://www.youtube.com/watch?v=..."
  python clipper.py URL1 URL2 ...            (several videos)
  python clipper.py --list links.txt         (one link per line)
  python clipper.py --file video.mp4 --credit "CreatorName"   (local file)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.json"

DEFAULTS = {
    "output_root": "",
    "num_clips": 5,
    "min_clip_seconds": 30,
    "max_clip_seconds": 59,
    "layout": "blur",            # "blur" = full frame on blurred bg, "crop" = center crop
    "zoom": 1.25,                # blur layout: how much to enlarge the gameplay (1.0 = fit width)
    "show_title_banner": True,   # hook text box at the top of the clip
    "words_per_caption": 3,
    "caption_font": "Arial Black",
    "caption_size": 84,
    "whisper_model": "large-v3-turbo",   # GPU model
    "whisper_cpu_model": "small",        # used if the GPU fails
    "whisper_device": "auto",            # auto | cuda | cpu
    "language": "en",                    # en | hi | "" (auto-detect)
    "hindi_caption_font": "Nirmala UI",  # supports Devanagari
    "hinglish_roman_captions": True,     # hi: transliterate captions to Roman Hinglish via Gemini
    "gemini_api_key": "",                # free key: https://aistudio.google.com/apikey
    "gemini_model": "gemini-3.8-flash",
    "gemini_fallback_models": ["gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite"],
    "max_download_height": 1080,
    "cookies_from_browser": "",          # e.g. "chrome" for age-restricted videos (not used with pot_provider)
    "pot_provider": True,                # start/check the bgutil PO Token server before YouTube downloads
    "pot_provider_dir": "~/bgutil-ytdlp-pot-provider",
    "pot_provider_port": 4416,
    "skip_start_seconds": 10,            # avoid intros
    "skip_end_seconds": 20,              # avoid outros / "subscribe"
    "keep_source": False,                # keep the downloaded full video
}

PROMO_WORDS = [
    "sponsor", "sponsored", "use code", "promo code", "link in the description",
    "link in description", "subscribe", "patreon", "merch", "today's video is brought",
    "check out", "discount", "affiliate", "like and subscribe",
]
HYPE_WORDS = [
    "oh my god", "no way", "let's go", "lets go", "what", "bro", "dude", "holy",
    "insane", "crazy", "wow", "clutch", "wait", "yes", "no no", "oh no", "run",
    "help", "dead", "kill", "won", "win", "lost", "scream", "huh", "omg", "rage",
    "impossible", "unbelievable", "hahaha", "haha", "dont", "stop", "please",
    # Hindi / Hinglish (Roman + Devanagari, since Whisper writes Hindi in Devanagari)
    "bhai", "kya", "arey", "arre", "oye", "bas", "chal", "maar", "gaya", "pagal", "bachao",
    "bhaag", "abe", "yaar", "jaldi", "maaro", "khatam", "mar gaya", "kya baat hai", "oh bhai",
    "भाई", "क्या", "अरे", "ओए", "बस", "चल", "मार", "गया", "पागल", "बचाओ", "भाग", "अबे", "यार",
    "जल्दी", "मारो", "खत्म", "मर गया", "क्या बात है",
]
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


# ----------------------------------------------------------------------------- utils
def log(msg: str) -> None:
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def slugify(text: str, max_len: int = 40) -> str:
    text = re.sub(r"[^\w\sऀ-ॿ-]", "", text or "", flags=re.UNICODE)
    text = re.sub(r"[।॥]", "", text)  # danda punctuation
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    return (text[:max_len].rstrip("_")) or "video"


def fmt_ts(sec: float) -> str:
    sec = max(0, int(sec))
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


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


def download(url: str, work: Path, cfg: dict) -> tuple[Path, dict]:
    import yt_dlp

    h = int(cfg["max_download_height"])
    opts = {
        "format": f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/bv*[height<={h}]+ba/b[height<={h}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(work / "source.%(ext)s"),
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
    if cfg.get("pot_provider") and "youtu" in url:
        ensure_pot_provider(cfg)
        opts["extractor_args"] = {"youtubepot-bgutilhttp": {"base_url": [f"http://127.0.0.1:{int(cfg['pot_provider_port'])}"]}}
        if cfg.get("cookies_from_browser"):
            log("Note: cookies_from_browser is ignored while pot_provider is on")
    elif cfg.get("cookies_from_browser"):
        opts["cookiesfrombrowser"] = (cfg["cookies_from_browser"],)
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
    }
    return path, meta


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


def _transcribe_child(wav: str, out_json: str, device: str, model_name: str, language: str) -> None:
    """Runs in a separate process so a CUDA crash can't kill the main pipeline."""
    _add_nvidia_dll_dirs()
    from faster_whisper import WhisperModel

    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute)
    segments, info = model.transcribe(
        wav, language=language or None, word_timestamps=True, vad_filter=True, beam_size=5,
        condition_on_previous_text=False,
    )
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
    if dev in ("auto", "cuda"):
        attempts.append(("cuda", cfg["whisper_model"]))
    if dev in ("auto", "cpu"):
        attempts.append(("cpu", cfg["whisper_cpu_model"]))
    for device, model in attempts:
        log(f"Transcribing on {device.upper()} with '{model}' (first run downloads the model)...")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--_transcribe", str(wav),
               str(out_json), device, model, cfg.get("language") or ""]
        res = subprocess.run(cmd)
        if res.returncode == 0 and out_json.exists():
            return json.loads(out_json.read_text(encoding="utf-8"))
        log(f"Transcription on {device} failed" + (" - falling back to CPU" if device == "cuda" else ""))
    log("WARNING: no transcript - clips will be picked by audio/heatmap only, without captions.")
    return {"language": "", "segments": []}


# ----------------------------------------------------------------------------- signals
def extract_wav(src: Path, wav: Path) -> None:
    run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(wav)])


def loudness_per_second(wav: Path, n_sec: int) -> np.ndarray:
    with wave.open(str(wav), "rb") as w:
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    hop = sr // 2  # 0.5 s
    n = len(data) // hop
    if n == 0:
        return np.zeros(n_sec)
    frames = data[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-6
    db = 20 * np.log10(rms)
    lo, hi = np.percentile(db, 10), np.percentile(db, 99)
    norm = np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1)
    per_sec = norm[: (len(norm) // 2) * 2].reshape(-1, 2).mean(axis=1)
    out = np.zeros(n_sec)
    out[: min(n_sec, len(per_sec))] = per_sec[:n_sec]
    return out


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
            if w in HYPE_WORDS or "!" in word or "?" in word:
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


def words_in(segments: list, start: float, end: float) -> list:
    return [w for seg in segments for w in seg["words"] if w[0] >= start - 0.05 and w[1] <= end + 0.05]


def text_in(segments: list, start: float, end: float) -> str:
    return "".join(w[2] for w in words_in(segments, start, end)).strip()


# ----------------------------------------------------------------------------- selection
def find_candidates(duration: float, segments: list, heat, loud, hype, speech, cfg: dict) -> list[dict]:
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
    w_heat, w_loud, w_hype, w_speech = (0.45, 0.30, 0.15, 0.10) if has_heat else (0.0, 0.55, 0.30, 0.15)
    cs = lambda a: np.concatenate([[0], np.cumsum(a)])
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
            seg_loud = loud[a:b]
            l_score = 0.5 * seg_loud.mean() + 0.5 * np.percentile(seg_loud, 90) if len(seg_loud) else 0
            h = mean(c_heat, s, e) if has_heat else 0
            hy = mean(c_hype, s, e)
            sp = mean(c_speech, s, e)
            # strong opening: viewers decide in the first seconds
            hook_end = min(e, s + 5)
            hook = 0.5 * mean(c_loud, s, hook_end) + 0.5 * (mean(c_heat, s, hook_end) if has_heat else mean(c_hype, s, hook_end))
            score = w_heat * h + w_loud * l_score + w_hype * hy + w_speech * sp + 0.15 * hook
            txt = text_in(segments, s, e).lower() if segments else ""
            if any(p in txt for p in PROMO_WORDS):
                score -= 0.35
            cands.append({"start": round(s, 2), "end": round(e, 2), "score": float(score),
                          "heat": float(h), "loud": float(l_score), "hype": float(hy)})
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


def gemini_json(client, cfg: dict, prompt: str, temperature: float):
    """generate_content with JSON output; retries busy models and falls back to other models."""
    from google.genai import types

    models = [cfg["gemini_model"]] + [m for m in cfg.get("gemini_fallback_models", []) if m != cfg["gemini_model"]]
    err = None
    for model in models:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json", temperature=temperature),
                )
                return json.loads(resp.text)
            except Exception as e:
                err = e
                busy = any(c in str(e) for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "timed out", "Timeout"))
                log(f"  Gemini {model} attempt {attempt + 1}: {str(e)[:70]}")
                if busy and attempt < 2:
                    time.sleep(5 if attempt == 0 else 20)
                    continue
                break
    raise err


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
    items = []
    for i, c in enumerate(pool):
        items.append({
            "id": i, "start": c["start"], "end": c["end"],
            "most_replayed": round(c["heat"], 2), "loudness": round(c["loud"], 2),
            "transcript": text_in(segments, max(0, c["start"] - 10), c["end"] + 10)[:1500],
        })
    prompt = f"""You are an expert short-form video editor for a gaming clips channel.
Source video: "{meta['title']}" by {meta['channel']}.
Below are candidate moments (seconds). "most_replayed" (0-1) is YouTube's real replay heatmap;
"loudness" (0-1) is how loud/hype the audio is. Transcript covers 10 s before and after each candidate.

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
{lang_note}
Return JSON: {{"clips": [{{"id": int, "start": float, "end": float, "title": str, "banner": str,
"description": str, "hashtags": [str], "reason": str}}]}} ordered best first.

Candidates:
{json.dumps(items, ensure_ascii=False)}"""
    try:
        client = genai.Client(api_key=key, http_options={"timeout": 90_000})  # ms; never hang forever
        data = gemini_json(client, cfg, prompt, 0.4)
        clips = data["clips"] if isinstance(data, dict) else data
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
                continue
            e = min(max(e, s + lo), s + hi, dur)
            if e - s < min(lo, dur) - 0.5:
                continue
            if any(not (e + 1 <= x["start"] or s >= x["end"] + 1) for x in out):
                continue
            out.append({**base, "start": round(s, 2), "end": round(e, 2),
                        "title": str(c.get("title", ""))[:100], "banner": str(c.get("banner", ""))[:60],
                        "description": str(c.get("description", "")),
                        "hashtags": [str(h).lstrip("#").replace(" ", "") for h in c.get("hashtags", [])][:8],
                        "reason": str(c.get("reason", ""))})
        except Exception:
            continue
        if len(out) >= k:
            break
    return out or None


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
    blur = cfg["layout"] == "blur"
    cap_margin = 420 if blur else 560
    banner_margin = 300 if blur else 180
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
    banner = (clip.get("banner") or "").strip().upper().replace("{", "").replace("}", "")
    if cfg.get("show_title_banner") and banner:
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(dur)},Banner,,0,0,0,,{banner}")

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
_ENCODER = None


def pick_encoder() -> list[str]:
    global _ENCODER
    if _ENCODER is None:
        try:
            run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=0.5",
                 "-c:v", "h264_nvenc", "-f", "null", "-"])
            _ENCODER = ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0",
                        "-maxrate", "14M", "-bufsize", "20M"]
            log("Using NVIDIA GPU encoder (h264_nvenc)")
        except Exception:
            _ENCODER = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
            log("Using CPU encoder (libx264)")
    return _ENCODER


def render_clip(src: Path, clip: dict, folder: Path, base: str, cfg: dict, audio: bool) -> Path:
    ass_name = f"{base}.ass"
    out_name = f"{base}.mp4"
    if cfg["layout"] == "crop":
        vf = f"scale=-2:1920,crop=1080:1920,ass={ass_name}"
    else:
        z = float(cfg.get("zoom", 1.0))
        fw = int(round(1080 * z / 2) * 2)
        vf = (f"split=2[a][b];"
              f"[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
              f"boxblur=20:2,eq=brightness=-0.10[bg];"
              f"[b]scale={fw}:-2,crop=min(iw\\,1080):ih[fg];"
              f"[bg][fg]overlay=(W-w)/2:(H-h)/2,ass={ass_name}")
    dur = clip["end"] - clip["start"]
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{clip['start']:.2f}", "-i", str(src),
           "-t", f"{dur:.2f}", "-filter_complex" if cfg["layout"] != "crop" else "-vf", vf,
           *pick_encoder(), "-pix_fmt", "yuv420p", "-r", "30"]
    if audio:
        cmd += ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "160k", "-ar", "48000"]
    else:
        cmd += ["-an"]
    cmd += ["-movflags", "+faststart", out_name]
    try:
        run(cmd, cwd=folder)
    except RuntimeError:
        if "nvenc" in " ".join(pick_encoder()):
            global _ENCODER
            _ENCODER = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
            log("GPU encode failed, retrying on CPU")
            cmd = [c for c in cmd]
            i = cmd.index("-c:v")
            j = cmd.index("-pix_fmt")
            cmd[i:j] = _ENCODER
            run(cmd, cwd=folder)
        else:
            raise
    return folder / out_name


def write_details(clip: dict, rank: int, meta: dict, path: Path) -> None:
    tags = " ".join("#" + h for h in clip["hashtags"])
    t = int(clip["start"])
    link = meta["url"]
    if meta.get("id") and "youtu" in link:
        link = f"https://www.youtube.com/watch?v={meta['id']}&t={t}s"
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
Source time: {fmt_ts(clip['start'])} - {fmt_ts(clip['end'])}  ({clip['end'] - clip['start']:.0f}s)
Most-replayed score: {clip['heat']:.2f}   Loudness score: {clip['loud']:.2f}
Why: {clip['reason']}
"""
    path.write_text(body, encoding="utf-8")


# ----------------------------------------------------------------------------- main flow
def make_video_folder(cfg: dict, meta: dict) -> Path:
    day = Path(cfg["output_root"]) / dt.date.today().strftime("%d-%m-%Y")
    day.mkdir(parents=True, exist_ok=True)
    idx = len([p for p in day.iterdir() if p.is_dir() and re.match(r"^\d{2}_", p.name)]) + 1
    folder = day / f"{idx:02d}_{slugify(meta['channel'], 20)}_{slugify(meta['title'], 40)}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def process(source: str, cfg: dict, local: bool, credit: str = "", local_title: str = "") -> Path:
    t0 = time.time()
    tmp_root = Path(cfg["output_root"]) / "_work"
    work = tmp_root / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    work.mkdir(parents=True, exist_ok=True)

    if local:
        src = Path(source).resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        meta = {"title": local_title or src.stem, "channel": credit or "creator", "channel_url": "",
                "url": "", "id": "", "duration": probe_duration(src), "heatmap": [], "view_count": None}
        log(f"Using local file: {src.name}")
    else:
        log(f"Downloading: {source}")
        src, meta = download(source, work, cfg)
        if not meta["duration"]:
            meta["duration"] = probe_duration(src)
        log(f"Got '{meta['title']}' by {meta['channel']} ({fmt_ts(meta['duration'])})"
            + ("  [Most-replayed heatmap found]" if meta["heatmap"] else "  [no heatmap - using audio/speech only]"))

    duration = meta["duration"]
    n_sec = int(np.ceil(duration)) + 1
    audio = has_audio(src)

    wav = work / "audio.wav"
    if audio:
        log("Extracting audio...")
        extract_wav(src, wav)
        transcript = transcribe(wav, work, cfg)
        loud = loudness_per_second(wav, n_sec)
    else:
        transcript, loud = {"segments": []}, np.zeros(n_sec)
    segments = transcript["segments"]
    if not cfg.get("language"):  # auto: follow what Whisper detected
        cfg["_lang"] = transcript.get("language") or ""
    heat =heat_per_second(meta["heatmap"], n_sec)
    hype, speech = hype_per_second(segments, n_sec)

    log("Scoring moments...")
    cands = find_candidates(duration, segments, heat, loud, hype, speech, cfg)
    k = int(cfg["num_clips"])
    pool = pick_non_overlapping(cands, max(12, 2 * k))
    clips = gemini_pick(pool, segments, meta, cfg) if pool else None
    if clips:
        log(f"Gemini chose {len(clips)} clips")
        if len(clips) < k:  # top up with automatic picks
            for c in pool:
                if len(clips) >= k:
                    break
                if all(c["end"] + 1 <= x["start"] or c["start"] >= x["end"] + 1 for x in clips):
                    clips.append(dict(c))
    else:
        clips = [dict(c) for c in pool[:k]]
        if not cfg.get("gemini_api_key"):
            log("No Gemini key - using automatic picks and simple titles")
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
        raise RuntimeError("Video too short to make clips with the current length settings.")

    folder = make_video_folder(cfg, meta)
    log(f"Rendering {len(clips)} clips -> {folder}")
    summary = [f"{meta['title']}  —  {meta['channel']}", meta["url"], ""]
    for rank, clip in enumerate(clips, 1):
        base = f"{rank:02d}_{slugify(clip['title'], 45)}"
        write_captions(clip, segments, cfg, folder / f"{base}.ass", folder / f"{base}.srt")
        render_clip(src, clip, folder, base, cfg, audio)
        write_details(clip, rank, meta, folder / f"{base}.txt")
        (folder / f"{base}.ass").unlink(missing_ok=True)
        summary.append(f"{rank:02d}. [{fmt_ts(clip['start'])}-{fmt_ts(clip['end'])}] {clip['title']}")
        log(f"  {rank}/{len(clips)} done: {clip['title']}")
    (folder / "_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    if segments:
        (folder / "_transcript.txt").write_text(
            "\n".join(f"[{fmt_ts(s['start'])}] {s['text']}" for s in segments), encoding="utf-8")

    if cfg.get("keep_source") and not local:
        shutil.move(str(src), str(folder / f"_source{src.suffix}"))
    shutil.rmtree(work, ignore_errors=True)
    try:
        tmp_root.rmdir()
    except OSError:
        pass
    log(f"Finished in {(time.time() - t0) / 60:.1f} min -> {folder}")
    return folder


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--_transcribe":
        _transcribe_child(*sys.argv[2:7])
        return
    ap = argparse.ArgumentParser(description="YouTube link -> best moments as vertical captioned clips")
    ap.add_argument("urls", nargs="*", help="YouTube link(s)")
    ap.add_argument("--list", help="text file with one link per line")
    ap.add_argument("--file", help="use a local video file instead of a link")
    ap.add_argument("--credit", default="", help="creator name for a local file")
    ap.add_argument("--title", default="", help="video title for a local file")
    ap.add_argument("--out", help="output folder (overrides config.json)")
    ap.add_argument("--clips", type=int, help="number of clips (default 5)")
    ap.add_argument("--lang", choices=["en", "hi", "auto"],
                    help="spoken language (default: config 'language', normally en)")
    ap.add_argument("--verbose", action="store_true", help="show yt-dlp debug output")
    args = ap.parse_args()

    check_tools()
    cfg = load_config(args)
    urls = list(args.urls)
    if args.list:
        urls += [l.strip() for l in Path(args.list).read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")]
    if not urls and not args.file:
        u = input("Paste YouTube link: ").strip()
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
