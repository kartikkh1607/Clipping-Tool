#!/usr/bin/env python3
"""
Clipper web app - a local browser front end for clipper.py.

    start_app.bat        (or: python app.py)  ->  http://127.0.0.1:7860

All pipeline work is done by clipper.py's own functions (load_config + process); this file only
queues jobs, streams their progress and shows the finished clips. Jobs run in a background worker
thread, one at a time (one GPU job), so closing the browser tab never stops a running job.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr

import clipper
import live

HERE = Path(__file__).resolve().parent
CONFIG_PATH = clipper.CONFIG_PATH
BACKUP_PATH = HERE / "config.json.bak"
CAMPAIGNS_PATH = clipper.CAMPAIGNS_PATH
ICON_PATH = HERE / "clipper.ico"
if os.name == "nt":
    DEFAULT_OUT = r"E:\GamingClips" if Path("E:\\").exists() else str(Path.home() / "Videos" / "Clipper")
else:
    DEFAULT_OUT = str(Path.home() / ("Movies" if sys.platform == "darwin" else "Videos") / "Clipper")
HOST, PORT = "127.0.0.1", 7860

GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.8-flash"]  # free-tier order (clipper.DEFAULTS)
LANGS = [("English", "en"), ("Hindi", "hi"), ("Auto", "auto")]
LAYOUTS = ["auto", "blur", "crop", "facecam"]
PRO_EDIT = {   # Settings checkbox label -> config.json key
    "Split layout for streams (auto)": "auto_facecam",
    "Face zooms on big reactions": "react_zoom",
    "Word behind the streamer's head": "punch_word",
    "Censor slurs in captions": "censor_captions",
}
DATE_DIR = re.compile(r"^\d{2}-\d{2}-\d{4}$")
CLIP_FILE = re.compile(r"^\d{2,3}_.+\.mp4$")   # 01_title.mp4 (live streams can pass 99 clips)
RES_CHOICES = list(clipper.RESOLUTIONS)                      # 1080p, 1440p, 4K
FPS_CHOICES = [("Source", "source"), ("60", "60"), ("30", "30")]
SIZE_CHOICES = [("Smaller", "smaller"), ("Balanced", "balanced"), ("Best", "best")]
VIDEO_TYPES = [".mp4", ".mkv", ".mov", ".webm", ".avi", ".ts", ".m4v", ".flv"]


# ============================================================================ config
def read_raw_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    except Exception:
        return {}


def read_config() -> dict:
    return {**clipper.DEFAULTS, **read_raw_config()}


def write_config(updates: dict) -> None:
    """Merge updates into config.json; keeps the previous file as config.json.bak."""
    raw = read_raw_config()
    if CONFIG_PATH.exists():
        shutil.copy2(CONFIG_PATH, BACKUP_PATH)
    raw.update(updates)
    tmp = CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, CONFIG_PATH)


def output_root() -> str:
    return read_config().get("output_root") or DEFAULT_OUT


def mask_key(key: str) -> str:
    return ("•" * 12 + key[-4:]) if key else ""


def scrub(text: str) -> str:
    """Never let the API key reach the UI or the log."""
    key = read_config().get("gemini_api_key") or ""
    if key and key in text:
        text = text.replace(key, mask_key(key))
    return re.sub(r"AIza[0-9A-Za-z_\-]{20,}", lambda m: mask_key(m.group(0)), text)


def campaigns() -> dict:
    try:
        return json.loads(CAMPAIGNS_PATH.read_text(encoding="utf-8")) if CAMPAIGNS_PATH.exists() else {}
    except Exception:
        return {}


DEMO: list = [None]   # the running gr.Blocks


def serve_folder(path: str) -> None:
    """Let the browser play clips from this folder (served in place, never copied). Works at runtime,
    so changing the output folder in Settings needs no restart."""
    try:
        p = Path(path).resolve()
        from gradio.data_classes import _StaticFiles
        if p not in _StaticFiles.all_paths:
            gr.set_static_paths([p])
        demo = DEMO[0]
        if demo is not None and str(p) not in (demo.allowed_paths or []):
            demo.allowed_paths = list(demo.allowed_paths or []) + [str(p)]
    except Exception:
        pass


# ============================================================================ links
def parse_links(text: str) -> list[str]:
    out = []
    for line in (text or "").replace(",", "\n").split():
        line = line.strip().strip('"\'<>')
        if not line:
            continue
        if not line.startswith("http") and re.match(r"(www\.|m\.)?(youtu|twitch|clips\.twitch)", line):
            line = "https://" + line
        if line.startswith("http") and line not in out:
            out.append(line)
    return out


_CHANNEL_CACHE: dict[str, str] = {}


def twitch_channel(url: str) -> str:
    """Channel name of a Twitch link (VOD links need a quick metadata lookup)."""
    m = re.search(r"twitch\.tv/([A-Za-z0-9_]+)", url)
    if m and m.group(1).lower() not in ("videos", "clip", "directory"):
        return m.group(1).lower()
    if url in _CHANNEL_CACHE:
        return _CHANNEL_CACHE[url]
    chan = ""
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "socket_timeout": 10}) as ydl:
            info = ydl.extract_info(url, download=False, process=False)
        chan = str(info.get("uploader_id") or info.get("channel") or info.get("uploader") or "").lower()
    except Exception:
        pass
    _CHANNEL_CACHE[url] = chan
    return chan


def detect(links_text: str):
    """Pick campaign/section defaults from the pasted links."""
    urls = parse_links(links_text)
    twitch = [u for u in urls if clipper.is_twitch(u)]
    camp = "none"
    if twitch and len(twitch) == len(urls):
        chans = {twitch_channel(u) for u in twitch}
        for name, preset in campaigns().items():
            if (preset.get("source_channel") or "").lower() in chans and len(chans) == 1:
                camp = name
                break
    return (gr.update(value=camp), gr.update(visible=bool(twitch)),
            gr.update(open=True) if twitch else gr.update())


# ============================================================================ hints
HINTS = [
    (r"confirm you.?re not a bot|sign in to confirm|cookies? (database|from browser)|database is locked|could not copy .*cookie",
     "YouTube blocked the download as a bot. Close the cookies browser (Settings) and try again."),
    (r"only allows twitch footage|only allows footage from",
     "This link isn't from the campaign's channel. Set Campaign to \"none\" in Options, or use a link from that channel."),
    (r"--section|section (starts|end)",
     "The section must look like 0:30:00-2:00:00 and start before the video ends."),
    (r"campaign '.*' not found", "That campaign is missing from campaigns.json. Pick \"none\" in Options."),
    (r"overlay image not found", "The campaign's overlay picture is missing from the overlays folder."),
    (r"ffmpeg not found|ffprobe not found", "FFmpeg is missing. Run setup_windows.bat once, then restart Clipper."),
    (r"no space left|errno 28|disk full", "The drive is full. Free some space or pick another output folder in Settings."),
    (r"too short", "The video is too short for the clip length in Settings. Lower the minimum clip length."),
    (r"private video|video (is )?unavailable|has been removed|does not exist|http error 404|is not available|members.only",
     "The video is private, deleted or members-only, or the link is wrong. Open it in the browser to check."),
    (r"unsupported url|is not a valid url", "That doesn't look like a YouTube or Twitch video link."),
    (r"po token provider", "YouTube's PO Token helper isn't set up. Turn \"pot_provider\" off in config.json or install it (see README)."),
    (r"http error|timed out|getaddrinfo|connection|network|ssl", "Network problem while downloading. Check your internet and try again."),
    (r"out of memory|cuda", "The graphics card ran out of memory. Close games/other GPU apps and try again."),
]


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def hint_for(err: str) -> str:
    low = err.lower()
    for pat, hint in HINTS:
        if re.search(pat, low):
            return hint
    return "Something went wrong. The log below has the details."


# ============================================================================ clips on disk
def parse_details(text: str) -> dict:
    sections, cur = {}, None
    for line in text.splitlines():
        if line.startswith("--- info"):
            break
        if line.strip() in ("TITLE", "CAPTION", "DESCRIPTION", "HASHTAGS", "SOURCE"):
            cur = line.strip()
            sections[cur] = []
        elif cur:
            sections[cur].append(line)
    get = lambda k: "\n".join(sections.get(k, [])).strip()
    return {"title": get("TITLE"), "caption": get("CAPTION") or get("DESCRIPTION")}


def clip_files(folder: Path) -> list[Path]:
    """Finished clips in a folder, in number order (a clip's .txt is written last, after its quality check,
    so an .mp4 without one is still being made)."""
    files = [p for p in Path(folder).glob("*.mp4") if CLIP_FILE.match(p.name) and p.with_suffix(".txt").exists()]
    return sorted(files, key=lambda p: (int(p.name.split("_", 1)[0]), p.name))


def clips_in(folder: Path) -> list[dict]:
    out = []
    for mp4 in clip_files(folder):
        txt = mp4.with_suffix(".txt")
        d = parse_details(txt.read_text(encoding="utf-8", errors="replace")) if txt.exists() else {}
        out.append({"video": str(mp4), "title": d.get("title") or mp4.stem[3:].replace("_", " "),
                    "caption": d.get("caption", ""), "group": Path(folder).name})
    return out


def date_folders() -> list[tuple[str, str]]:
    root = Path(output_root())
    if not root.is_dir():
        return []
    items = []
    for p in root.iterdir():
        if p.is_dir() and DATE_DIR.match(p.name):
            try:
                day = dt.datetime.strptime(p.name, "%d-%m-%Y").date()
            except ValueError:
                continue
            vids = [v for v in p.iterdir() if v.is_dir()]
            n = sum(len(clip_files(v)) for v in vids)
            label = f"{day:%a %d %b %Y}  ·  {len(vids)} video{'s' * (len(vids) != 1)}, {n} clip{'s' * (n != 1)}"
            items.append((day, label, str(p)))
    items.sort(reverse=True)
    return [(label, path) for _, label, path in items]


def open_in_explorer(path: str) -> None:
    p = Path(path)
    if not p.exists():
        gr.Warning("That file or folder no longer exists.")
        return
    if os.name == "nt":
        if p.is_file():
            subprocess.Popen(f'explorer /select,"{p}"')
        else:
            os.startfile(str(p))
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(p)] if p.is_file() else ["open", str(p)])
    else:
        subprocess.Popen(["xdg-open", str(p if p.is_dir() else p.parent)])


# ============================================================================ jobs
class Video:
    """One item of a job: a link, a local file, or one window of a live stream."""

    def __init__(self, url: str, kind: str = "link", credit: str = "", title: str = "", window=None, run=None):
        self.url = url              # link, file path, or the stream link (live)
        self.kind = kind            # link | file | live
        self.credit, self.title = credit, title
        self.window, self.run = window, run     # live: the live.Window and the LiveRun it belongs to
        self.name = Path(url).name if kind == "file" else url
        self.state = "waiting"      # waiting | running | done | failed
        self.pct = 0.0
        self.stage = "Waiting"
        self.duration = 0.0
        self.n_clips = 0
        self.folder = ""
        self.error = ""
        self.hint = ""
        self.warnings: list[str] = []
        self.clips: list[dict] = []


class Job:
    def __init__(self, videos: list[Video], opts: dict):
        self.id = uuid.uuid4().hex[:8]
        self.videos = videos
        self.opts = opts
        self.live = any(v.kind == "live" for v in videos)
        self.log: list[str] = []
        self.state = "queued"       # queued | running | done
        self.version = 0            # bumped on every change, so viewers know when to refresh

    def touch(self):
        self.version += 1

    def clips(self) -> list[dict]:
        return [c for v in self.videos for c in v.clips]


JOBS: dict[str, Job] = {}
JOB_ORDER: list[str] = []
JOB_QUEUE: "queue.Queue[Job]" = queue.Queue()
CURRENT: list = [None, None]   # (job, video) being processed

WARN_RULES = [
    ("Gemini step failed", "Gemini was busy or unavailable: clips were made with automatic picks; titles are simpler."),
    ("No Gemini key", "No Gemini key set: automatic picks and simple titles. Add a free key in Settings."),
    ("facecam not found in its box", "The facecam wasn't visible in some clips, so those use the full-frame layout."),
    ("No usable facecam box", "No facecam position for this video, so the clips use the full-frame layout."),
    ("Hinglish transliteration failed", "Some Hindi captions stayed in Devanagari (Gemini didn't answer)."),
    ("no transcript", "Speech recognition failed: clips have no captions."),
    ("High-quality download failed", "The high-quality re-download failed, so clips were made from the first download (lower resolution)."),
    ("couldn't line up the high-quality", "Some clips couldn't use the high-quality footage and were made from the first download."),
    ("rendering from the high-quality footage failed", "Some clips couldn't use the high-quality footage and were made from the first download."),
    ("quality check failed", "A clip failed the quality check and was re-rendered on the CPU. Check its .txt file."),
    ("quality check:", "The quality check flagged something on a clip (see the log or the clip's .txt file)."),
]


def on_log(line: str) -> None:
    job, v = CURRENT
    if not job:
        return
    line = scrub(ANSI.sub("", line))
    job.log.append(line)
    if len(job.log) > 3000:
        del job.log[:1000]
    if v:
        update_progress(v, line)
        if v.run:   # a live window: its progress also shows on the Live tab
            v.run.add(line, echo=False)
    job.touch()


def update_progress(v: Video, line: str) -> None:
    msg = line.split("] ", 1)[1] if line.startswith("[") and "] " in line[:12] else line
    m = re.match(r"\s+(\d+(?::\d{2}){1,2})\s\s", line)   # a transcript line from Whisper
    if m and v.pct < 62:
        t = clipper.parse_time(m.group(1))
        frac = min(1.0, t / v.duration) if v.duration else 0.5
        v.pct, v.stage = max(v.pct, 30 + 30 * frac), f"Listening to the speech  {m.group(1)}"
        return
    if msg.startswith("Downloading"):
        v.pct, v.stage = max(v.pct, 3), "Downloading"
    elif msg.startswith("Using local file"):
        v.pct, v.stage = max(v.pct, 25), "Reading the file"
    elif msg.startswith("Got '"):
        g = re.match(r"Got '(.*)' by (.*?) \((\d[\d:]*)\)", msg)
        if g:
            v.name = f"{g.group(1)}  ·  {g.group(2)}"
            v.duration = clipper.parse_time(g.group(3))
        v.pct, v.stage = max(v.pct, 25), "Downloaded"
    elif msg.startswith("Extracting audio"):
        v.pct, v.stage = max(v.pct, 27), "Extracting audio"
    elif msg.startswith("Transcribing on"):
        v.pct, v.stage = max(v.pct, 30), "Listening to the speech (first run downloads the model)"
    elif msg.startswith("Scoring moments"):
        v.pct, v.stage = max(v.pct, 62), "Finding the best moments"
    elif msg.strip().startswith("Gemini ") and "attempt" in msg:
        v.stage = "Asking Gemini (busy, retrying)"
    elif msg.startswith("Gemini chose"):
        v.pct, v.stage = max(v.pct, 70), msg
    elif msg.startswith("Fetching the"):
        v.pct, v.stage = max(v.pct, 70), "Getting the chosen moments in high quality"
    elif msg.startswith("High-quality footage ready"):
        v.pct, v.stage = max(v.pct, 72), msg
    elif msg.startswith("Rendering"):
        r = re.match(r"Rendering (\d+) clips", msg)
        v.n_clips = int(r.group(1)) if r else 0
        v.pct, v.stage = max(v.pct, 72), f"Making {v.n_clips} clips"
    elif re.match(r"\s+(\d+)/(\d+) done", msg):
        k, n = map(int, re.match(r"\s+(\d+)/(\d+) done", msg).groups())
        v.pct, v.stage = max(v.pct, 72 + 26 * k / max(n, 1)), f"Made clip {k} of {n}"
    elif msg.startswith("Finished in"):
        v.pct, v.stage = 100, "Done"
    for key, warn in WARN_RULES:
        if key in msg and warn not in v.warnings:
            v.warnings.append(warn)


def download_hook(v: Video, d: dict) -> None:
    if d.get("status") != "downloading":
        return
    total = d.get("total_bytes") or d.get("total_bytes_estimate")
    frac = None
    if total and d.get("downloaded_bytes"):
        frac = d["downloaded_bytes"] / total
    elif d.get("fragment_count") and d.get("fragment_index"):
        frac = d["fragment_index"] / d["fragment_count"]
    if frac is not None:
        new = 3 + 22 * min(1.0, frac)
        if new - v.pct >= 0.5 or frac >= 1:
            v.pct, v.stage = max(v.pct, new), f"Downloading  {frac * 100:.0f}%"
            if CURRENT[0]:
                CURRENT[0].touch()


def job_cfg(o: dict, section: str | None = None) -> dict:
    """Pipeline settings for one item: config.json is re-read here, so new settings/keys apply at once."""
    args = argparse.Namespace(
        campaign=o["campaign"] if o.get("campaign", "none") != "none" else None,
        overlay_image=None, section=section or None, out=output_root(), clips=int(o.get("clips") or 5),
        lang=o.get("lang") or "en", verbose=False)
    clipper.check_tools()
    cfg = clipper.load_config(args)
    serve_folder(cfg["output_root"])
    return cfg


def gradio_upload(path: str) -> bool:
    """True for a file Gradio stored in its upload cache (its own copy of what was dropped in the browser)."""
    try:
        cache = Path(os.environ.get("GRADIO_TEMP_DIR") or Path(tempfile.gettempdir()) / "gradio").resolve()
        return Path(path).resolve().is_relative_to(cache)
    except Exception:
        return False


def run_video(job: Job, v: Video) -> None:
    o = job.opts
    if v.kind == "live":
        run_live_window(job, v)
        return
    twitch = v.kind == "link" and clipper.is_twitch(v.url)
    cfg = job_cfg(o, o["section"] if (twitch and o.get("section")) else None)
    cfg["_progress_hook"] = lambda d: download_hook(v, d)
    if hasattr(clipper, "GEMINI_USED"):
        clipper.GEMINI_USED.clear()
    work_root = Path(cfg["output_root"]) / "_work"
    before = set(work_root.iterdir()) if work_root.is_dir() else set()
    try:
        if v.kind == "file":   # the path Gradio gave us is used as it is: no second copy of a big file
            try:
                v.duration = clipper.probe_duration(Path(v.url))
            except Exception:
                v.duration = 0.0
            v.name = f"{v.title or Path(v.url).stem}  ·  {v.credit or 'local file'}"
            folder = clipper.process(v.url, cfg, True, credit=v.credit, local_title=v.title or Path(v.url).stem)
        else:
            folder = clipper.process(v.url, cfg, False)
    except BaseException:
        if work_root.is_dir():   # remove this run's half-finished download
            for p in set(work_root.iterdir()) - before:
                shutil.rmtree(p, ignore_errors=True)
        raise
    finally:
        if v.kind == "file" and gradio_upload(v.url):   # Gradio's upload copy isn't needed any more
            Path(v.url).unlink(missing_ok=True)
    v.folder = str(folder)
    v.clips = clips_in(folder)


def worker() -> None:
    while True:
        job = JOB_QUEUE.get()
        job.state = "running"
        clipper.LOG_HOOK = on_log
        for i, v in enumerate(job.videos, 1):
            CURRENT[:] = [job, v]
            v.state, v.stage = "running", "Starting"
            on_log(f"[{dt.datetime.now():%H:%M:%S}] === Video {i}/{len(job.videos)}: {v.url} ===")
            try:
                run_video(job, v)
                v.state, v.pct, v.stage = "done", 100, f"Done · {len(v.clips)} clips"
            except BaseException as e:   # SystemExit from load_config too; one bad link never stops the rest
                err = str(e.code if isinstance(e, SystemExit) else e) or e.__class__.__name__
                err = scrub(re.sub(r"^ERROR:\s*", "", ANSI.sub("", err)))
                v.state, v.stage = "failed", "Failed"
                v.error, v.hint = err.strip().splitlines()[0][:300] if err.strip() else "Unknown error", hint_for(err)
                on_log(f"FAILED: {v.url}\n{err}")
            job.touch()
        CURRENT[:] = [None, None]
        job.state = "done"
        job.touch()


threading.Thread(target=worker, name="clipper-worker", daemon=True).start()


def submit(videos: list[Video], opts: dict) -> Job:
    job = Job(videos, opts)
    JOBS[job.id] = job
    JOB_ORDER.append(job.id)
    JOB_QUEUE.put(job)
    return job


# ============================================================================ live streams
class LiveRun:
    """The live stream being clipped (one at a time): its LiveSession, status and log for the Live tab."""

    def __init__(self, url: str, opts: dict):
        self.url, self.opts = url, opts
        self.session: live.LiveSession | None = None
        self.state = "connecting"   # connecting | live | stopping | finished | failed
        self.error = ""
        self.hint = ""
        self.log: list[str] = []
        self.started = time.time()

    def add(self, msg: str, echo: bool = True) -> None:
        line = msg if msg.startswith("[") else f"[{dt.datetime.now():%H:%M:%S}] {msg}"
        if echo:
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                print(line.encode("ascii", "replace").decode(), flush=True)
        self.log.append(scrub(ANSI.sub("", line)))
        if len(self.log) > 3000:
            del self.log[:1000]

    @property
    def active(self) -> bool:
        return self.state in ("connecting", "live", "stopping")


LIVE: list = [None]   # the current (or last) LiveRun


def live_link_problem(url: str) -> str:
    if not url:
        return "Paste a live Twitch or YouTube link first."
    if clipper.is_twitch(url):
        if re.search(r"twitch\.tv/(videos/|[^/]+/clip/)|clips\.twitch", url):
            return "That's a VOD or clip, not a live channel. Paste it on the Make Clips tab instead."
        if not re.search(r"twitch\.tv/[A-Za-z0-9_]+/?(\?.*)?$", url):
            return "A live Twitch link looks like https://www.twitch.tv/<channel>."
        return ""
    if clipper.is_youtube(url):
        return ""
    return "Live mode works with Twitch channels and YouTube live streams."


def submit_live_window(run: LiveRun, win) -> None:
    sess = run.session
    chan = (sess.info.get("channel") if sess else "") or "stream"
    v = Video(run.url, kind="live", window=win, run=run)
    v.name = f"🔴 {chan} live · window {win.number} ({win.label()} recorded)"
    submit([v], {**run.opts, "section": ""})
    run.add(f"Live: window {win.number} queued ({win.label()} recording time)")


def run_live_window(job: Job, v: Video) -> None:
    run, sess = v.run, v.run.session
    cfg = job_cfg(job.opts)
    before = {c["video"] for c in clips_in(sess.folder)} if sess.folder.exists() else set()
    try:
        sess.process_window(v.window, cfg)
    finally:
        if sess.ended and not sess.pending:   # the last window of a finished stream: tidy up
            sess.finish()
            if run.state != "failed":
                run.state = "finished"
    v.folder = str(sess.folder)
    v.clips = [c for c in clips_in(sess.folder) if c["video"] not in before]
    v.stage = f"Done · {len(v.clips)} new clip{'s' * (len(v.clips) != 1)}"


def live_boot(run: LiveRun) -> None:
    """Background: check the stream, then record it; each closed chunk queues a window job."""
    try:
        cfg = job_cfg(run.opts)
        run.add(f"Live: checking {run.url} ...")
        info = live.stream_info(run.url, cfg)
        clipper.check_campaign_source(run.url, info.pop("raw"), cfg)
        if cfg.get("creator_name"):
            info["channel"] = cfg["creator_name"]
        sess = live.LiveSession(run.url, cfg, int(run.opts["per_window"]), info=info, log=run.add,
                                on_window=lambda w: submit_live_window(run, w))
        run.session = sess
        serve_folder(str(Path(cfg["output_root"]).resolve()))
        first = [info]

        def resolve():   # first call: the info we just fetched; reconnects: fresh URL (they expire)
            if first:
                return first.pop()
            fresh = live.stream_info(run.url, cfg)
            sess.info.update({k: v for k, v in fresh.items() if k != "raw"})
            return fresh

        run.state = "live"
        run.add(f"Live: recording '{info['title']}' by {info['channel']} in {live.CHUNK_SECONDS // 60}-minute "
                f"chunks -> {sess.folder}")
        rec = sess.start_recording(resolve=resolve)
        rec.thread.join()
        if rec.state == "failed" and not sess.chunks:
            raise RuntimeError(rec.error or "the recording failed")
        if run.state != "stopping":
            run.add("Live: the stream has ended - clipping what's left")
        while sess.pending:   # windows still waiting for the GPU
            time.sleep(2)
        if sess.work.exists():
            sess.finish()
        run.state = "finished"
        run.add("Live: finished")
    except BaseException as e:
        err = str(e.code if isinstance(e, SystemExit) else e).strip() or type(e).__name__
        err = scrub(re.sub(r"^ERROR:\s*", "", ANSI.sub("", err)))
        run.state, run.error, run.hint = "failed", err.splitlines()[0][:300], hint_for(err)
        run.add(f"Live: FAILED - {err}")


def live_start(url: str, campaign: str, lang: str, per_window: int):
    url = (parse_links(url) or [""])[0]
    cur = LIVE[0]
    if cur and cur.active:
        return live_status_html(cur, "A live stream is already being clipped. Stop it first.")
    problem = live_link_problem(url)
    if problem:
        return f'<div class="cl-err"><b>{html.escape(problem)}</b></div>'
    run = LiveRun(url, {"campaign": campaign or "none", "lang": lang or "en", "clips": int(per_window or 2),
                        "per_window": int(per_window or 2)})
    LIVE[0] = run
    threading.Thread(target=live_boot, args=(run,), name="clipper-live", daemon=True).start()
    return live_status_html(run)


def live_stop():
    run = LIVE[0]
    if not run or not run.active:
        return live_status_html(run, "Nothing is being recorded.")
    if run.session:
        run.state = "stopping"
        run.add("Live: stopping the recording (the part already recorded is still clipped)...")
        run.session.stop()
    else:
        run.state, run.error = "failed", "Stopped before the recording started."
    return live_status_html(run)


def live_status_html(run: "LiveRun | None", note: str = "") -> str:
    parts = []
    if note:
        parts.append(f'<div class="cl-warn">⚠️ {html.escape(note)}</div>')
    if not run:
        return '<div class="cl-status">' + "".join(parts) + "</div>" if parts else ""
    sess = run.session
    s = sess.status() if sess else {}
    label = {"connecting": "⏳ Connecting to the stream…", "live": "🔴 Recording", "stopping": "⏹️ Stopping…",
             "finished": "✅ Finished", "failed": "❌ Failed"}[run.state]
    rstate = s.get("state")
    if run.state == "live" and rstate == "reconnecting":
        label = "🟠 Reconnecting (the stream dropped)…"
    elif run.state in ("live", "stopping") and rstate in ("ended", "stopped") and sess and sess.pending:
        label = "⚙️ Recording over · clipping the last windows"
    title = (sess.info.get("title") if sess else "") or run.url
    chan = (sess.info.get("channel") if sess else "") or ""
    parts.append(f'''<div class="cl-vid cl-{'failed' if run.state == 'failed' else 'done' if run.state == 'finished' else 'running'}">
  <div class="cl-row"><span class="cl-name">{html.escape(title)}{'  ·  ' + html.escape(chan) if chan else ''}</span>
  <span class="cl-stage">{label}</span></div>''')
    if sess:
        cur = next((w for w in sess.windows if w.status == "running"), None)
        stats = [("Recorded", clipper.fmt_ts(s["recording"])), ("Chunks", s["chunks"]),
                 ("Windows clipped", s["windows_done"]), ("Waiting", s["windows_queued"] - (1 if cur else 0)),
                 ("Clips made", s["clips"])]
        if s["reconnects"]:
            stats.append(("Reconnects", s["reconnects"]))
        if s["windows_failed"]:
            stats.append(("Windows failed", s["windows_failed"]))
        parts.append('<div class="cl-stats">' + "".join(
            f'<div><span>{html.escape(k)}</span><b>{html.escape(str(v))}</b></div>' for k, v in stats) + "</div>")
        if cur:
            parts.append(f'<div class="cl-note">⚙️ Clipping window {cur.number} ({cur.label()} recorded)…</div>')
        ahead = sum(1 for j in JOBS.values() if j.state != "done" and not j.live)
        if ahead and sess.pending and not cur:
            parts.append(f'<div class="cl-note">⏳ The next window waits for {ahead} link/file job'
                         f'{"s" * (ahead != 1)} to finish (one GPU job at a time).</div>')
        parts.append(f'<div class="cl-note">📂 {html.escape(str(sess.folder))}</div>')
        for e in s["errors"]:
            parts.append(f'<div class="cl-warn">⚠️ {html.escape(scrub(e))}</div>')
    if run.error:
        parts.append(f'<div class="cl-err"><b>{html.escape(run.hint or "")}</b><br><span>{html.escape(run.error)}</span></div>')
    parts.append("</div>")
    return '<div class="cl-status">' + "".join(parts) + "</div>"


def live_clips() -> list[dict]:
    run = LIVE[0]
    if not run or not run.session or not run.session.folder.exists():
        return []
    return clips_in(run.session.folder)


def live_tick(shown: list):
    """Live tab refresh (every 2 s): status, log, and the clip cards only when a new clip is finished."""
    run = LIVE[0]
    clips = live_clips()
    same = [c["video"] for c in clips] == [c["video"] for c in (shown or [])]
    return live_status_html(run), "\n".join(run.log[-400:]) if run else "", gr.skip() if same else clips


# ============================================================================ status view
def status_html(job: Job | None) -> str:
    if not job:
        return ""
    ahead = sum(1 for jid in JOB_ORDER if JOBS[jid].state != "done" and jid != job.id
                and JOB_ORDER.index(jid) < JOB_ORDER.index(job.id))
    parts = []
    if job.state == "queued":
        parts.append(f'<div class="cl-note">⏳ Waiting for {ahead} earlier job{"s" * (ahead != 1)} to finish (one at a time on the GPU)…</div>')
    for v in job.videos:
        pct = 100 if v.state == "done" else v.pct
        icon = {"waiting": "⏳", "running": "⚙️", "done": "✅", "failed": "❌"}[v.state]
        parts.append(f'''<div class="cl-vid cl-{v.state}">
  <div class="cl-row"><span class="cl-name">{icon} {html.escape(v.name)}</span><span class="cl-stage">{html.escape(v.stage)}</span></div>
  <div class="cl-bar"><div style="width:{pct:.0f}%"></div></div>''')
        if v.error:
            parts.append(f'<div class="cl-err"><b>{html.escape(v.hint)}</b><br><span>{html.escape(v.error)}</span></div>')
        for w in v.warnings:
            parts.append(f'<div class="cl-warn">⚠️ {html.escape(w)}</div>')
        parts.append("</div>")
    if job.state == "done":
        ok = sum(v.state == "done" for v in job.videos)
        n = len(job.clips())
        parts.append(f'<div class="cl-note">Finished: {ok} of {len(job.videos)} video{"s" * (len(job.videos) != 1)}, '
                     f'{n} clip{"s" * (n != 1)}.</div>')
    return '<div class="cl-status">' + "".join(parts) + "</div>"


def watch(job_id: str):
    """Stream one job's log, progress and clips until it finishes."""
    last = None
    while True:
        job = JOBS.get(job_id)
        if not job:
            return
        snap = (job.version, job.state)
        if snap != last:
            last = snap
            yield "\n".join(job.log[-400:]), status_html(job), job.clips(), job.id
        if job.state == "done":
            return
        time.sleep(0.5)


SECTION_RE = re.compile(r"\s*([\d:.]+)\s*-\s*([\d:.]+)\s*")


def file_paths(files) -> list[str]:
    """Paths from the gr.File box (a list of paths, or of file objects in older Gradio versions)."""
    out = []
    for f in files or []:
        p = f if isinstance(f, str) else getattr(f, "path", None) or getattr(f, "name", None) or (
            f.get("path") if isinstance(f, dict) else None)
        if p and Path(p).is_file() and str(p) not in out:
            out.append(str(p))
    return out


def start(links: str, files, credit: str, title: str, campaign: str, lang: str, clips: int, section: str):
    urls = parse_links(links)
    paths = file_paths(files)
    if not urls and not paths:
        yield gr.update(), ('<div class="cl-err"><b>Paste at least one YouTube or Twitch link, or drop a video '
                            'file, first.</b></div>'), [], None
        return
    section = (section or "").strip()
    if section:
        m = SECTION_RE.fullmatch(section)
        if not m or clipper.parse_time(m.group(2)) <= clipper.parse_time(m.group(1)):
            yield gr.update(), ('<div class="cl-err"><b>The section must look like 0:30:00-2:00:00 '
                                '(start-end, end after start). Leave it empty for the whole VOD.</b></div>'), [], None
            return
    credit, title = (credit or "").strip(), (title or "").strip()
    videos = [Video(u) for u in urls]
    for i, p in enumerate(paths, 1):   # local files: no download, no campaign source check
        t = (title + (f" ({i})" if len(paths) > 1 else "")) if title else ""
        videos.append(Video(p, kind="file", credit=credit, title=t))
    job = submit(videos, {"campaign": campaign or "none", "lang": lang or "en", "clips": clips or 5, "section": section})
    yield from watch(job.id)


def reattach():
    """On page load: show the running (or last) link/file job, even after the tab was closed (live windows
    show on the Live tab)."""
    ids = [j for j in JOB_ORDER if not JOBS[j].live]
    if not ids:
        return
    yield from watch(ids[-1])


# ============================================================================ settings
def key_status(key: str) -> str:
    return f"Saved key: `{mask_key(key)}`" if key else "No key saved yet - clips still work, with simpler titles."


def resolve_key(field: str) -> str:
    field = (field or "").strip().strip('"\'')
    return read_config().get("gemini_api_key", "") if "•" in field else field


def save_key(field: str):
    field = (field or "").strip().strip('"\'')
    if "•" in field:
        return gr.update(), "Key unchanged (paste a new key to replace it)."
    if field and (" " in field or len(field) < 20):
        return gr.update(), "❌ That doesn't look like a Gemini API key (they're about 39 characters, no spaces)."
    write_config({"gemini_api_key": field})
    msg = "✅ Key saved. The next run uses it." if field else "Key removed. Clips will use automatic picks."
    return gr.update(value=mask_key(field)), msg + "  \n" + key_status(field)


def test_key(field: str, model: str) -> str:
    key = resolve_key(field)
    if not key:
        return "❌ No key to test. Paste one first."
    model = model or read_config().get("gemini_model") or GEMINI_MODELS[0]
    try:
        from google import genai
        client = genai.Client(api_key=key, http_options={"timeout": 30_000})
        client.models.generate_content(model=model, contents="Reply with the single word OK.")
        return f"Key works ✅ (model: {model})"
    except Exception as e:
        msg = scrub(str(e))
        low = msg.lower()
        if "api_key_invalid" in low or "api key not valid" in low or "permission_denied" in low or "401" in msg or "403" in msg:
            head = "❌ Invalid key: Google rejected it. Copy the key again from AI Studio."
        elif "429" in msg or "resource_exhausted" in low or "quota" in low:
            head = "⚠️ Quota exceeded: the key works but today's free limit is used up (or too many requests per minute)."
        elif "503" in msg or "unavailable" in low or "overloaded" in low:
            head = f"⚠️ Overloaded: {model} is busy right now. The key is probably fine; try again in a minute."
        elif "404" in msg or "not_found" in low:
            head = f"❌ Model not found: {model} isn't available for this key. Pick another model."
        elif "timed out" in low or "timeout" in low or "connect" in low:
            head = "⚠️ No answer from Google (network problem or timeout)."
        else:
            head = "❌ The test failed."
        return f"{head}\n\n```\n{msg[:600]}\n```"


def check_folder(path: str) -> tuple[bool, str]:
    path = (path or "").strip().strip('"')
    if not path:
        return False, "Output folder can't be empty."
    p = Path(path)
    if not p.is_absolute():
        return False, f"Use a full path like {DEFAULT_OUT}."
    try:
        existed = p.is_dir()
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".clipper_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, ("Folder OK." if existed else "Folder created.")
    except Exception as e:
        return False, f"Can't use that folder: {e}"


def res_label(cfg: dict) -> str:
    h = clipper.out_height(cfg)
    return next((k for k, v in clipper.RESOLUTIONS.items() if v == h), "1080p")


def size_choice(cfg: dict) -> str:
    """The File size option matching config.json (older configs only have nvenc_cq: match on that)."""
    fs = str(cfg.get("file_size") or "").lower()
    if fs in clipper.FILE_SIZES:
        return fs
    cq = cfg.get("nvenc_cq", 22)
    return next((k for k, v in clipper.FILE_SIZES.items() if v["nvenc_cq"] == cq), "balanced")


def quality_cfg(res: str, fps: str, size: str) -> dict:
    return {**read_config(), "output_height": clipper.RESOLUTIONS.get(res, 1920),
            "output_fps": fps if fps in ("60", "30") else "source", **clipper.FILE_SIZES.get(size, {})}


def estimate_text(res: str, fps: str, size: str) -> str:
    cfg = quality_cfg(res, fps, size)
    typ, worst = clipper.estimate_clip_mb(cfg, 30)
    w, h = clipper.out_size(cfg)
    rate = "the source frame rate (60 fps for most gameplay)" if cfg["output_fps"] == "source" else f"{cfg['output_fps']} fps"
    return (f"**Estimated size per 30 s clip: about {typ} MB** (up to {worst} MB for very busy gameplay) · "
            f"{w}x{h} at {rate}, bitrate cap {clipper.scaled_bitrate(cfg)}bps")


def save_settings(key_field, model, folder, min_len, max_len, clips, layout, cookies, pro=None,
                  res=None, fps=None, size=None):
    errors = []
    ok, folder_msg = check_folder(folder)
    if not ok:
        errors.append(folder_msg)
    try:
        min_len, max_len = int(min_len), int(max_len)
        if not 5 <= min_len <= 170:
            errors.append("Minimum clip length must be 5-170 seconds.")
        if not 10 <= max_len <= 180:
            errors.append("Maximum clip length must be 10-180 seconds.")
        if min_len >= max_len:
            errors.append("Minimum clip length must be shorter than the maximum.")
    except (TypeError, ValueError):
        errors.append("Clip lengths must be whole numbers.")
    if layout not in LAYOUTS:
        errors.append("Pick a layout.")
    if errors:
        return gr.update(), "❌ Not saved:\n" + "\n".join(f"- {e}" for e in errors)
    updates = {"output_root": str(Path(folder.strip().strip('"'))), "min_clip_seconds": min_len,
               "max_clip_seconds": max_len, "num_clips": int(clips), "layout": layout,
               "cookies_from_browser": "" if cookies == "none" else cookies, "gemini_model": model}
    if pro is not None:
        updates.update({k: (label in pro) for label, k in PRO_EDIT.items()})
    if res in clipper.RESOLUTIONS:
        updates["output_height"] = clipper.RESOLUTIONS[res]
    if fps:
        updates["output_fps"] = int(fps) if fps in ("60", "30") else "source"
    if size in clipper.FILE_SIZES:   # the 1080p values; 1440p/4K scale max_bitrate up when rendering
        updates.update({"file_size": size, **clipper.FILE_SIZES[size]})
    key_msg = ""
    k = (key_field or "").strip().strip('"\'')
    if k and "•" not in k:
        if " " in k or len(k) < 20:
            return gr.update(), "❌ Not saved: that doesn't look like a Gemini API key."
        updates["gemini_api_key"] = k
        key_msg = " New API key saved too."
    write_config(updates)
    serve_folder(updates["output_root"])
    key = read_config().get("gemini_api_key", "")
    return gr.update(value=mask_key(key)), f"✅ Settings saved ({folder_msg}){key_msg} Backup: config.json.bak"


# ============================================================================ UI
CSS = """
.gradio-container { max-width: 1180px !important; width: 100% !important; margin: 0 auto !important; }
.gradio-container .tabs, .gradio-container .tabitem { width: 100%; }
#days .wrap { gap: 8px; }
#hero h1 { font-size: 2rem; margin: 0.2em 0 0; }
#hero p { margin: 0.2em 0 0.6em; opacity: .75; }
#make-btn { font-size: 1.25rem !important; min-height: 58px !important; font-weight: 700; }
#paste-btn { min-height: 100%; }
.cl-status { display: flex; flex-direction: column; gap: 10px; }
.cl-vid { border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 10px 14px;
          background: var(--background-fill-secondary); }
.cl-row { display: flex; justify-content: space-between; gap: 12px; font-size: .95rem; }
.cl-name { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.cl-stage { opacity: .75; white-space: nowrap; }
.cl-bar { height: 8px; border-radius: 99px; background: var(--border-color-primary); margin-top: 8px; overflow: hidden; }
.cl-bar > div { height: 100%; border-radius: 99px; background: var(--color-accent); transition: width .5s ease; }
.cl-done .cl-bar > div { background: #22c55e; }
.cl-failed .cl-bar > div { background: #ef4444; }
.cl-err { color: #ef4444; margin-top: 8px; font-size: .95rem; }
.cl-err span { opacity: .8; font-size: .85rem; word-break: break-word; }
.cl-warn { color: #d97706; margin-top: 6px; font-size: .9rem; }
.cl-note { opacity: .85; font-size: .95rem; }
.clip-grid { display: grid !important; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
.clip-card { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 10px !important;
             background: var(--background-fill-secondary); gap: 8px !important; min-width: 0 !important; }
.clip-card .clip-title { font-weight: 700; font-size: 1.02rem; line-height: 1.3; }
.clip-card .clip-group { opacity: .6; font-size: .8rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.clip-card video { max-height: 460px; background: #000; }
.clip-btns { flex-wrap: nowrap !important; gap: 6px !important; }
.clip-btns button { font-size: .8rem !important; padding: 6px 2px !important; white-space: nowrap; }
#log textarea { font-family: Consolas, ui-monospace, monospace !important; font-size: .82rem !important; }
.empty-note { opacity: .6; padding: 20px 0; }
.set-card { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 16px 18px !important;
            background: var(--background-fill-secondary); margin-bottom: 14px; }
.set-card h3 { margin: 0 0 2px; }
.cl-stats { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px; margin-top: 10px; }
.cl-stats > div { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 6px 10px;
                  display: flex; flex-direction: column; }
.cl-stats span { font-size: .78rem; opacity: .7; }
.cl-stats b { font-size: 1.15rem; font-variant-numeric: tabular-nums; }
#files { min-height: 100%; }
#live-start { font-size: 1.1rem !important; min-height: 52px !important; font-weight: 700; }
#live-stop { min-height: 52px !important; }
"""

# Enter in the link box starts the job; Shift+Enter adds a new line (Gradio's default is the reverse).
JS = """
() => {
  document.addEventListener('keydown', (e) => {
    const ta = e.target;
    if (e.key !== 'Enter' || !ta || !ta.closest || !ta.closest('#links')) return;
    e.stopPropagation();
    if (!e.shiftKey) {
      e.preventDefault();
      setTimeout(() => document.querySelector('#make-btn')?.click(), 120);
    }
  }, true);
  // Live log: follow new lines unless the user scrolled up to read.
  setInterval(() => {
    const ta = document.querySelector('#log textarea');
    if (!ta || ta.value.length === Number(ta.dataset.len || -1)) return;
    const atEnd = !ta.dataset.len || ta.scrollHeight - ta.scrollTop - ta.clientHeight < 80 || ta.dataset.follow === '1';
    ta.dataset.len = ta.value.length;
    if (atEnd) { ta.scrollTop = ta.scrollHeight; ta.dataset.follow = '1'; }
  }, 600);
  document.addEventListener('wheel', (e) => {
    const ta = e.target.closest && e.target.closest('#log textarea');
    if (ta) setTimeout(() => { ta.dataset.follow = (ta.scrollHeight - ta.scrollTop - ta.clientHeight < 40) ? '1' : '0'; }, 50);
  }, true);
}
"""


def copy_js(text: str) -> str:
    return f"() => {{ navigator.clipboard.writeText({json.dumps(text)}); }}"


def clip_cards(clips: list[dict], empty: str) -> None:
    if not clips:
        gr.Markdown(empty, elem_classes="empty-note")
        return
    with gr.Row(elem_classes="clip-grid"):
        for c in clips:
            with gr.Column(elem_classes="clip-card", min_width=0):
                gr.Video(value=c["video"], show_label=False, interactive=False, height=460,
                         buttons=["download"])
                gr.HTML(f'<div class="clip-title">{html.escape(c["title"])}</div>'
                        f'<div class="clip-group">{html.escape(c.get("group", ""))}</div>')
                gr.Textbox(value=c["caption"], label="Caption + hashtags", lines=4, max_lines=8,
                           interactive=False)
                with gr.Row(elem_classes="clip-btns"):
                    bt = gr.Button("📋 Copy title", size="sm", min_width=0)
                    bc = gr.Button("📋 Copy caption", size="sm", min_width=0)
                    bo = gr.Button("📂 Open folder", size="sm", min_width=0)
                bt.click(lambda: gr.Info("Title copied"), js=copy_js(c["title"]), queue=False)
                bc.click(lambda: gr.Info("Caption copied"), js=copy_js(c["caption"]), queue=False)
                bo.click(lambda p=c["video"]: open_in_explorer(p), queue=False)


def build() -> gr.Blocks:
    cfg = read_config()
    camp_names = ["none"] + list(campaigns())
    with gr.Blocks(title="Clipper") as demo:
        with gr.Tabs():
            # ---------------------------------------------------------------- Make clips
            with gr.Tab("✂️ Make Clips", id="make"):
                gr.HTML('<div id="hero"><h1>✂️ Clipper</h1><p>Paste a YouTube or Twitch link (or drop video '
                        'files), press <b>Make clips</b>, get vertical captioned clips.</p></div>')
                with gr.Row(equal_height=True):
                    links = gr.Textbox(label="Paste link(s)", lines=3, max_lines=10, scale=5, elem_id="links",
                                       placeholder="https://www.youtube.com/watch?v=...\nhttps://www.twitch.tv/videos/...",
                                       info="One link per line · Enter starts · Shift+Enter adds a line")
                    paste = gr.Button("📋 Paste from clipboard", scale=1, min_width=150, elem_id="paste-btn")
                    files = gr.File(label="…or drop video files here", file_count="multiple", file_types=VIDEO_TYPES,
                                    type="filepath", scale=3, height=150, elem_id="files")
                with gr.Row():
                    credit = gr.Textbox(label="Creator name (files)", placeholder="CreatorName", max_lines=1,
                                        info="Credit line in each clip's description. Optional.")
                    file_title = gr.Textbox(label="Title (files)", placeholder="the video's title", max_lines=1,
                                            info="Used for the folder name and Gemini. Empty = the file name.")
                with gr.Accordion("Options", open=False) as options:
                    with gr.Row():
                        campaign = gr.Dropdown(camp_names, value="none", label="Campaign",
                                               info="Twitch links from a campaign channel select it automatically")
                        lang = gr.Radio(LANGS, value="en", label="Language")
                        n_clips = gr.Slider(1, 10, value=int(cfg.get("num_clips") or 5), step=1,
                                            label="Clips per video")
                    section = gr.Textbox(label="Section (Twitch)", placeholder="0:30:00-2:00:00",
                                         info="Only this part of the VOD. Empty = whole VOD.", visible=False)
                make = gr.Button("✂️ Make clips", variant="primary", size="lg", elem_id="make-btn")
                status = gr.HTML()
                with gr.Accordion("Live log", open=False):
                    log_box = gr.Textbox(show_label=False, lines=14, max_lines=14, autoscroll=True,
                                         interactive=False, elem_id="log")
                clips_state = gr.State([])
                job_id = gr.State(None)

                @gr.render(inputs=clips_state)
                def _cards(clips):
                    if clips:
                        gr.Markdown("### Your clips")
                    clip_cards(clips, "")

            # ---------------------------------------------------------------- Live
            with gr.Tab("🔴 Live", id="live"):
                gr.HTML('<div id="hero"><h1>🔴 Live</h1><p>Clip a Twitch or YouTube stream while it\'s still '
                        'live. It\'s recorded in 5-minute chunks; every time one closes, the last 10 minutes are '
                        'clipped and the clips appear below.</p></div>')
                with gr.Row(equal_height=True):
                    live_url = gr.Textbox(label="Live link", scale=5, max_lines=1,
                                          placeholder="https://www.twitch.tv/<channel>  or  https://www.youtube.com/watch?v=...",
                                          info="A Twitch channel or a YouTube live stream")
                    live_per = gr.Slider(1, 5, value=int(cfg.get("live_clips_per_window") or 2), step=1, scale=2,
                                         label="Clips per window", info="from each 10-minute window")
                with gr.Row():
                    live_campaign = gr.Dropdown(camp_names, value="none", label="Campaign",
                                                info="Source check, facecam layout and overlay still apply")
                    live_lang = gr.Radio(LANGS, value="en", label="Language")
                with gr.Row():
                    live_go = gr.Button("🔴 Start", variant="primary", elem_id="live-start", scale=3)
                    live_halt = gr.Button("⏹️ Stop", variant="stop", elem_id="live-stop", scale=1)
                gr.Markdown("Stop ends the recording; the part already recorded is still clipped. Windows wait "
                            "behind link/file jobs (one GPU job at a time).", elem_classes="empty-note")
                live_status = gr.HTML()
                with gr.Accordion("Live log", open=False):
                    live_log = gr.Textbox(show_label=False, lines=14, max_lines=14, autoscroll=True,
                                          interactive=False, elem_id="live-log")
                live_state = gr.State([])
                live_timer = gr.Timer(2.0)

                @gr.render(inputs=live_state)
                def _live_cards(clips):
                    if clips:
                        gr.Markdown(f"### Clips from this stream ({len(clips)})")
                    clip_cards(list(reversed(clips)), "")

            # ---------------------------------------------------------------- History
            with gr.Tab("🕘 History", id="history") as history_tab:
                with gr.Row(equal_height=True):
                    days = gr.Radio(choices=date_folders(), label="Pick a day (newest first)", type="value",
                                    scale=6, elem_id="days")
                    with gr.Column(scale=1, min_width=200):
                        refresh = gr.Button("🔄 Refresh", size="sm")
                        open_day = gr.Button("📂 Open this day's folder", size="sm")
                h_clips = gr.State([])

                @gr.render(inputs=h_clips)
                def _h_cards(clips):
                    clip_cards(clips, "Pick a day above to see its clips.")

            # ---------------------------------------------------------------- Settings
            with gr.Tab("⚙️ Settings", id="settings"):
                with gr.Column(elem_classes="set-card"):
                    gr.Markdown("### 🔑 Gemini API key\nPicks the best moments and writes the titles. Free.")
                    key = gr.Textbox(value=mask_key(cfg.get("gemini_api_key", "")), label="Gemini API key",
                                     placeholder="Paste your key here", max_lines=1,
                                     info="Only the last 4 characters are shown. Select all and paste to replace it.")
                    with gr.Row():
                        save_key_btn = gr.Button("💾 Save key", variant="primary")
                        test_btn = gr.Button("🧪 Test key")
                        gr.Button("🔑 Get a free key", link="https://aistudio.google.com/apikey",
                                  link_target="_blank")
                    key_msg = gr.Markdown(key_status(cfg.get("gemini_api_key", "")))
                with gr.Column(elem_classes="set-card"):
                    gr.Markdown("### 🎬 Clip settings")
                    model = gr.Dropdown(GEMINI_MODELS, value=cfg.get("gemini_model") or GEMINI_MODELS[0],
                                        label="Gemini model", allow_custom_value=True)
                    folder = gr.Textbox(value=cfg.get("output_root") or DEFAULT_OUT, label="Output folder",
                                        info="Clips go to <folder>\\<date>\\<video>\\")
                    with gr.Row():
                        min_len = gr.Number(value=int(cfg["min_clip_seconds"]), label="Min clip length (s)", precision=0)
                        max_len = gr.Number(value=int(cfg["max_clip_seconds"]), label="Max clip length (s)", precision=0)
                        def_clips = gr.Slider(1, 10, value=int(cfg.get("num_clips") or 5), step=1,
                                              label="Default clips per video")
                    with gr.Row():
                        layout = gr.Radio(LAYOUTS, value=cfg.get("layout") or "auto", label="Layout",
                                          info="auto = follows the speaker on talking-head clips, blur otherwise · blur = full frame on blurred "
                                               "background · crop = 9:16 crop that follows faces · facecam = cam on top")
                        cookies = gr.Radio(["firefox", "chrome", "safari", "none"], value=cfg.get("cookies_from_browser") or "none",
                                           label="Cookies browser", info="YouTube bot check: read cookies from this browser")
                    pro = gr.CheckboxGroup(list(PRO_EDIT), label="Pro edit",
                                           value=[l for l, k in PRO_EDIT.items() if cfg.get(k, True)],
                                           info="Streams with a webcam: cam on top + gameplay below, a cut to the "
                                                "streamer's face on the loudest reactions, and one big word behind "
                                                "their head during the first one")
                with gr.Column(elem_classes="set-card"):
                    gr.Markdown("### 📐 Output quality")
                    with gr.Row():
                        res = gr.Radio(RES_CHOICES, value=res_label(cfg), label="Resolution",
                                       info="1080p = 1080x1920 · 1440p = 1440x2560 · 4K = 2160x3840")
                        fps = gr.Radio(FPS_CHOICES, value=str(cfg.get("output_fps") or "source"), label="Frame rate",
                                       info=f"Source = the video's own rate (up to {int(cfg.get('max_fps') or 60)} fps)")
                        fsize = gr.Radio(SIZE_CHOICES, value=size_choice(cfg), label="File size",
                                         info="Smaller = faster uploads · Best = cleanest master")
                    estimate = gr.Markdown(estimate_text(res_label(cfg), str(cfg.get("output_fps") or "source"),
                                                         size_choice(cfg)))
                    gr.Markdown("4K only adds real detail when the source is 1440p or higher (Twitch is 1080p), "
                                "but YouTube encodes 4K uploads with a better codec, so they can still look "
                                "cleaner after upload. 1440p and 4K take longer to render.", elem_classes="empty-note")
                save_btn = gr.Button("💾 Save settings", variant="primary", size="lg")
                settings_msg = gr.Markdown()

        # ---------------------------------------------------------------- events
        paste.click(None, inputs=links, outputs=links, js="""async (cur) => {
            try { const t = (await navigator.clipboard.readText()).trim();
                  if (!t) return cur; return cur && cur.trim() ? cur.trimEnd() + '\\n' + t : t; }
            catch (e) { console.warn('Clipboard blocked - press Ctrl+V in the box instead.', e); return cur; } }""")
        links.change(detect, inputs=links, outputs=[campaign, section, options],
                     trigger_mode="always_last", show_progress="minimal")
        run_args = dict(fn=start, inputs=[links, files, credit, file_title, campaign, lang, n_clips, section],
                        outputs=[log_box, status, clips_state, job_id],
                        concurrency_limit=1, concurrency_id="gpu", show_progress="minimal")
        make.click(**run_args)
        links.submit(**run_args)
        demo.load(reattach, outputs=[log_box, status, clips_state, job_id], show_progress="hidden")

        # live
        live_url.change(lambda t: detect(t)[0], inputs=live_url, outputs=live_campaign,
                        trigger_mode="always_last", show_progress="hidden")
        live_go.click(live_start, [live_url, live_campaign, live_lang, live_per], live_status, show_progress="minimal")
        live_url.submit(live_start, [live_url, live_campaign, live_lang, live_per], live_status, show_progress="minimal")
        live_halt.click(live_stop, None, live_status, show_progress="hidden")
        live_timer.tick(live_tick, live_state, [live_status, live_log, live_state], show_progress="hidden",
                        queue=False)

        def load_settings():   # fresh from config.json on every page load (never the startup values)
            c = read_config()
            k = c.get("gemini_api_key", "")
            n = int(c.get("num_clips") or 5)
            return (mask_key(k), key_status(k), c.get("gemini_model") or GEMINI_MODELS[0],
                    c.get("output_root") or DEFAULT_OUT, int(c["min_clip_seconds"]), int(c["max_clip_seconds"]),
                    n, c.get("layout") or "auto", c.get("cookies_from_browser") or "none", n,
                    gr.update(choices=["none"] + list(campaigns())),
                    [l for l, k in PRO_EDIT.items() if c.get(k, True)],
                    res_label(c), str(c.get("output_fps") or "source"), size_choice(c),
                    estimate_text(res_label(c), str(c.get("output_fps") or "source"), size_choice(c)),
                    gr.update(choices=["none"] + list(campaigns())))

        demo.load(load_settings, outputs=[key, key_msg, model, folder, min_len, max_len, def_clips, layout,
                                          cookies, n_clips, campaign, pro, res, fps, fsize, estimate, live_campaign],
                  show_progress="hidden")
        for q in (res, fps, fsize):
            q.change(estimate_text, [res, fps, fsize], estimate, show_progress="hidden", queue=False)
        def_clips.change(lambda v: gr.update(value=v), def_clips, n_clips)

        def show_day(path):
            if not path:
                return []
            serve_folder(str(Path(path).parent))
            return [c for v in sorted(p for p in Path(path).iterdir() if p.is_dir()) for c in clips_in(v)]

        def refresh_days(cur):
            choices = date_folders()
            vals = [v for _, v in choices]
            sel = cur if cur in vals else (vals[0] if vals else None)
            return gr.update(choices=choices, value=sel), show_day(sel)

        refresh.click(refresh_days, days, [days, h_clips])
        history_tab.select(refresh_days, days, [days, h_clips])
        days.change(show_day, days, h_clips)
        open_day.click(lambda p: open_in_explorer(p) if p else gr.Warning("Pick a day first."), days, None)

        save_key_btn.click(save_key, key, [key, key_msg])
        test_btn.click(test_key, [key, model], key_msg, show_progress="minimal")
        save_btn.click(save_settings, [key, model, folder, min_len, max_len, def_clips, layout, cookies, pro,
                                       res, fps, fsize], [key, settings_msg])
    return demo


def main() -> None:
    serve_folder(output_root())
    demo = build()
    DEMO[0] = demo
    demo.queue(default_concurrency_limit=4)
    theme = gr.themes.Soft(primary_hue="violet", secondary_hue="indigo", neutral_hue="slate",
                           font=[gr.themes.GoogleFont("Inter"), "Segoe UI", "sans-serif"])
    demo.launch(server_name=HOST, server_port=PORT, share=False, inbrowser=False, show_error=True,
                theme=theme, css=CSS, js=JS, footer_links=[], allowed_paths=[str(Path(output_root()).resolve())],
                favicon_path=str(ICON_PATH) if ICON_PATH.exists() else None)


if __name__ == "__main__":
    main()
