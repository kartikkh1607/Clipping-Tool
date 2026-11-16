"""
Live mode: clip a Twitch or YouTube stream while it's still live (used by clipper.py --live and app.py).

  Recorder     - gets the stream URL with yt-dlp and records it with ffmpeg "-c copy" into numbered
                 5-minute .ts chunks (segment muxer). Reconnects when the stream hiccups or the URL expires,
                 and stops cleanly when the stream ends or stop() is called.
  LiveSession  - every time a chunk closes, a window of the last 2 chunks (10 minutes, sliding, so a moment
                 on a chunk boundary is never cut in half) is joined and run through the normal pipeline as a
                 local file. Moments already clipped from an earlier window are skipped (tracked in recording
                 time), and all clips go to one "<channel>_LIVE_<time>" folder in today's date folder.

Times: "recording time" is seconds since recording started (always known, used for de-duplication);
"stream time" is seconds since the stream went live (when the platform reports the start), used for the
.txt files and VOD links.
"""
from __future__ import annotations

import csv
import datetime as dt
import queue
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import clipper

CHUNK_SECONDS = 300          # 5-minute chunks
WINDOW_CHUNKS = 2            # a window = the last 2 chunks (10 minutes)
KEEP_CHUNKS = 3              # recorded chunks kept on disk (unless keep_source)
RECONNECT_TRIES = 8          # tries to get a dropped stream back before calling it ended


class StreamOffline(RuntimeError):
    """The stream isn't live (never was, or it has ended)."""


# ----------------------------------------------------------------------------- stream info
def _offline(msg: str) -> bool:
    low = msg.lower()
    return any(k in low for k in ("offline", "not currently live", "is not live", "not live", "has ended",
                                  "this live event will begin", "premieres in", "live stream recording is not available"))


def stream_info(url: str, cfg: dict) -> dict:
    """Live stream details + a recordable URL (HLS). Raises StreamOffline when it isn't live."""
    import yt_dlp

    h = int(cfg["twitch_hq_max_height"] if clipper.is_twitch(url) else cfg["hq_max_height"])
    opts = {**clipper.ydl_base_opts(url, cfg), "quiet": True, "noprogress": True,
            "format": f"b[height<={h}][protocol^=m3u8]/b[height<={h}]/b", "format_sort": ["hdr:sdr", f"res:{h}", "fps"]}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        if _offline(str(e)):
            raise StreamOffline(str(e).strip().splitlines()[0][:200]) from e
        raise
    if not info.get("is_live") and info.get("live_status") not in ("is_live", "post_live"):
        raise StreamOffline("This link isn't a live stream right now. For a finished stream, paste its VOD link on "
                            "the Make Clips tab.")
    fmt = info
    if not fmt.get("url"):
        fmt = next((d for d in info.get("requested_formats") or [] if d.get("url")), {})
    if not fmt.get("url"):
        raise RuntimeError("yt-dlp found the stream but no recordable URL")
    start = info.get("release_timestamp") or info.get("timestamp")
    return {
        "url": fmt["url"], "headers": fmt.get("http_headers") or info.get("http_headers") or {},
        "title": re.sub(r"\s+\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", "", info.get("title") or "live"),   # yt-dlp adds the date
        "channel": info.get("channel") or info.get("uploader") or info.get("uploader_id") or "streamer",
        "channel_id": (info.get("uploader_id") or info.get("channel_id") or "").lower(),
        "channel_url": info.get("channel_url") or info.get("uploader_url") or "",
        "id": info.get("id") or "", "webpage_url": info.get("webpage_url") or url,
        "started": float(start) if start else None, "height": fmt.get("height") or info.get("height"),
        "fps": fmt.get("fps") or info.get("fps"), "raw": info,
    }


def twitch_live_vod(channel: str, started: float | None, title: str, cfg: dict) -> str:
    """ID of the VOD Twitch is recording for the current stream ('' if it can't be confirmed)."""
    if not channel:
        return ""
    import yt_dlp
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True, "playlistend": 3,
                               "socket_timeout": 15}) as ydl:
            info = ydl.extract_info(f"https://www.twitch.tv/{channel}/videos?filter=archives&sort=time", download=False)
    except Exception:
        return ""
    for e in info.get("entries") or []:
        vid = str(e.get("id") or "").lstrip("v")
        ts = e.get("timestamp") or e.get("release_timestamp")
        same_time = started and ts and abs(float(ts) - started) < 15 * 60
        same_title = title and (e.get("title") or "").strip().lower().startswith(title.strip().lower()[:40])
        if vid.isdigit() and (same_time or (not ts and same_title)):
            return vid
    return ""


# ----------------------------------------------------------------------------- recorder
@dataclass
class Chunk:
    index: int
    path: Path
    rec_start: float          # recording time (s since recording started) of its first frame
    dur: float
    wall: float = 0.0         # wall-clock time of its first frame

    @property
    def rec_end(self) -> float:
        return self.rec_start + self.dur


class Recorder:
    """Records a live stream into chunk_00000.ts, chunk_00001.ts, ... and calls on_chunk(Chunk) as each closes.

    resolve(): returns {"url", "headers"} (raises StreamOffline when the stream is over) - stream_info by default.
    """

    def __init__(self, work: Path, on_chunk: Callable[[Chunk], None], resolve: Callable[[], dict],
                 chunk_seconds: float = CHUNK_SECONDS, log: Callable[[str], None] = clipper.log,
                 on_end: Callable[[str], None] | None = None, retry_wait: float = 5.0):
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.on_chunk, self.resolve, self.log, self.on_end = on_chunk, resolve, log, on_end
        self.chunk_seconds = float(chunk_seconds)
        self.retry_wait = retry_wait
        self.state = "starting"      # starting | recording | reconnecting | ended | stopped | failed
        self.error = ""
        self.reconnects = 0
        self.chunks: list[Chunk] = []
        self.t0: float | None = None          # wall time recording started
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None   # the running ffmpeg
        self._next = 0
        self._seen: set[str] = set()
        self.thread = threading.Thread(target=self._run, name="clipper-live-recorder", daemon=True)

    # -- public
    def start(self) -> "Recorder":
        self.thread.start()
        return self

    def stop(self) -> None:
        """Finish the current chunk and stop (the chunk recorded so far is kept and handed on)."""
        self._stop.set()
        p = self._proc
        if p and p.poll() is None:
            try:
                p.stdin.write(b"q")
                p.stdin.flush()
            except Exception:
                pass

    @property
    def recording_seconds(self) -> float:
        if self.t0 is None:
            return 0.0
        done = self.chunks[-1].rec_end if self.chunks else 0.0
        return max(done, time.time() - self.t0) if self.state in ("recording", "reconnecting") else done

    # -- internals
    def _cmd(self, src: dict, run_no: int) -> list[str]:
        # no -nostdin: stop() asks ffmpeg to finish the current chunk by sending "q" on stdin
        cmd = ["ffmpeg", "-hide_banner", "-v", "error", "-y"]
        hdrs = dict(src.get("headers") or {})
        ua = hdrs.pop("User-Agent", None)
        url = str(src["url"])
        if url.startswith("http"):
            cmd += ["-rw_timeout", "20000000", "-reconnect", "1", "-reconnect_streamed", "1",
                    "-reconnect_on_network_error", "1", "-reconnect_delay_max", "10"]
            if ua:
                cmd += ["-user_agent", ua]
            if hdrs:
                cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in hdrs.items())]
        cmd += ["-i", url, "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-f", "segment",
                "-segment_time", f"{self.chunk_seconds:g}", "-reset_timestamps", "1",
                "-segment_start_number", str(self._next), "-segment_list", str(self.work / f"run_{run_no:03d}.csv"),
                "-segment_list_type", "csv", "-segment_list_flags", "+live", "-segment_format", "mpegts",
                str(self.work / "chunk_%05d.ts")]
        return cmd

    def _emit(self, name: str, start: float | None, end: float | None, run_wall: float) -> None:
        """start/end: the chunk's times in its ffmpeg run (from the segment list); None = unknown (a chunk
        left behind by a run that died), then its length is probed and it's taken to end now."""
        if name in self._seen:
            return
        p = self.work / name
        if not p.exists() or p.stat().st_size < 1024:
            return
        self._seen.add(name)
        dur = (end - start) if (start is not None and end is not None) else 0.0
        if dur <= 0:
            try:
                dur = clipper.probe_duration(p)
            except Exception:
                dur = 0.0
        if dur < 1.0:   # a stub left by a drop: nothing to clip
            p.unlink(missing_ok=True)
            return
        wall = run_wall + start if start is not None else time.time() - dur
        idx = int(re.search(r"(\d+)", name).group(1))
        prev_end = self.chunks[-1].rec_end if self.chunks else 0.0
        rec = max(prev_end, wall - (self.t0 or wall))   # never overlap the previous chunk
        ch = Chunk(idx, p, round(rec, 3), round(dur, 3), wall)
        self.chunks.append(ch)
        self._next = max(self._next, idx + 1)
        self.log(f"Live: chunk {idx + 1} recorded ({clipper.fmt_ts(ch.rec_start)}-{clipper.fmt_ts(ch.rec_end)} "
                 f"recording time)")
        try:
            self.on_chunk(ch)
        except Exception as e:   # a processing problem must never stop the recording
            self.log(f"Live: couldn't queue chunk {idx + 1} ({e})")

    def _read_list(self, path: Path, run_wall: float) -> None:
        if not path.exists():
            return
        try:
            rows = list(csv.reader(path.read_text(encoding="utf-8", errors="replace").splitlines()))
        except Exception:
            return
        for r in rows:
            if len(r) >= 3:
                try:
                    self._emit(Path(r[0]).name, float(r[1]), float(r[2]), run_wall)
                except ValueError:
                    continue

    def _run(self) -> None:
        run_no, tries, dropped = 0, 0, False
        try:
            while not self._stop.is_set():
                try:
                    src = self.resolve()
                    if dropped:   # it's back: that was a hiccup, not the end
                        self.reconnects += 1
                        dropped = False
                        self.log("Live: reconnected")
                except StreamOffline as e:
                    if not self.chunks and tries == 0:
                        self.state, self.error = "failed", str(e)
                        self.log(f"Live: {e}")
                        return
                    self.log("Live: the stream has ended")
                    self.state = "ended"
                    return
                except Exception as e:
                    tries += 1
                    self.error = str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__
                    if tries > RECONNECT_TRIES:
                        self.state = "failed" if not self.chunks else "ended"
                        self.log(f"Live: giving up after {tries - 1} tries ({self.error})")
                        return
                    self.state = "reconnecting"
                    self.log(f"Live: couldn't get the stream ({self.error}) - retrying ({tries}/{RECONNECT_TRIES})")
                    self._stop.wait(self.retry_wait * min(tries, 6))
                    continue
                run_no += 1
                lst = self.work / f"run_{run_no:03d}.csv"
                run_wall = time.time()
                if self.t0 is None:
                    self.t0 = run_wall
                self.state = "recording"
                self._proc = subprocess.Popen(self._cmd(src, run_no), stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.PIPE)
                err_lines: list[str] = []
                threading.Thread(target=lambda p=self._proc: err_lines.extend(
                    l.decode("utf-8", "replace").strip() for l in p.stderr), daemon=True).start()
                started = time.time()
                while self._proc.poll() is None:
                    self._read_list(lst, run_wall)
                    if self._stop.wait(1.0):
                        self.stop()
                        try:
                            self._proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            self._proc.kill()
                        break
                self._proc.wait()
                self._read_list(lst, run_wall)
                for p in sorted(self.work.glob("chunk_*.ts")):   # the last chunk of a run that died mid-way
                    if p.name not in self._seen and int(re.search(r"(\d+)", p.name).group(1)) >= self._next:
                        self._emit(p.name, None, None, run_wall)
                if self._stop.is_set():
                    self.state = "stopped"
                    return
                # ffmpeg ended by itself: the stream ended, or it dropped / the URL expired
                tail = " | ".join(l for l in err_lines[-3:] if l)
                ran = time.time() - started
                tries = 0 if ran > 60 else tries + 1
                if tries > RECONNECT_TRIES:
                    self.state = "ended"
                    self.log(f"Live: the stream keeps dropping ({tail[:160]}) - stopping")
                    return
                dropped = True
                self.state = "reconnecting"
                self.log("Live: the stream stopped sending" + (f" ({tail[:160]})" if tail else "")
                         + " - checking whether it's still live")
                self._stop.wait(min(self.retry_wait * max(tries, 1), 30))
        except Exception as e:
            self.state, self.error = "failed", str(e)[:300]
            self.log(f"Live: recorder failed ({e})")
        finally:
            if self.on_end:
                try:
                    self.on_end(self.state)
                except Exception:
                    pass


# ----------------------------------------------------------------------------- windows + processing
@dataclass
class Window:
    number: int
    chunks: list[Chunk]
    status: str = "queued"          # queued | running | done | failed
    clips: list[dict] = field(default_factory=list)
    error: str = ""

    @property
    def rec_start(self) -> float:
        return self.chunks[0].rec_start

    @property
    def rec_end(self) -> float:
        return self.chunks[-1].rec_end

    def file_to_rec(self, t: float) -> float:
        """Window file time -> recording time (the chunks are joined back to back, gaps removed)."""
        off = 0.0
        for c in self.chunks:
            if t < off + c.dur or c is self.chunks[-1]:
                return c.rec_start + (t - off)
            off += c.dur
        return t

    def rec_to_file(self, r: float) -> float | None:
        off = 0.0
        for c in self.chunks:
            if c.rec_start - 0.01 <= r <= c.rec_end + 0.01:
                return off + (r - c.rec_start)
            off += c.dur
        return None

    def label(self) -> str:
        return f"{clipper.fmt_ts(self.rec_start)}-{clipper.fmt_ts(self.rec_end)}"


class LiveSession:
    """One live stream being clipped: recorder + sliding windows + the clips made so far."""

    def __init__(self, url: str, cfg: dict, clips_per_window: int = 2, out_folder: Path | None = None,
                 work: Path | None = None, info: dict | None = None, log: Callable[[str], None] = clipper.log,
                 on_window: Callable[["Window"], None] | None = None):
        self.url, self.cfg, self.log = url, cfg, log
        self.clips_per_window = max(1, int(clips_per_window))
        self.info = info or {}
        self.on_window = on_window            # called with each new window (app: queue it behind other jobs)
        self.started_wall = time.time()
        name = clipper.slugify(self.info.get("channel") or "stream", 20)
        if out_folder is None:
            day = Path(cfg["output_root"]) / dt.date.today().strftime("%d-%m-%Y")
            out_folder = day / f"{name}_LIVE_{dt.datetime.now():%H%M}"
            n = 2
            while out_folder.exists():
                out_folder = day / f"{name}_LIVE_{dt.datetime.now():%H%M}_{n}"
                n += 1
        self.folder = Path(out_folder)
        self.work = Path(work) if work else Path(cfg["output_root"]) / "_work" / f"live_{dt.datetime.now():%Y%m%d_%H%M%S}"
        self.chunks: list[Chunk] = []
        self.windows: list[Window] = []
        self.made: list[dict] = []            # {"rec_start", "rec_end", "title", "number"}
        self.errors: list[str] = []
        self.recorder: Recorder | None = None
        self.ended = False
        self.vod_id = ""
        self._vod_checked = 0.0
        self._lock = threading.Lock()

    # -- feeding chunks (from the Recorder, or directly in tests)
    def add_chunk(self, ch: Chunk) -> Window:
        with self._lock:
            self.chunks.append(ch)
            win = Window(len(self.windows) + 1, self.chunks[-WINDOW_CHUNKS:])
            self.windows.append(win)
        if self.on_window:
            self.on_window(win)
        return win

    def start_recording(self, resolve: Callable[[], dict] | None = None, chunk_seconds: float = CHUNK_SECONDS) -> Recorder:
        def default_resolve():
            fresh = stream_info(self.url, self.cfg)
            self.info.update({k: v for k, v in fresh.items() if k not in ("raw",)})
            return fresh
        self.recorder = Recorder(self.work, self.add_chunk, resolve or default_resolve, chunk_seconds, self.log,
                                 on_end=self._ended)
        return self.recorder.start()

    def stop(self) -> None:
        if self.recorder:
            self.recorder.stop()

    def _ended(self, state: str) -> None:
        self.ended = True
        if state == "failed" and self.recorder and self.recorder.error:
            self.errors.append(self.recorder.error)

    # -- status
    @property
    def pending(self) -> list[Window]:
        return [w for w in self.windows if w.status in ("queued", "running")]

    @property
    def active(self) -> bool:
        rec = self.recorder and self.recorder.state in ("starting", "recording", "reconnecting")
        return bool(rec or self.pending)

    def status(self) -> dict:
        r = self.recorder
        return {"state": r.state if r else ("ended" if self.ended else "idle"),
                "recording": r.recording_seconds if r else (self.chunks[-1].rec_end if self.chunks else 0.0),
                "chunks": len(self.chunks), "windows_done": sum(w.status == "done" for w in self.windows),
                "windows_failed": sum(w.status == "failed" for w in self.windows),
                "windows_queued": len(self.pending), "clips": len(self.made), "reconnects": r.reconnects if r else 0,
                "errors": list(self.errors[-5:]), "folder": str(self.folder)}

    # -- time + links
    def stream_time(self, rec: float) -> float | None:
        """Recording time -> seconds since the stream went live (None if the start isn't known)."""
        started = self.info.get("started")
        if not started or not self.chunks:
            return None
        first = self.chunks[0]
        return max(0.0, first.wall - first.rec_start + rec - float(started))

    def _vod(self) -> str:
        if self.vod_id or not clipper.is_twitch(self.url) or time.time() - self._vod_checked < 120:
            return self.vod_id
        self._vod_checked = time.time()
        self.vod_id = twitch_live_vod(self.info.get("channel_id") or "", self.info.get("started"),
                                      self.info.get("title") or "", self.cfg)
        if self.vod_id:
            self.log(f"Live: found the stream's VOD (videos/{self.vod_id}) - source links point to it")
        return self.vod_id

    def link(self, rec: float) -> str:
        st = self.stream_time(rec)
        page = self.info.get("webpage_url") or self.url
        if st is not None:
            t = int(st)
            if clipper.is_youtube(self.url) and self.info.get("id"):
                return f"https://www.youtube.com/watch?v={self.info['id']}&t={t}s"
            vod = self._vod()
            if vod:
                return f"https://www.twitch.tv/videos/{vod}?t={t // 3600}h{t % 3600 // 60}m{t % 60}s"
            return f"{page} (stream time {clipper.fmt_ts(t)})"
        return f"{page} (recording time {clipper.fmt_ts(rec)}; the stream's start time isn't known)"

    # -- processing one window
    def window_cfg(self, win: Window, cfg: dict) -> dict:
        """Settings for one window's run through the normal pipeline."""
        final = self.ended and win is self.windows[-1]
        made_here = []
        for m in self.made:   # the parts of earlier clips that fall inside this window's chunks, in file time
            off = 0.0
            for c in win.chunks:
                lo, hi = max(m["rec_start"], c.rec_start), min(m["rec_end"], c.rec_end)
                if lo < hi:
                    made_here.append((round(off + lo - c.rec_start, 3), round(off + hi - c.rec_start, 3)))
                off += c.dur
        known = self.stream_time(0.0) is not None
        out = {**cfg, "num_clips": self.clips_per_window, "_folder": str(self.folder), "_allow_empty": True,
               "_avoid": made_here, "_made": [], "hq_refetch": False,
               "_first_number": self._next_number(),
               "ai_pool_size": int(cfg.get("live_ai_pool_size") or 12),
               # the window's start was the middle of the previous window; its end will be the middle of the
               # next one, so moments right at the end wait for that window (unless the stream is over)
               "skip_start_seconds": 0, "skip_end_seconds": 0 if final else 30,
               "_meta": {"url": self.info.get("webpage_url") or self.url, "channel_url": self.info.get("channel_url", ""),
                         "id": "", "twitch": False,
                         "_abs": lambda t, w=win: (self.stream_time(w.file_to_rec(t)) if known else w.file_to_rec(t)),
                         "_link": lambda t, w=win: self.link(w.file_to_rec(t)),
                         "_time_label": "Stream time" if known else "Recording time"}}
        if cfg.get("creator_name"):
            out["_meta"]["channel"] = cfg["creator_name"]
        return out

    def _next_number(self) -> int:
        nums = [int(m.group(1)) for p in self.folder.glob("*.mp4") if (m := re.match(r"^(\d{2,3})_", p.name))]
        return max(nums, default=0) + 1

    def process_window(self, win: Window, cfg: dict) -> list[dict]:
        """Join the window's chunks and run the normal pipeline on them. Returns the new clips."""
        win.status = "running"
        joined = self.work / f"window_{win.number:03d}.mkv"
        lst = self.work / f"window_{win.number:03d}.txt"
        try:
            lst.write_text("".join(f"file '{c.path.resolve().as_posix()}'\n" for c in win.chunks), encoding="utf-8")
            clipper.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                         "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", str(joined)])
            wcfg = self.window_cfg(win, cfg)
            channel = wcfg.get("creator_name") or self.info.get("channel") or "streamer"
            st = self.stream_time(win.rec_start)
            when = clipper.fmt_ts(st if st is not None else win.rec_start)
            title = f"{self.info.get('title') or 'Live'} [{'stream' if st is not None else 'rec'} {when}]"
            self.log(f"Live: clipping window {win.number} ({win.label()} recording time, "
                     f"{len(win.chunks)} chunk{'s' * (len(win.chunks) != 1)})")
            clipper.process(str(joined), wcfg, True, credit=channel, local_title=title)
            new = []
            for c in wcfg["_made"]:
                rec = {"rec_start": float(win.file_to_rec(c["start"])), "rec_end": float(win.file_to_rec(c["end"])),
                       "title": c["title"], "number": c.get("number"), "window": win.number}
                new.append(rec)
            with self._lock:
                self.made.extend(new)
            win.clips, win.status = new, "done"
            self.log(f"Live: window {win.number} done - {len(new)} new clip{'s' * (len(new) != 1)}")
            return new
        except BaseException as e:
            win.status = "failed"
            msg = str(e.code if isinstance(e, SystemExit) else e).strip()
            win.error = (msg.splitlines()[0] if msg else type(e).__name__)[:300]
            self.errors.append(f"window {win.number}: {win.error}")
            raise
        finally:
            joined.unlink(missing_ok=True)
            lst.unlink(missing_ok=True)
            self.cleanup()

    def cleanup(self) -> None:
        """Delete chunks older than the last 3, except ones a waiting window still needs (keep_source keeps all)."""
        if self.cfg.get("keep_source"):
            return
        with self._lock:
            needed = {c.index for w in self.windows if w.status in ("queued", "running") for c in w.chunks}
            keep = {c.index for c in self.chunks[-KEEP_CHUNKS:]} | needed
            for c in self.chunks:
                if c.index not in keep and c.path.exists():
                    c.path.unlink(missing_ok=True)

    def finish(self) -> None:
        """After the last window: remove the work folder (chunks too, unless keep_source moves them)."""
        if self.cfg.get("keep_source"):
            dest = self.folder / "_chunks"
            dest.mkdir(parents=True, exist_ok=True)
            for c in self.chunks:
                if c.path.exists():
                    shutil.move(str(c.path), str(dest / c.path.name))
        shutil.rmtree(self.work, ignore_errors=True)


# ----------------------------------------------------------------------------- terminal version
def run_terminal(url: str, cfg: dict, clips_per_window: int = 2) -> Path:
    """clipper.py --live URL: record and clip until the stream ends (Ctrl+C stops recording; the windows
    already recorded are still clipped, a second Ctrl+C quits at once)."""
    info = stream_info(url, cfg)
    clipper.check_campaign_source(url, info["raw"], cfg)
    info.pop("raw", None)
    if cfg.get("creator_name"):
        info["channel"] = cfg["creator_name"]
    todo: "queue.Queue[Window | None]" = queue.Queue()
    sess = LiveSession(url, cfg, clips_per_window, info=info, on_window=todo.put)
    clipper.log(f"Live: '{info['title']}' by {info['channel']} - recording in {CHUNK_SECONDS // 60}-minute chunks, "
                f"{clips_per_window} clip(s) per {CHUNK_SECONDS * WINDOW_CHUNKS // 60}-minute window -> {sess.folder}")
    clipper.log("Live: press Ctrl+C to stop recording (what's recorded is still clipped)")
    used = []

    def resolve():   # the first call reuses the info fetched for the source check; reconnects refresh it
        if not used:
            used.append(1)
            return info
        fresh = stream_info(url, cfg)
        info.update({k: v for k, v in fresh.items() if k != "raw"})
        return fresh

    rec = sess.start_recording(resolve=resolve)
    stopping = False
    try:
        while True:
            try:
                win = todo.get(timeout=1.0)
            except queue.Empty:
                if sess.ended and todo.empty():
                    break
                continue
            except KeyboardInterrupt:
                raise
            try:
                sess.process_window(win, dict(cfg))
            except KeyboardInterrupt:
                raise
            except Exception as e:
                clipper.log(f"Live: window {win.number} failed ({str(e).strip()[:200]})")
    except KeyboardInterrupt:
        if not stopping:
            stopping = True
            clipper.log("Live: stopping the recording...")
            rec.stop()
            rec.thread.join(timeout=60)
            try:
                while not todo.empty():
                    win = todo.get_nowait()
                    try:
                        sess.process_window(win, dict(cfg))
                    except Exception as e:
                        clipper.log(f"Live: window {win.number} failed ({str(e).strip()[:200]})")
            except KeyboardInterrupt:
                clipper.log("Live: quitting without clipping the rest")
    finally:
        rec.stop()
        sess.finish()
    s = sess.status()
    clipper.log(f"Live: done - {clipper.fmt_ts(s['recording'])} recorded, {s['windows_done']} windows, "
                f"{s['clips']} clips -> {sess.folder}")
    return sess.folder
