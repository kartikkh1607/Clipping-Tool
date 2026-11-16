"""Live mode without a live stream: the recorder cuts generated footage into chunks, and the sliding windows
are clipped with no moment clipped twice."""
import re
from pathlib import Path

import live
import clipper
from test_render import FAST


def _cfg(tmp_path, **extra):
    return {**clipper.DEFAULTS, **FAST, "output_root": str(tmp_path / "out"), "min_clip_seconds": 3,
            "max_clip_seconds": 4, "gemini_api_key": "", "whisper_device": "cpu", "whisper_cpu_model": "none",
            "campaign": "", "section": None, "verbose": False, "_lang": "en", "render_workers": 2, **extra}


def test_fake_live_run_sliding_windows_and_no_duplicates(media, tmp_path):
    cfg = _cfg(tmp_path)
    windows = []
    sess = live.LiveSession("https://www.twitch.tv/tester", cfg, clips_per_window=2, work=tmp_path / "work",
                            info={"channel": "Tester", "title": "Test stream", "started": None,
                                  "webpage_url": "https://www.twitch.tv/tester"}, on_window=windows.append)
    calls = []

    def resolve():   # the "stream": 30 s of generated footage, then it goes offline
        calls.append(1)
        if len(calls) > 1:
            raise live.StreamOffline("offline")
        return {"url": str(media / "src60.mp4"), "headers": {}}

    rec = sess.start_recording(resolve=resolve, chunk_seconds=10)
    rec.thread.join(120)
    assert rec.state == "ended" and sess.ended and rec.reconnects == 0   # an end, not a hiccup
    assert [c.path.suffix for c in sess.chunks] == [".ts"] * 3
    assert [round(c.dur) for c in sess.chunks] == [10, 10, 10]
    # every closed chunk opens a window of the last 2 chunks
    assert [[c.index for c in w.chunks] for w in windows] == [[0], [0, 1], [1, 2]]
    assert windows[1].rec_start == 0.0 and abs(windows[2].rec_start - sess.chunks[1].rec_start) < 1e-6

    for w in windows:
        sess.process_window(w, dict(cfg))
    assert [w.status for w in windows] == ["done"] * 3
    made = sorted(sess.made, key=lambda m: m["rec_start"])
    assert len(made) >= 3
    for a, b in zip(made, made[1:]):                 # nothing clipped twice: no two clips overlap in time
        assert a["rec_end"] <= b["rec_start"], (a, b)
    assert {m["window"] for m in made} == {1, 2, 3}   # later windows still find NEW moments

    # one folder for the stream, clips numbered on across windows, each with a .txt
    folder = sess.folder
    assert folder.parent.parent == Path(cfg["output_root"]) and re.match(r"Tester_LIVE_\d{4}", folder.name)
    mp4s = sorted(folder.glob("*.mp4"))
    assert [int(p.name[:2]) for p in mp4s] == list(range(1, len(made) + 1))
    for p in mp4s:
        txt = p.with_suffix(".txt").read_text(encoding="utf-8")
        assert "Recording time:" in txt and "twitch.tv/tester (recording time" in txt
        assert "Quality check: passed" in txt
    assert "Test stream" in (folder / "_summary.txt").read_text(encoding="utf-8")
    sess.finish()
    assert not (tmp_path / "work").exists()


def test_recorder_reconnects_and_numbers_on(media, tmp_path):
    """The stream drops once and comes back: chunks number on, times never overlap, then it ends."""
    calls, got = [], []

    def resolve():
        calls.append(1)
        if len(calls) > 2:
            raise live.StreamOffline("offline")
        return {"url": str(media / "analysis720.mp4"), "headers": {}}

    rec = live.Recorder(tmp_path / "rec", got.append, resolve, chunk_seconds=15, retry_wait=0.05).start()
    rec.thread.join(120)
    assert rec.state == "ended" and rec.reconnects == 1
    assert [c.index for c in got] == [0, 1, 2, 3]
    assert all(a.rec_end <= b.rec_start + 1e-6 for a, b in zip(got, got[1:]))
    assert sorted(p.name for p in (tmp_path / "rec").glob("chunk_*.ts")) == [f"chunk_{i:05d}.ts" for i in range(4)]


def _chunk(i, rec, dur, tmp):
    p = tmp / f"chunk_{i:05d}.ts"
    p.write_bytes(b"x")
    return live.Chunk(i, p, rec, dur)


def test_window_time_mapping_dedup_window_and_chunk_cleanup(tmp_path):
    cfg = _cfg(tmp_path)
    sess = live.LiveSession("https://www.youtube.com/watch?v=abc", cfg, 2, work=tmp_path / "w",
                            info={"channel": "Ch", "title": "T", "id": "abc", "started": 1000.0})
    # a reconnect gap between chunk 1 and 2: the window joins them back to back
    for i, (rec, dur) in enumerate([(0, 300), (300, 300), (650, 300), (950, 300), (1250, 300)]):
        sess.add_chunk(_chunk(i, rec, dur, tmp_path))
    w = sess.windows[2]                       # chunks 1 and 2
    assert w.file_to_rec(10) == 310 and w.file_to_rec(310) == 660
    assert w.rec_to_file(660) == 310 and w.rec_to_file(625) is None
    # an earlier clip at 590-605 s (end of chunk 1) is off limits in window 3 (file time 290-300)
    sess.made = [{"rec_start": 590.0, "rec_end": 605.0}, {"rec_start": 100.0, "rec_end": 110.0}]
    wc = sess.window_cfg(w, cfg)
    assert wc["_avoid"] == [(290.0, 300.0)] and wc["_first_number"] == 1 and wc["num_clips"] == 2
    assert wc["skip_end_seconds"] == 30                          # not the last window of an ended stream
    # stream time = wall clock since the stream went live: links point to the live video at that time
    for c in sess.chunks:
        c.wall = 2000.0 + c.rec_start
    assert sess.stream_time(0) == 1000.0
    assert sess.link(30) == "https://www.youtube.com/watch?v=abc&t=1030s"
    assert wc["_meta"]["_time_label"] == "Stream time"
    # cleanup keeps the last 3 chunks plus any a waiting window still needs
    for x in sess.windows[:3]:
        x.status = "done"
    sess.cleanup()
    assert [c.path.exists() for c in sess.chunks] == [False, False, True, True, True]
    sess.chunks[2].path.unlink()
    sess.windows[3].status = "queued"   # needs chunks 2 and 3
    sess.cleanup()
    assert sess.chunks[3].path.exists()
    sess2 = live.LiveSession("u", {**cfg, "keep_source": True}, 2, work=tmp_path / "w2", info={})
    c = _chunk(9, 0, 10, tmp_path)
    sess2.add_chunk(c)
    sess2.windows[0].status = "done"
    sess2.cleanup()
    assert c.path.exists()


def test_stub_chunk_number_is_not_reused(tmp_path):
    """A <1 s stub left by a drop is deleted, and its number is used up: the next ffmpeg run must not write a
    real chunk under the same name (it would be skipped as already seen and 5 minutes would be lost)."""
    import time
    rec = live.Recorder(tmp_path, on_chunk=lambda c: None, resolve=lambda: {})
    (tmp_path / "chunk_00000.ts").write_bytes(b"\x47" + b"\x00" * 4000)   # not a playable chunk: probes as 0 s
    rec._emit("chunk_00000.ts", None, None, time.time())
    assert not (tmp_path / "chunk_00000.ts").exists() and rec.chunks == []
    assert rec._next == 1
    assert "-segment_start_number" in (cmd := rec._cmd({"url": "in.ts"}, 2)) and cmd[cmd.index("-segment_start_number") + 1] == "1"
