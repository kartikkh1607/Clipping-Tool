"""Pure-logic tests: no video needed."""
import random
import types
import wave

import numpy as np
import pytest

import clipper


def words_track(gap=0.1, n=400, per_sentence=8):
    words, t = [], 0.0
    for i in range(n):
        words.append([round(t, 2), round(t + 0.4, 2), " word" + ("." if i % per_sentence == per_sentence - 1 else "")])
        t += 0.4 + gap
    segs = [{"start": words[i][0], "end": words[i + per_sentence - 1][1], "text": "",
             "words": words[i:i + per_sentence]} for i in range(0, n, per_sentence)]
    return words, segs


def inside_word(words, t):
    return any(w[0] + 1e-6 < t < w[1] - 1e-6 for w in words)


# ----------------------------------------------------------------------------- cuts
@pytest.mark.parametrize("gap", [0.0, 0.02, 0.1, 0.6])
def test_refine_cut_never_splits_words_and_respects_length(gap):
    words, segs = words_track(gap)
    total = words[-1][1]
    rng = random.Random(gap)
    for _ in range(1500):
        s = rng.uniform(0, total - 50)
        e = s + rng.uniform(15, 50)
        payoff = rng.choice([None, rng.uniform(s - 5, e + 5)])
        a, b = clipper.refine_cut(s, e, segs, 20, 40, total, payoff)
        assert not inside_word(words, a) and not inside_word(words, b), (s, e, a, b)
        assert 19.99 <= b - a <= 40.01, (s, e, a, b)


def test_refine_cut_keeps_payoff_and_finishes_sentence():
    words, segs = words_track(0.1)
    a, b = clipper.refine_cut(10.0, 30.0, segs, 20, 40, 200, payoff=33.1)
    assert a + 0.3 <= 33.1 <= b - 1.5
    last = max((w for w in words if w[1] <= b + 1e-6), key=lambda w: w[1])
    assert last[2].endswith(".")


def test_refine_cut_trims_long_clip_on_a_sentence():
    words, segs = words_track(0.1)
    a, b = clipper.refine_cut(5.0, 52.3, segs, 20, 40, 200)
    assert b - a <= 40
    last = max((w for w in words if w[1] <= b + 1e-6), key=lambda w: w[1])
    assert last[2].endswith(".")


def test_refine_cut_without_transcript_only_enforces_rules():
    assert clipper.refine_cut(10.13, 30.77, [], 20, 40, 200) == (10.13, 30.77)


# ----------------------------------------------------------------------------- scoring inputs
def test_hype_words_skip_everyday_words():
    for common in ("what", "yes", "run", "win", "bro", "kya", "wait", "stop"):
        assert common not in clipper.HYPE_WORDS
    segs = [{"start": 0, "end": 3, "text": "what? yes what", "words": [[0, 0.5, " what?"], [1, 1.5, " yes"], [2, 2.5, " what"]]}]
    hype, _ = clipper.hype_per_second(segs, 5)
    assert hype.max() == 0   # questions and everyday words are not hype any more


def test_loudness_streaming_matches_whole_file(tmp_path):
    sr = 16000
    rng = np.random.default_rng(0)
    data = (rng.standard_normal(sr * 700) * np.repeat(rng.uniform(100, 8000, 700), sr)).clip(-32000, 32000)
    path = tmp_path / "a.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr)
        w.writeframes(data.astype(np.int16).tobytes())
    got = clipper.loudness_per_second(path, 700)
    # reference: the old whole-file implementation
    x = data.astype(np.int16).astype(np.float32)
    hop = sr // 2
    rms = np.sqrt((x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(axis=1)) + 1e-6
    db = 20 * np.log10(rms)
    lo, hi = np.percentile(db, 10), np.percentile(db, 99)
    ref = np.clip((db - lo) / max(hi - lo, 1e-6), 0, 1).reshape(-1, 2).mean(axis=1)
    assert np.allclose(got, ref[:700], atol=1e-4)


# ----------------------------------------------------------------------------- formats / fps
def test_format_prefers_sdr():
    fmt, sort = clipper.format_spec("https://www.youtube.com/watch?v=x", 2160)
    assert sort[0] == "hdr:sdr" and "res:2160" in sort
    info = {"formats": [{"vcodec": "vp9", "height": 2160, "fps": 60, "dynamic_range": "HDR10"},
                        {"vcodec": "vp9", "height": 1440, "fps": 60, "dynamic_range": "SDR"},
                        {"vcodec": "none", "acodec": "opus"}]}
    assert clipper.best_available(info) == (1440, 60.0)


@pytest.mark.parametrize("src,want", [(59.94, "60000/1001"), (60.0, "60"), (29.97, "30000/1001"),
                                      (25.0, "25"), (120.0, "60"), (23.976, "24000/1001")])
def test_output_fps(src, want):
    assert clipper.output_fps(src, 60)[0] == want


# ----------------------------------------------------------------------------- output files
def test_folder_numbers_never_reused(tmp_path):
    cfg = {"output_root": str(tmp_path)}
    a = clipper.make_video_folder(cfg, {"channel": "A", "title": "one"})
    b = clipper.make_video_folder(cfg, {"channel": "B", "title": "two"})
    a.rmdir()
    c = clipper.make_video_folder(cfg, {"channel": "C", "title": "three"})
    assert [p.name[:2] for p in (a, b, c)] == ["01", "02", "03"]


def test_banner_cannot_inject_ass_codes(tmp_path):
    clip = {"start": 0.0, "end": 5.0, "banner": "HE DID {\\b0}NOT\\N SEE \\hTHAT", "words": []}
    cfg = {**clipper.DEFAULTS, "_layout": "blur"}
    clipper.write_captions(clip, [], cfg, tmp_path / "c.ass", tmp_path / "c.srt")
    line = [l for l in (tmp_path / "c.ass").read_text().splitlines() if ",Banner," in l][0]
    assert line.endswith(",HE DID B0NOT SEE THAT")


# ----------------------------------------------------------------------------- Gemini picks
def test_gemini_picks_are_snapped_and_never_overlap(monkeypatch):
    words, segs = words_track(0.05)
    pool = [{"start": 10.0 + 60 * i, "end": 35.0 + 60 * i, "score": 1.0, "heat": 0, "loud": 0.5, "hype": 0,
             "payoff_abs": 30.0 + 60 * i if i == 0 else None} for i in range(3)]
    fake = {"clips": [{"id": 0, "start": 10.37, "end": 28.81, "title": "a", "banner": "b"},     # payoff 30 outside
                      {"id": 1, "start": 70.2, "end": 95.33, "title": "c", "banner": "d"},
                      {"id": 2, "start": 85.0, "end": 110.0, "title": "e", "banner": "f"},      # drifted 25 s
                      {"id": 1, "start": 72.0, "end": 96.0, "title": "dup", "banner": "x"}]}   # overlaps id 1
    genai_mod = types.SimpleNamespace(Client=lambda **kw: object(), types=types.SimpleNamespace())
    google_mod = types.SimpleNamespace(genai=genai_mod)
    monkeypatch.setitem(__import__("sys").modules, "google", google_mod)
    monkeypatch.setitem(__import__("sys").modules, "google.genai", genai_mod)
    monkeypatch.setitem(__import__("sys").modules, "google.genai.types", genai_mod.types)
    monkeypatch.setattr(clipper, "gemini_json", lambda *a, **k: fake)
    cfg = {**clipper.DEFAULTS, "gemini_api_key": "k" * 39, "num_clips": 5}
    meta = {"title": "t", "channel": "c", "duration": 200.0, "twitch": False}
    out = clipper.gemini_pick(pool, segs, meta, cfg)
    assert [c["title"] for c in out] == ["a", "c"]
    for c in out:
        assert not inside_word(words, c["start"]) and not inside_word(words, c["end"])
        assert 20 <= c["end"] - c["start"] <= 40
    assert out[0]["start"] + 0.3 <= 30.0 <= out[0]["end"] - 1.5
