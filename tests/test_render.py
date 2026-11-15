"""Real renders from generated footage: quality settings, layouts, loudness, quality check, HQ alignment."""
import json
import subprocess

import pytest

import clipper
from conftest import ffmpeg

FAST = {"x264_preset": "ultrafast", "encoder": "x264"}


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,r_frame_rate,color_primaries,color_transfer,sample_rate:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True).stdout
    return json.loads(out)


def lufs(path):
    err = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    import re
    return float(re.findall(r"I:\s+(-?[\d.]+) LUFS", err)[-1])


def render(media, tmp_path, layout, extra=None, src="src60.mp4", clip=None):
    clip = clip or {"start": 5.0, "end": 13.0, "banner": "TEST", "title": "t", "words": [[5.5, 5.9, " no"], [6.0, 6.4, " way!"]]}
    cfg = {**clipper.DEFAULTS, **FAST, "layout": layout, "_layout": layout, **(extra or {})}
    clipper.write_captions(clip, [], cfg, tmp_path / f"{layout}.ass", tmp_path / f"{layout}.srt")
    out = clipper.render_clip(media / src, clip, tmp_path, layout, cfg, True)
    return out, clip


@pytest.mark.parametrize("layout,extra", [
    ("blur", {}),
    ("crop", {"_track": [(0.0, 0.3), (3.0, 0.3), (3.6, 0.7)]}),
    ("facecam", {"_cam_box_n": (0.77, 0.72, 0.21, 0.2), "_cam_crop_n": (0.79, 0.72, 0.17, 0.2)}),
])
def test_layouts_keep_60fps_bt709_and_pass_qa(media, tmp_path, layout, extra):
    out, clip = render(media, tmp_path, layout, extra)
    info = probe(out)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    a = next(s for s in info["streams"] if s["codec_type"] == "audio")
    assert (v["width"], v["height"], v["r_frame_rate"]) == (1080, 1920, "60/1")
    assert v["color_primaries"] == "bt709" and v["color_transfer"] == "bt709"
    assert a["sample_rate"] == "48000"
    assert abs(float(info["format"]["duration"]) - 8.0) < 0.1
    assert clipper.qa_check(out, 8.0, 60.0, True) == ([], [])
    assert abs(lufs(out) + 14) < 1.0


def test_30fps_source_stays_30fps(media, tmp_path):
    out, _ = render(media, tmp_path, "blur", src="analysis720.mp4")
    v = next(s for s in probe(out)["streams"] if s["codec_type"] == "video")
    assert v["r_frame_rate"] == "30/1"


def test_qa_catches_broken_files(media, tmp_path):
    bad = tmp_path / "bad.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=720x1280:r=24:d=6", "-c:v", "libx264", "-preset", "ultrafast", bad)
    problems, _ = clipper.qa_check(bad, 8.0, 60.0, True)
    text = "; ".join(problems)
    assert "size 720x1280" in text and "length" in text and "fps" in text and "no audio" in text and "black" in text


def test_hq_segment_lines_up_exactly(media, tmp_path):
    wav = tmp_path / "audio.wav"
    clipper.extract_wav(media / "analysis720.mp4", wav)
    # a stream-copied cut starts on the keyframe BEFORE the requested time (here 8.0 s instead of 10.3 s)
    seg = tmp_path / "seg.mkv"
    ffmpeg("-ss", "10.3", "-i", media / "hq1440.mp4", "-t", "12", "-c", "copy", "-avoid_negative_ts", "make_zero", seg)
    at = clipper.align_segment(seg, wav, 10.3, tmp_path)
    assert at is not None and abs(at - 8.0) < 0.01
    # rendering from the HQ segment with that offset gives the same moment as the analysis copy
    out, _ = render(media, tmp_path, "blur", src="hq1440.mp4", clip={"start": 12.0, "end": 18.0, "title": "t", "words": []})
    assert clipper.probe_size(seg) == (2560, 1440)


def test_silent_segment_is_not_guessed(tmp_path, media):
    wav = tmp_path / "audio.wav"
    clipper.extract_wav(media / "analysis720.mp4", wav)
    silent = tmp_path / "silent.mkv"
    ffmpeg("-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-f", "lavfi", "-i", "color=s=320x240:d=20",
           "-t", "20", "-shortest", silent)
    assert clipper.align_segment(silent, wav, 5.0, tmp_path) is None


def test_full_pipeline_on_a_local_file(media, tmp_path):
    """Download-free end-to-end run: scoring, cut snapping, layout choice, render, QA, files, cleanup."""
    cfg = {**clipper.DEFAULTS, **FAST, "output_root": str(tmp_path), "num_clips": 2, "min_clip_seconds": 4,
           "max_clip_seconds": 5, "gemini_api_key": "", "whisper_device": "cpu", "whisper_cpu_model": "none",
           "campaign": "", "section": None, "verbose": False, "_lang": "en", "render_workers": 2}
    folder = clipper.process(str(media / "src60.mp4"), cfg, True, credit="Tester")
    mp4s = sorted(folder.glob("[0-9][0-9]_*.mp4"))
    assert len(mp4s) == 2
    for m in mp4s:
        txt = m.with_suffix(".txt").read_text()
        assert "Quality check: passed" in txt and "from the original file" in txt
    assert not (tmp_path / "_work").exists()
