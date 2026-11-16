"""Real renders from generated footage: quality settings, layouts, loudness, quality check, HQ alignment."""
import json
import subprocess

import pytest

import clipper
from conftest import ffmpeg

FAST = {"x264_preset": "ultrafast", "encoder": "x264"}


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,r_frame_rate,color_primaries,color_transfer,color_space,sample_rate:format=duration",
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
    assert (v.get("color_primaries"), v.get("color_transfer"), v.get("color_space")) == ("bt709", "bt709", "bt709")
    assert a["sample_rate"] == "48000"
    assert abs(float(info["format"]["duration"]) - 8.0) < 0.1
    assert clipper.qa_check(out, 8.0, 60.0, True) == ([], [])
    assert abs(lufs(out) + 14) < 0.5, lufs(out)


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


def test_facecam_face_zoom_and_punch_word(media, tmp_path):
    """Face zooms cut to the cam full-screen for their window; the punch word is overlaid in the first one."""
    extra = {"_cam_box_n": (0.77, 0.72, 0.21, 0.2), "_cam_crop_n": (0.79, 0.72, 0.17, 0.2), "_face_n": None,
             "_react": [(1.0, 3.0), (5.0, 6.5)], "_punch_word": "SPIDER"}
    out, clip = render(media, tmp_path, "facecam", extra)
    v = next(s for s in probe(out)["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"], v["r_frame_rate"]) == (1080, 1920, "60/1")
    assert clipper.qa_check(out, 8.0, 60.0, True) == ([], [])
    assert clip["_punch"].startswith("'SPIDER'")
    assert not list(tmp_path.glob(".*_punch.mov"))          # the temporary word video is cleaned up

    def frame(t):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1",
                              "-vf", "scale=108:192", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                             capture_output=True).stdout
        import numpy as np
        return np.frombuffer(raw, np.uint8).reshape(192, 108).astype(float)
    split, zoom = frame(0.5), frame(2.0)
    band = slice(73, 86)    # the dark band between cam and gameplay in the split layout
    assert split[band].mean() < 40 and zoom[band].mean() > split[band].mean() + 20


@pytest.mark.parametrize("height,size", [(2560, (1440, 2560)), (3840, (2160, 3840))])
@pytest.mark.parametrize("layout,extra", [
    ("blur", {}),
    ("facecam", {"_cam_box_n": (0.77, 0.72, 0.21, 0.2), "_cam_crop_n": (0.79, 0.72, 0.17, 0.2), "_face_n": None,
                 "_react": [(1.0, 2.5)], "_punch_word": "CLUTCH"}),
])
def test_real_1440p_and_4k_renders(media, tmp_path, height, size, layout, extra):
    """1440p / 4K are rendered at that size (every layout piece scaled), keep 60 fps and pass the QA check."""
    clip = {"start": 5.0, "end": 9.0, "banner": "TEST", "title": "t", "words": [[5.5, 5.9, " no"], [6.0, 6.4, " way!"]]}
    out, clip = render(media, tmp_path, layout, {"output_height": height, **extra}, src="hq1440.mp4", clip=clip)
    v = next(s for s in probe(out)["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"], v["r_frame_rate"]) == (*size, "60/1")
    assert clip["_size"] == size
    assert clipper.qa_check(out, 4.0, 60.0, True, size=size) == ([], [])
    assert clipper.qa_check(out, 4.0, 60.0, True)[0] == [f"size {size[0]}x{size[1]}"]   # 1080x1920 expected
    if layout == "facecam":
        assert clip["_punch"].startswith("'CLUTCH'")


def test_layout_geometry_scales_and_stays_even(tmp_path):
    from conftest import ffmpeg as ff
    png = tmp_path / "ov.png"
    ff("-f", "lavfi", "-i", "color=c=red:s=600x100", "-frames:v", "1", png)
    for h, (W, H) in ((1920, (1080, 1920)), (2560, (1440, 2560)), (3840, (2160, 3840))):
        cfg = {**clipper.DEFAULTS, "output_height": h, "overlay_image": str(png)}
        assert clipper.out_size(cfg) == (W, H)
        cam, band, game_y = clipper.fc_geometry(cfg)
        k = h / 1920
        assert all(v % 2 == 0 for v in (cam, band, game_y)) and abs(cam - 720 * k) <= 1 and game_y == cam + band
        for layout in ("blur", "crop", "facecam"):
            w, oh, y = clipper.overlay_geometry(cfg, layout)
            assert w % 2 == 0 and oh % 2 == 0 and 0 <= y and y + oh <= H and w <= W
        assert clipper.overlay_geometry(cfg, "facecam")[2] >= cam            # in the band, below the cam
        chain = clipper.video_chain("facecam", {**cfg, "_cam_crop_n": (0.8, 0.7, 0.17, 0.2),
                                                "_cam_box_n": (0.77, 0.7, 0.21, 0.2)}, 1920, 1080, "60")
        assert f"pad={W}:{H}" in chain and f"overlay=0:{game_y}" in chain


def test_file_size_presets_and_bitrate_scaling():
    base = {**clipper.DEFAULTS, **clipper.FILE_SIZES["balanced"]}
    assert clipper.scaled_bitrate(base) == "20M"
    assert clipper.scaled_bitrate({**base, "output_height": 2560}) == "34M"
    assert clipper.scaled_bitrate({**base, "output_height": 3840}) == "70M"
    assert clipper.scaled_bitrate({**base, **clipper.FILE_SIZES["smaller"], "output_height": 3840}) == "42M"
    args = clipper.encoder_args("x264", {**base, "output_height": 2560}, 60)
    assert args[args.index("-maxrate") + 1] == "34M" and args[args.index("-bufsize") + 1] == "68M"
    small, best = (clipper.estimate_clip_mb({**base, **clipper.FILE_SIZES[k]}) for k in ("smaller", "best"))
    k4 = clipper.estimate_clip_mb({**base, "output_height": 3840})
    assert small[0] < clipper.estimate_clip_mb(base)[0] < best[0] < k4[0] and k4[0] < k4[1]


def test_output_fps_30_from_a_60fps_source(media, tmp_path):
    out, clip = render(media, tmp_path, "blur", {"output_fps": 30})
    v = next(s for s in probe(out)["streams"] if s["codec_type"] == "video")
    assert v["r_frame_rate"] == "30/1" and clip["_fps"] == 30.0
    assert clipper.qa_check(out, 8.0, 30.0, True) == ([], [])
    assert clipper.render_fps({"output_fps": "source", "max_fps": 60}, 59.94) == ("60000/1001", 60000 / 1001)
    assert clipper.render_fps({"output_fps": 60}, 30.0) == ("60", 60.0)


def test_align_segment_corrects_a_late_audio_start(media, tmp_path):
    """A segment whose audio starts 0.25 s after its video: the result is where the VIDEO starts."""
    wav = tmp_path / "audio.wav"
    clipper.extract_wav(media / "analysis720.mp4", wav)
    seg = tmp_path / "late.mkv"
    # video cut at 8.0 s; audio taken from 8.25 s on and stamped to start 0.25 s into the file (stream copy
    # keeps that start time, like a real stream-copied download)
    v, a = tmp_path / "v.mkv", tmp_path / "a.mka"
    ffmpeg("-ss", "8.0", "-i", media / "hq1440.mp4", "-t", "12", "-an", "-c:v", "libx264", "-preset", "ultrafast", v)
    ffmpeg("-ss", "8.25", "-i", media / "hq1440.mp4", "-t", "11.75", "-vn", "-c:a", "flac", a)
    ffmpeg("-i", v, "-itsoffset", "0.25", "-i", a, "-map", "0:v", "-map", "1:a", "-c", "copy", seg)
    st = clipper.stream_start_times(seg)
    assert abs(clipper.av_start_gap(seg) - 0.25) < 0.03, st
    at = clipper.align_segment(seg, wav, 8.0, tmp_path)
    assert at is not None and abs(at - 8.0) < 0.01


def test_reaction_picking_and_punch_word():
    import numpy as np
    import reactfx
    spike = np.zeros(60)
    spike[[12, 14, 30]] = [0.9, 0.7, 0.6]
    loud = np.zeros(60)
    loud[[13, 31]] = 1.0
    r = reactfx.pick_reactions(spike, loud, None, 10.0, 40.0, n=2, seconds=2.0, min_score=0.35, min_gap=6)
    assert r == [(2.75, 4.75), (20.75, 22.75)]                # strongest first; 14 s is too close to 12 s
    assert reactfx.pick_reactions(spike, loud, None, 40.0, 60.0) == []   # nothing loud in this clip
    words = [[2.0, 2.3, " no"], [2.3, 2.9, " spider!"], [3.0, 3.4, " bro"], [3.4, 3.9, " fucking"]]
    assert reactfx.pick_punch_word("", words, 2.75, 4.75) == "SPIDER"
    assert reactfx.pick_punch_word("Clutch", words, 2.75, 4.75) == "CLUTCH"
    assert reactfx.pick_punch_word("shit", words, 2.75, 4.75) == "SPIDER"   # no swearing on screen
