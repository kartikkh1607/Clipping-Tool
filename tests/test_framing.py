"""Face-tracking decisions, using made-up detections (so they don't depend on test photos)."""
import numpy as np
import pytest

import framing
from conftest import ffmpeg, needs_ffmpeg

FPS = framing.TRACK_FPS


def face(cx, cy=0.4, h=0.3):
    return (cx, cy, h * 0.56, h, 0.9)


def test_podcast_camera_switch_is_a_hard_cut_and_ignores_glitches():
    track = []
    for i in range(60):
        t = i / FPS
        faces = [] if i == 17 else [face(0.1, h=0.2)] if i == 40 else [face(0.3)] if t < 6 else [face(0.72)]
        track.append((t, faces))
    plan = framing.plan_tracked_crop(track, 1920, 1080, auto=True)
    assert plan["points"][0] == (0.0, 0.3)
    (ta, xa), (tb, xb) = plan["points"][-2:]
    assert xa == 0.3 and xb == 0.72 and tb - ta < 0.01 and 5.7 <= ta <= 6.0
    assert len(plan["points"]) == 3          # the dropped frame and the false face moved nothing


def test_slow_drift_pans_smoothly():
    track = [(i / FPS, [face(0.4 + 0.2 * i / 50)]) for i in range(50)]
    pts = framing.plan_tracked_crop(track, 1920, 1080, auto=True)["points"]
    assert all(b[0] - a[0] >= 0.5 for a, b in zip(pts[1::2], pts[2::2]))   # moves are pans, not jumps
    assert pts[-1][1] > 0.5


@pytest.mark.parametrize("track", [
    [(i / FPS, [face(0.86, 0.82, 0.12)]) for i in range(50)],   # streamer facecam in a corner
    [(i / FPS, []) for i in range(50)],                           # gameplay, no faces
    [(i / FPS, [face(0.5, h=0.06)]) for i in range(50)],          # tiny face in the distance
])
def test_auto_layout_says_no_for_gameplay(track):
    assert framing.plan_tracked_crop(track, 1920, 1080, auto=True) is None


def test_two_people_who_fit_are_both_framed():
    track = [(i / FPS, [face(0.45, h=0.25), face(0.56, h=0.25)]) for i in range(30)]
    assert framing.plan_tracked_crop(track, 1920, 1080)["points"] == [(0.0, 0.505)]


def test_vertical_source_needs_no_crop():
    assert framing.plan_tracked_crop([(i / FPS, [face(0.5)]) for i in range(30)], 1080, 1920) is None


def test_crop_expression_stays_in_frame():
    expr = framing.crop_x_expr([(0.0, 0.0), (5.0, 0.0), (5.6, 1.0)], 1920, 606)
    assert "lt(t\\,5.000)" in expr and expr.endswith("1314))")    # clamped to 1920 - 606


def test_moved_facecam_is_found():
    seen = [[face(0.86, 0.82, 0.12), face(0.3 + 0.1 * k, 0.5, 0.2)] for k in range(5)]
    cx, cy, fh = framing.locate_facecam(seen)
    assert abs(cx - 0.86) < 0.01 and abs(cy - 0.82) < 0.01
    x, y, w, h = framing.cam_box_from_face(cx, cy, fh, 1920, 1080)
    assert x <= 0.86 * 1920 <= x + w and y <= 0.82 * 1080 <= y + h and x + w <= 1920 and y + h <= 1080
    assert not framing.face_in_box(seen, (100, 100, 300, 200), 1920, 1080)
    assert framing.face_in_box(seen, (x, y, w, h), 1920, 1080)


def test_detector_loads_and_handles_empty_frames():
    det = framing.detector()
    if not det.ok:
        pytest.skip("OpenCV not installed")
    assert det.kind == "yunet"            # the bundled model is used
    assert det.detect(np.zeros((288, 512, 3), np.uint8)) == []


@needs_ffmpeg
def test_cuts_snap_to_the_shot_change(tmp_path):
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=60:duration=6", "-f", "lavfi",
           "-i", "smptebars=size=1920x1080:rate=60:duration=6", "-filter_complex",
           "[0:v]trim=0:3.07,setpts=PTS-STARTPTS[a];[1:v]trim=3.07:6,setpts=PTS-STARTPTS[b];[a][b]concat=n=2:v=1[v]",
           "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast", tmp_path / "cut.mp4")
    pts = framing.snap_cuts_to_shots(tmp_path / "cut.mp4", 0.0, [(0.0, 0.3), (3.1, 0.3), (3.101, 0.7)], 1920, 1080)
    assert abs(pts[-1][0] - 3.067) < 0.02 and pts[-1][1] == 0.7


def test_static_image_is_not_mistaken_for_a_webcam():
    rng = np.random.default_rng(0)
    still = rng.integers(0, 255, (288, 512, 3), dtype=np.uint8)
    frozen = [(t, still) for t in range(5)]
    live = [(t, rng.integers(0, 255, (288, 512, 3), dtype=np.uint8)) for t in range(5)]
    box = (1400, 700, 400, 300)
    assert not framing.region_is_live(frozen, box, 1920, 1080)
    assert framing.region_is_live(live, box, 1920, 1080)
