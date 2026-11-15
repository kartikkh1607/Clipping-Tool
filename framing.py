"""
Face-aware framing for vertical clips (used by clipper.py).

  * face_track()        - faces found in a clip, sampled a few times per second
  * plan_tracked_crop() - a 9:16 "virtual camera" that follows the speaker: it holds still while the
                          face stays near the middle, pans when they drift, and cuts when the shot changes
  * crop_x_expr()       - that camera path as an ffmpeg expression for the crop filter
  * locate_facecam()    - finds a streamer's facecam (a small face that stays in one place)

Everything works in normalised 0-1 coordinates, so a plan made on the low-res analysis copy also fits the
high-quality copy that is actually rendered. Needs OpenCV (opencv-python-headless); without it every
function degrades to "no faces found" and the caller falls back to the blur layout.
"""
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")   # keep OpenCV's backend notices out of the clip log

HERE = Path(__file__).resolve().parent
YUNET_MODEL = HERE / "models" / "face_detection_yunet_2023mar.onnx"   # OpenCV Zoo YuNet (MIT licence)

SAMPLE_W = 512        # analysis frames are scaled to this width (faces stay detectable, detection stays fast)
TRACK_FPS = 5         # samples per second while tracking a speaker
MIN_FACE_H = 0.07     # ignore faces smaller than this fraction of the frame height (crowds, posters)
AUTO_FACE_H = 0.11    # "auto" layout: the main face must be at least this big to switch to the tracked crop
AUTO_FACE_SHARE = 0.5  # ...and visible in at least this share of the samples
PAN_SECONDS = 0.6     # how long a pan takes when the speaker drifts
MAX_POINTS = 60       # cap on camera keyframes (keeps the ffmpeg expression small)

_LOCK = threading.Lock()   # OpenCV detectors are not thread-safe; clips render in parallel


class FaceDetector:
    """OpenCV YuNet with the bundled model, or OpenCV's Haar cascade if the model is missing."""

    def __init__(self, min_score: float = 0.75):
        self.kind = None
        try:
            import cv2
        except ImportError:
            return
        self.cv2 = cv2
        if YUNET_MODEL.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.net = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (320, 320), min_score, 0.3, 50)
                self.kind = "yunet"
                return
            except Exception:
                pass
        try:
            self.net = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if not self.net.empty():
                self.kind = "haar"
        except Exception:
            pass

    @property
    def ok(self) -> bool:
        return self.kind is not None

    def detect(self, img: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        """Faces in a BGR image as (cx, cy, w, h, score), all normalised to 0-1 of the image size."""
        h, w = img.shape[:2]
        with _LOCK:
            if self.kind == "yunet":
                self.net.setInputSize((w, h))
                _, faces = self.net.detect(img)
                raw = [] if faces is None else [(f[0], f[1], f[2], f[3], f[14]) for f in faces]
            elif self.kind == "haar":
                gray = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2GRAY)
                side = max(24, h // 25)
                raw = [(x, y, fw, fh, 0.8) for x, y, fw, fh in
                       self.net.detectMultiScale(gray, 1.1, 6, minSize=(side, side))]
            else:
                return []
        return [(float((x + fw / 2) / w), float((y + fh / 2) / h), float(fw / w), float(fh / h), float(s))
                for x, y, fw, fh, s in raw]


_DETECTOR: list = [None]


def detector() -> FaceDetector:
    if _DETECTOR[0] is None:
        _DETECTOR[0] = FaceDetector()
    return _DETECTOR[0]


# ----------------------------------------------------------------------------- frames
def _sample_size(src_w: int, src_h: int) -> tuple[int, int]:
    w = min(SAMPLE_W, src_w) // 2 * 2
    return w, max(2, int(round(src_h * w / src_w / 2)) * 2)


def sample_frames(src: Path, start: float, dur: float, fps: float, src_w: int, src_h: int) -> list:
    """[(t, BGR frame)] for t = 0, 1/fps, ... seconds after `start`, at SAMPLE_W width."""
    w, h = _sample_size(src_w, src_h)
    res = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, start):.2f}", "-i", str(src),
                          "-t", f"{dur:.2f}", "-an", "-vf", f"fps={fps},scale={w}:{h}",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], capture_output=True)
    size = w * h * 3
    n = len(res.stdout) // size
    frames = np.frombuffer(res.stdout[: n * size], dtype=np.uint8).reshape(n, h, w, 3)
    return [(i / fps, frames[i]) for i in range(n)]


def frames_at(src: Path, times: list[float], src_w: int, src_h: int) -> list:
    """[(t, BGR frame)] at the given source times (one quick seek each)."""
    w, h = _sample_size(src_w, src_h)
    out = []
    for t in times:
        res = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t):.2f}", "-i", str(src),
                              "-frames:v", "1", "-an", "-vf", f"scale={w}:{h}", "-f", "rawvideo",
                              "-pix_fmt", "bgr24", "-"], capture_output=True)
        if len(res.stdout) >= w * h * 3:
            out.append((t, np.frombuffer(res.stdout[: w * h * 3], dtype=np.uint8).reshape(h, w, 3)))
    return out


def face_track(src: Path, start: float, dur: float, src_w: int, src_h: int, fps: float = TRACK_FPS) -> list:
    """[(t, faces)] through the clip, t relative to its start. Empty if no detector is available."""
    det = detector()
    if not det.ok:
        return []
    return [(t, det.detect(img)) for t, img in sample_frames(src, start, dur, fps, src_w, src_h)]


# ----------------------------------------------------------------------------- tracked crop
def _target_x(faces: list, crop_frac: float) -> float | None:
    """Where the camera should point for one sample: all main faces if they fit, else the biggest one."""
    faces = [f for f in faces if f[3] >= MIN_FACE_H]
    if not faces:
        return None
    big = max(f[2] * f[3] for f in faces)
    main = [f for f in faces if f[2] * f[3] >= 0.4 * big]
    lo = min(f[0] - f[2] / 2 for f in main)
    hi = max(f[0] + f[2] / 2 for f in main)
    if hi - lo <= crop_frac * 0.85:
        return (lo + hi) / 2
    return max(main, key=lambda f: f[2] * f[3])[0]


def _is_corner_cam(faces_seen: list) -> bool:
    """A small face that sits still in a corner is a streamer's facecam, not a talking head."""
    if not faces_seen:
        return False
    xs = np.array([f[0] for f in faces_seen])
    ys = np.array([f[1] for f in faces_seen])
    hs = np.array([f[3] for f in faces_seen])
    corner = (np.abs(np.median(xs) - 0.5) > 0.25) and (np.abs(np.median(ys) - 0.5) > 0.2)
    still = xs.std() < 0.03 and ys.std() < 0.03
    return bool(corner and still and np.median(hs) < 0.25)


def plan_tracked_crop(track: list, src_w: int, src_h: int, auto: bool = False,
                      out_w: int = 1080, out_h: int = 1920) -> dict | None:
    """Camera path for a 9:16 crop that follows the speaker.

    Returns {"points": [(t, cx), ...], "faces": share of samples with a face} with cx the crop centre
    (0-1 of the source width), or None if there is nothing to follow. With auto=True it also returns None
    unless the clip is clearly a talking-head shot (a big face, most of the time, not a corner facecam).
    """
    if len(track) < 3 or src_w <= 0 or src_h <= 0:
        return None
    crop_frac = (src_h * out_w / out_h) / src_w      # crop width as a fraction of the source width
    if crop_frac >= 0.9:                              # source is already (nearly) vertical
        return None
    times = [t for t, _ in track]
    raw = [_target_x(faces, crop_frac) for _, faces in track]
    hits = sum(x is not None for x in raw)
    share = hits / len(track)
    if hits == 0:
        return None
    if auto:
        mains = [max(f, key=lambda a: a[2] * a[3]) for _, f in track if f]
        heights = [m[3] for m in mains]
        if share < AUTO_FACE_SHARE or not heights or float(np.median(heights)) < AUTO_FACE_H:
            return None
        if _is_corner_cam(mains):
            return None
    # hold the last known position through samples without a face (head turned, blink, motion blur)
    first = next(x for x in raw if x is not None)
    xs, last = [], first
    for x in raw:
        last = x if x is not None else last
        xs.append(last)
    # 3-sample median: one wrong detection never moves the camera
    xs = [float(np.median(xs[max(0, i - 1): i + 2])) for i in range(len(xs))]

    dead = 0.18 * crop_frac          # stay still while the subject is within this distance of the centre
    cut = 0.45 * crop_frac           # jumps bigger than this are shot changes: cut instead of panning
    step = times[1] - times[0] if len(times) > 1 else 1 / TRACK_FPS
    cam = xs[0]
    points = [(0.0, cam)]
    pending = 0
    for i in range(1, len(xs)):
        d = xs[i] - cam
        if abs(d) <= dead:
            pending = 0
            continue
        pending += 1
        if abs(d) > cut:                       # new shot / other speaker: hard cut on the change
            t = max(points[-1][0] + 0.002, times[i] - step / 2)
            points += [(t, cam), (t + 0.001, xs[i])]
            cam, pending = xs[i], 0
        elif pending >= 2:                     # drifted for a moment: smooth pan
            t = max(points[-1][0] + 0.002, times[i - 1])
            points += [(t, cam), (t + PAN_SECONDS, xs[i])]
            cam, pending = xs[i], 0
    points = _cap_points(points)
    return {"points": [(round(t, 3), round(x, 4)) for t, x in points], "faces": round(share, 2)}


def snap_cuts_to_shots(src: Path, start: float, points: list, src_w: int, src_h: int) -> list:
    """Move each hard cut in the camera path onto the exact frame where the shot changes, so the crop
    jumps on the same frame as the video's own edit (no flash of the wrong framing)."""
    out = list(points)
    for k in range(1, len(out)):
        (ta, _), (tb, xb) = out[k - 1], out[k]
        if tb - ta > 0.002:              # only the 1 ms steps are cuts
            continue
        lo = max(0.0, ta - 0.25)
        frames = sample_frames(src, start + lo, 0.5, 60, max(64, src_w // 12), max(36, src_h // 12))
        if len(frames) < 3:
            continue
        diffs = [float(np.abs(frames[i][1].astype(np.int16) - frames[i - 1][1]).mean())
                 for i in range(1, len(frames))]
        i = int(np.argmax(diffs))
        if diffs[i] > 3 * (np.median(diffs) + 1):    # a real shot change, not just movement
            t = lo + frames[i + 1][0]
            if (k < 2 or t > out[k - 2][0] + 0.003) and (k + 1 >= len(out) or t + 0.001 < out[k + 1][0]):
                out[k - 1], out[k] = (round(t - 0.001, 3), out[k - 1][1]), (round(t, 3), xb)
    return out


def _cap_points(points: list) -> list:
    if len(points) <= MAX_POINTS:
        return points
    keep = [points[0]] + points[1:][-(MAX_POINTS - 1):]   # very busy clip: keep the latest moves
    return keep


def crop_x_expr(points: list, src_w: int, crop_w: int) -> str:
    """ffmpeg expression (variable t, clip-relative seconds) for the crop's left edge in pixels."""
    hi = max(0, src_w - crop_w)
    px = [(t, int(round(min(hi, max(0, cx * src_w - crop_w / 2))))) for t, cx in points]
    expr = str(px[-1][1])
    for (ta, xa), (tb, xb) in reversed(list(zip(px, px[1:]))):
        if xa == xb or tb - ta < 0.002:
            seg = str(xb if tb - ta < 0.002 else xa)
        else:
            seg = f"({xa}+({xb - xa})*(t-{ta:.3f})/{tb - ta:.3f})"
        expr = f"if(lt(t\\,{tb:.3f})\\,{seg}\\,{expr})"
    return expr


# ----------------------------------------------------------------------------- facecam
def locate_facecam(frames_faces: list[list], min_share: float = 0.6) -> tuple[float, float, float] | None:
    """(cx, cy, face_h), normalised, of a face that stays in one place in most samples, or None."""
    n = len(frames_faces)
    if n == 0:
        return None
    best = None
    same = lambda f, g: abs(g[0] - f[0]) < 0.04 and abs(g[1] - f[1]) < 0.05 and 0.6 < g[3] / f[3] < 1.6
    for faces in frames_faces:
        for f in faces:
            if not (0.04 <= f[3] <= 0.35):     # a cam face, not a crowd speck or a full-screen face
                continue
            near = [g for fs in frames_faces for g in fs if same(f, g)]
            frames_hit = sum(any(same(f, g) for g in fs) for fs in frames_faces)
            if frames_hit / n >= min_share and (best is None or frames_hit > best[0]):
                cx = float(np.median([g[0] for g in near]))
                cy = float(np.median([g[1] for g in near]))
                fh = float(np.median([g[3] for g in near]))
                best = (frames_hit, cx, cy, fh)
    return None if best is None else best[1:]


def cam_box_from_face(cx: float, cy: float, fh: float, src_w: int, src_h: int,
                      aspect: float = 1.5) -> tuple[int, int, int, int]:
    """A facecam-shaped box (aspect = w/h) around a face, in source pixels, kept inside the frame."""
    h = min(src_h, fh * 2.6 * src_h)
    w = min(src_w, h * aspect)
    h = w / aspect
    x = cx * src_w - w / 2
    y = cy * src_h + 0.15 * fh * src_h - h / 2       # faces sit a little above the middle of a cam
    x = min(max(0.0, x), src_w - w)
    y = min(max(0.0, y), src_h - h)
    even = lambda v: int(v) // 2 * 2
    return even(x), even(y), even(w), even(h)


def region_is_live(frames: list, box: tuple, src_w: int, src_h: int, min_diff: float = 1.5) -> bool:
    """True if the picture inside box (source pixels) changes between samples: a webcam always does,
    a static image (a game HUD portrait, a profile picture) doesn't."""
    if len(frames) < 2:
        return False
    x, y, w, h = box
    crops = []
    for _, img in frames:
        ih, iw = img.shape[:2]
        x0, y0 = int(x * iw / src_w), int(y * ih / src_h)
        x1, y1 = max(x0 + 2, int((x + w) * iw / src_w)), max(y0 + 2, int((y + h) * ih / src_h))
        crops.append(img[y0:y1, x0:x1].astype(np.int16))
    diffs = [float(np.abs(a - b).mean()) for a, b in zip(crops, crops[1:]) if a.shape == b.shape]
    return bool(diffs) and float(np.median(diffs)) >= min_diff


def face_in_box(frames_faces: list[list], box: tuple, src_w: int, src_h: int, min_share: float = 0.4) -> bool:
    """True if a face sits inside box (source pixels) in at least min_share of the samples."""
    if not frames_faces:
        return False
    x, y, w, h = box
    hit = 0
    for faces in frames_faces:
        if any(x <= f[0] * src_w <= x + w and y <= f[1] * src_h <= y + h for f in faces):
            hit += 1
    return hit / len(frames_faces) >= min_share
