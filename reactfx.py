"""
Reaction effects for streamer clips (used by clipper.py), adapted from the Reel Pro "Layercut" look.

  * pick_reactions() - the loudest jumps in a clip (screams, "NO WAY"), where the clip cuts to the streamer's
                       face full-screen for a moment, the way pro clippers edit
  * pick_punch_word()- one word that sums the moment up (Gemini's pick, else from the transcript)
  * word_overlay()   - that word, huge, BEHIND the streamer's head during the first face zoom: the person is
                       cut out of each frame with OpenCV Zoo's PP-HumanSeg model, and the word is drawn
                       everywhere except on the person. Rendered as a transparent video that ffmpeg overlays.

Needs OpenCV and Pillow. Without the segmentation model the word is drawn in front of the streamer instead.
"""
from __future__ import annotations

import re
import subprocess
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEG_MODEL = HERE / "models" / "human_segmentation_pphumanseg_2023mar.onnx"   # OpenCV Zoo, Apache 2.0
FONT = HERE / "assets" / "Poppins-Bold.ttf"                                  # SIL Open Font License

_LOCK = threading.Lock()   # one cv2.dnn net shared by the parallel renders

# ----------------------------------------------------------------------------- words
# Captions show what was said, but slurs and hard swears get masked (N****, F***): Shorts with them on
# screen get age-restricted or demonetised, and clipping campaigns reject them. The audio is untouched.
PROFANITY = re.compile(
    r"^(nigg\w*|niga\w*|fuck\w*|motherfuck\w*|mf|shit\w*|bullshit\w*|bitch\w*|cunt\w*|fags?|faggot\w*|"
    r"retard\w*|whores?|sluts?|dicks?|dickhead\w*|puss(y|ies)|asshole\w*|cocks?|cocksucker\w*|bastards?|wtf|stfu|"
    r"chutiy\w*|madarchod\w*|behenchod\w*|bhenchod\w*|bc|mc|bsdk\w*|gaand\w*|lund\w*|randi\w*)$", re.I)

STOP = set("""the a an and or but if then so of to in on at for with as by from that this these those is are
was were be been being it its i im i'm you we they he she them his her our your my me do does did have has
had not no yes just very really can could would should will about into than too also there here what when
which who how why all any some more most bro bruh yeah okay ok like gonna wanna gotta chat guys man dude
got get go going come came look see know think want right well oh ah uh um wait""".split())


def is_profane(word: str) -> bool:
    return bool(PROFANITY.match(re.sub(r"[^\w]", "", word)))


def mask_word(word: str) -> str:
    """'NIGGA!' -> 'N****!' (first letter kept, the rest starred, punctuation kept)."""
    core = re.sub(r"[^\w]", "", word)
    if not core or not PROFANITY.match(core):
        return word
    return word.replace(core, core[0] + "*" * (len(core) - 1), 1)


def pick_punch_word(gemini_word: str, words: list, a: float, b: float) -> str:
    """The word shown behind the streamer: Gemini's, if it gave a clean one, else the strongest content word
    spoken around the reaction (words are [start, end, text] clip-relative)."""
    w = re.sub(r"[^A-Za-z0-9' ]", "", gemini_word or "").strip().upper()
    if w and len(w) <= 14 and len(w.split()) <= 2 and not any(is_profane(p) for p in w.split()):
        return w
    best = ""
    for s, e, txt in words:
        if s > b + 1.0 or e < a - 1.5:
            continue
        c = re.sub(r"[^A-Za-z']", "", txt).strip("'")
        if len(c) >= 4 and c.lower() not in STOP and not is_profane(c) and len(c) > len(best):
            best = c
    return best.upper()[:12]


# ----------------------------------------------------------------------------- reaction moments
def pick_reactions(spike: np.ndarray, loud: np.ndarray, hype: np.ndarray | None, s: float, e: float,
                   n: int = 2, seconds: float = 2.0, min_score: float = 0.35, min_gap: float = 6.0) -> list:
    """[(a, b)] clip-relative windows for face zooms: the strongest sudden jumps in loudness inside the
    clip (per-second scores on the analysis timeline), strongest first, at most n, min_gap apart."""
    dur = e - s
    if n <= 0 or dur < seconds + 1.0 or spike is None or len(spike) == 0:
        return []
    first, last = int(np.ceil(s + 0.3)), int(np.floor(e - seconds - 0.3))
    cands = []
    for i in range(max(0, first), min(len(spike), last + 1)):
        score = float(spike[i]) + (0.3 * float(hype[i]) if hype is not None and i < len(hype) else 0.0)
        if score >= min_score:
            cands.append((score, i))
    out = []
    for score, i in sorted(cands, reverse=True):
        seg = loud[i:i + 3] if loud is not None and len(loud) > i else []
        peak = i + (int(np.argmax(seg)) if len(seg) else 0)        # spike looks 2 s ahead: find the actual peak
        a = min(max(0.0, peak - s - 0.25), dur - seconds - 0.2)
        if a < 0 or any(abs(a - x) < min_gap for x, _ in out):
            continue
        out.append((round(a, 3), round(a + seconds, 3)))
        if len(out) >= n:
            break
    return out


# ----------------------------------------------------------------------------- person matte
class Matte:
    """Soft person mask (0-1) for a BGR image, from PP-HumanSeg (192x192 input, run through cv2.dnn)."""

    def __init__(self):
        self.net = None
        try:
            import cv2
            self.cv2 = cv2
            if SEG_MODEL.exists():
                self.net = cv2.dnn.readNet(str(SEG_MODEL))
        except Exception:
            self.net = None

    @property
    def ok(self) -> bool:
        return self.net is not None

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        cv2 = self.cv2
        h, w = bgr.shape[:2]
        x = cv2.cvtColor(cv2.resize(bgr, (192, 192), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (x.astype(np.float32) / 255.0 - 0.5) / 0.5
        with _LOCK:
            self.net.setInput(cv2.dnn.blobFromImage(x))
            prob = self.net.forward()[0][1]                     # softmax: channel 1 = person
        m = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip((m - 0.3) / 0.4, 0, 1)                    # firmer edge than the raw probability


_MATTE: list = [None]


def matte() -> Matte:
    if _MATTE[0] is None:
        _MATTE[0] = Matte()
    return _MATTE[0]


# ----------------------------------------------------------------------------- the word behind the head
def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype(str(FONT), max(8, int(size)))
    except Exception:
        for name in ("ariblk.ttf", "Arial Black.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
            try:
                return ImageFont.truetype(name, max(8, int(size)))
            except Exception:
                continue
        return ImageFont.load_default()


def _text_rgba(word: str, W: int, H: int, cy: float, cap: float, color: tuple) -> np.ndarray:
    """RGBA float 0-1 layer: the word centred at height cy (pixels), cap height ~cap pixels, max 92% wide,
    with a soft drop shadow so it reads on bright backgrounds."""
    from PIL import Image, ImageDraw, ImageFilter
    size = int(cap / 0.72)
    f = _font(size)
    probe = ImageDraw.Draw(Image.new("L", (8, 8)))
    for _ in range(12):
        x0, y0, x1, y1 = probe.textbbox((0, 0), word, font=f)
        if x1 - x0 <= W * 0.92:
            break
        size = int(size * (W * 0.92) / (x1 - x0))
        f = _font(size)
    x0, y0, x1, y1 = probe.textbbox((0, 0), word, font=f)
    x, y = (W - (x1 - x0)) / 2 - x0, cy - (y1 - y0) / 2 - y0
    alpha = Image.new("L", (W, H), 0)
    ImageDraw.Draw(alpha).text((x, y), word, font=f, fill=255)
    shadow = alpha.filter(ImageFilter.GaussianBlur(max(4, size // 18)))
    shadow = Image.fromarray(np.roll(np.asarray(shadow), (size // 30, size // 40), axis=(0, 1)))
    a = np.asarray(alpha).astype(np.float32) / 255.0
    sa = np.asarray(shadow).astype(np.float32) / 255.0 * 0.55
    out = np.zeros((H, W, 4), np.float32)
    col = np.array(color[:3], np.float32) / 255.0
    out[..., 3] = a + sa * (1 - a)
    out[..., :3] = np.where(out[..., 3:4] > 0, (col * a[..., None]) / np.maximum(out[..., 3:4], 1e-6), 0)
    return out


def _anim(t: float, dur: float, rise: float = 0.18, fall: float = 0.22) -> tuple[float, float]:
    """(scale, alpha): quick pop-in that settles, short fade at the end."""
    if t < rise:
        p = t / rise
        return 0.82 + 0.18 * (1 - (1 - p) ** 3), min(1.0, p * 2)
    if dur - t < fall:
        return 1.0, max(0.0, (dur - t) / fall)
    return 1.0, 1.0


def word_overlay(src: Path, start: float, dur: float, fps_str: str, fps: float, crop: tuple, word: str,
                 out: Path, color: str = "#FFFFFF", W: int = 1080, H: int = 1920) -> dict | None:
    """Transparent video (PNG-in-MOV, W x H) of `word` behind the person, for the source window
    [start, start+dur] framed by crop (x, y, w, h source pixels) exactly like the face-zoom render.
    Returns {"path", "behind": bool (False = no model, word drawn in front)} or None on failure."""
    import cv2
    x, y, w, h = crop
    mw, mh = 288, 512                       # matte resolution (the model itself works at 192x192)
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{dur:.3f}", "-an",
           "-vf", f"fps={fps_str},crop={w}:{h}:{x}:{y},scale={mw}:{mh}:flags=area",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, check=True, timeout=300).stdout
    except Exception:
        return None
    n = len(raw) // (mw * mh * 3)
    if n == 0:
        return None
    frames = np.frombuffer(raw[: n * mw * mh * 3], np.uint8).reshape(n, mh, mw, 3)
    m = matte()
    masks = []
    if m.ok:
        prev = None
        for fr in frames:
            k = m(fr)
            prev = k if prev is None else 0.35 * prev + 0.65 * k     # steady edges, still follows movement
            masks.append(prev)
    # the word sits at the top of the head so the head covers its lower half
    top = 0.30
    if masks:
        tops = []
        for k in masks[:: max(1, n // 12)]:
            centre = k[:, int(mw * 0.25): int(mw * 0.75)]
            rows = np.where(centre.max(axis=1) > 0.5)[0]
            if len(rows):
                tops.append(rows[0] / mh)
        if tops:
            top = float(np.median(tops))
    cy = min(max(top + 0.07, 0.13), 0.40) * H
    hexs = color.lstrip("#")
    rgb = tuple(int(hexs[i:i + 2], 16) for i in (0, 2, 4))
    full = _text_rgba(word, W, H, cy, H * 0.115, rgb)
    # only the rows the word (and its shadow) touch are worked on: at 4K a full float frame is ~130 MB.
    # The pop-in scales about the word's centre and never grows past 1.0, so the band always holds it.
    rows = np.where(full[..., 3].max(axis=1) > 0)[0]
    y0 = int(rows[0]) if len(rows) else 0
    y1 = int(rows[-1]) + 1 if len(rows) else 2
    y0, y1 = max(0, y0 - 2), min(H, y1 + 2)
    base = np.ascontiguousarray(full[y0:y1])
    del full
    bh = y1 - y0
    frame = np.zeros((H, W, 4), np.uint8)     # transparent everywhere except the band
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}",
                            "-r", fps_str, "-i", "-", "-c:v", "png", "-pix_fmt", "rgba", str(out)],
                           stdin=subprocess.PIPE)
    try:
        for i in range(n):
            sc, al = _anim(i / fps, dur)
            layer = base
            if sc < 0.999:   # scale the finished layer about the word's centre
                M = cv2.getRotationMatrix2D((W / 2, cy - y0), 0, sc)
                layer = cv2.warpAffine(base, M, (W, bh), flags=cv2.INTER_LINEAR)
            a = layer[..., 3] * al
            if masks:
                k = cv2.resize(masks[i], (W, H), interpolation=cv2.INTER_LINEAR)[y0:y1]
                a = a * (1 - k)
            frame[y0:y1, :, :3] = (layer[..., :3] * 255).astype(np.uint8)
            frame[y0:y1, :, 3] = (np.clip(a, 0, 1) * 255).astype(np.uint8)
            enc.stdin.write(frame.tobytes())
        enc.stdin.close()
        enc.wait(timeout=300)
    except Exception:
        enc.kill()
        return None
    if enc.returncode or not out.exists():
        return None
    return {"path": out, "behind": bool(masks)}
