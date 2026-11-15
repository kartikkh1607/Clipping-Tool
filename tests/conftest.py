"""Shared test fixtures: generated footage (no downloads needed). Run with:  python -m pytest tests"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg not installed")


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    """Test footage: a 30 s 1080p60 clip with uneven noise audio (unique enough to line up by sound),
    a 1440p60 'original' of the same thing, and a 720p30 'analysis copy' of it."""
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg not installed")
    d = tmp_path_factory.mktemp("media")
    audio = "[1:a]volume='0.2+0.8*abs(sin(t*1.3))*abs(sin(t*0.31))':eval=frame[a]"
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=60:duration=30",
           "-f", "lavfi", "-i", "anoisesrc=d=30:c=pink:r=48000:a=0.3:seed=7", "-filter_complex", audio,
           "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast", "-g", "120",
           "-c:a", "aac", d / "src60.mp4")
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=2560x1440:rate=60:duration=30",
           "-f", "lavfi", "-i", "anoisesrc=d=30:c=pink:r=48000:a=0.3:seed=9", "-filter_complex", audio,
           "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast", "-g", "240",
           "-c:a", "aac", d / "hq1440.mp4")
    ffmpeg("-i", d / "hq1440.mp4", "-vf", "scale=1280:720", "-r", "30", "-c:v", "libx264", "-preset",
           "ultrafast", "-c:a", "aac", d / "analysis720.mp4")
    return d
