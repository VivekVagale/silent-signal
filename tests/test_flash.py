import random

from silent_signal.analyze import analyze
from silent_signal.benchmark import render, timeline
from silent_signal.detectors import REGISTRY, get


def test_flash_video_decodes_end_to_end(tmp_path):
    r = random.Random(3)
    clip = tmp_path / "sos.mp4"
    render(timeline("SOS", 0.3, 0.0, r), 30, r, clip)
    out = analyze(str(clip), "flash")
    assert out["text"] == "SOS"
    assert [s["mark"] for s in out["signals"]] == list("...---...")


def test_channels_are_registered_by_name():
    assert get("flash").name == "flash"
    assert "flash" in REGISTRY
