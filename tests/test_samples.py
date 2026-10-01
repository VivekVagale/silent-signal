"""The demo's sample clips decode to what they say, through the real detectors."""
from pathlib import Path

import pytest

from silent_signal.analyze import analyze

SAMPLES = Path(__file__).resolve().parents[1] / "web" / "samples"


@pytest.mark.parametrize("clip, channel", [("blink_sos_help.mp4", "blink"), ("flash_sos_help.mp4", "flash")])
def test_sample_clip_decodes(clip, channel):
    assert analyze(str(SAMPLES / clip), channel)["text"] == "SOS HELP"
