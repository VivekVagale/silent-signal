"""The demo's sample clips decode to what they say, through the real detectors."""
from pathlib import Path

import pytest

from silent_signal.analyze import analyze

SAMPLES = Path(__file__).resolve().parents[1] / "web" / "samples"


@pytest.mark.parametrize("clip, channel, text", [("blink_sos_help.mp4", "blink", "SOS HELP"),
                                                ("flash_sos_help.mp4", "flash", "SOS HELP"),
                                                ("sign_be_bold.mp4", "sign", "BE BOLD")])
def test_sample_clip_decodes(clip, channel, text):
    assert analyze(str(SAMPLES / clip), channel)["text"] == text
