import random

import pytest

from silent_signal.benchmark import cer, timeline
from silent_signal.morse import Pulse, decode_morse, decode_pulses, encode, pulses_from_states


def test_encode_decode_round_trip():
    assert encode("SOS help") == "... --- ... / .... . .-.. .--."
    assert decode_morse("... --- ... / .... ..")[0] == "SOS HI"


def test_unknown_code_becomes_question_mark():
    text, unknown = decode_morse("...... .-")
    assert text == "?A" and unknown == 1


@pytest.mark.parametrize("unit", [0.1, 0.25, 0.6])
def test_perfect_timing_decodes_at_any_speed(unit):
    pulses = timeline("SOS NEED WATER", unit, jitter=0.0, r=random.Random(0))
    d = decode_pulses(pulses)
    assert d.text == "SOS NEED WATER"
    assert d.unit == pytest.approx(unit, rel=0.01)


def test_single_kind_of_pulse_uses_default_split():
    three_dots = [Pulse(0, 0.2), Pulse(0.4, 0.6), Pulse(0.8, 1.0)]
    assert decode_pulses(three_dots, default_split=0.4).text == "S"
    three_dashes = [Pulse(0, 0.9), Pulse(1.2, 2.1), Pulse(2.4, 3.3)]
    assert decode_pulses(three_dashes, default_split=0.4).text == "O"


def test_states_to_pulses_merges_dropouts_and_drops_flicker():
    t = [i / 10 for i in range(20)]
    s = [False, True, True, False, True, True, False, False, False, False,   # one frame dropout inside a pulse
         True, False, False, False, False, False, False, False, False, False]  # one-frame flicker
    pulses = pulses_from_states(t, s, min_on=0.15, min_off=0.15)
    assert len(pulses) == 1
    assert (pulses[0].start, pulses[0].end) == pytest.approx((0.1, 0.6))


def test_cer():
    assert cer("SOS", "SOS") == 0
    assert cer("SOS", "SOT") == pytest.approx(1 / 3)
