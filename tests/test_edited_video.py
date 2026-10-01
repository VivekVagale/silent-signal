"""Edited footage: cuts, cutaways, and the Denton clip as a regression fixture."""

import json
from pathlib import Path

import pytest

from silent_signal.analyze import blind_frames, drop_unseen
from silent_signal.morse import Pulse, decode_pulses

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "denton_pulses.json").read_text(encoding="utf-8"))


def test_long_face_loss_is_blind_short_miss_is_not():
    times = [i / 10 for i in range(40)]
    found = [True] * 10 + [False] * 2 + [True] * 8 + [False] * 10 + [True] * 10
    cuts = [False] * 40
    cuts[35] = True
    blind = blind_frames(times, found, cuts, max_lost=0.25)
    assert not any(blind[10:12])                 # 0.2 s miss: bridged
    assert all(blind[20:30])                     # 1 s gone: unseen
    assert blind[35]                             # a cut


def test_pulse_cut_off_by_an_edit_is_dropped():
    times = [i / 10 for i in range(40)]
    blind = [False] * 40
    blind[20:30] = [True] * 10
    pulses = [Pulse(0.5, 0.8), Pulse(1.5, 2.2), Pulse(3.2, 3.5)]   # middle one runs into the cutaway
    keep, breaks = drop_unseen(pulses, times, blind)
    assert keep == [pulses[0], pulses[2]]
    assert breaks[0] == pytest.approx(2.0)


def test_a_cut_ends_the_word_whatever_the_pause():
    # a dash, a 0.3 s pause, a dot: one letter (N)... unless the pause spans a cut
    pulses = [Pulse(0.0, 0.6), Pulse(0.9, 1.1)]
    assert decode_pulses(pulses, unit=0.2).text == "N"
    assert decode_pulses(pulses, unit=0.2, breaks=[0.7]).text == "T E"


def denton():
    return decode_pulses([Pulse(a, b) for a, b in FIXTURE["pulses"]], default_split=0.4,
                         learn_gaps=True, breaks=FIXTURE["breaks_s"])


def test_denton_what_we_do_read():
    d = denton()
    assert len(d.marks) == 15
    # shot 1, aligned by hand against T-O-R-T-U-R-E: T is the first mark, O the next three, T the eighth
    assert d.marks[0] == "-" and d.marks[1:4] == ["-", "-", "-"] and d.marks[7] == "-"
    assert d.morse.count(" / ") >= 4             # cuts split the reading


@pytest.mark.xfail(strict=True, reason="known limit: Denton's letter pause after T is as short as his pauses "
                   "inside O, and at 15 fps R's dash and U's dots come out the wrong length")
def test_denton_shot_one_reads_tortur():
    assert denton().text.split()[0] == "TORTUR"
