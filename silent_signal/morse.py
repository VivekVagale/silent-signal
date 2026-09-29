"""Morse code: turning on/off pulse timings into dots, dashes, letters and words.

Every channel (blinks, finger presses, light flashes) ends up as a list of
Pulses: moments when the signal was ON, with start and end times. This module
does not care where they came from, which is what keeps channels pluggable.

Timing, in standard Morse "units":
    dot = 1 unit on, dash = 3 units on,
    gap inside a letter = 1 unit off, between letters = 3, between words = 7.

People are not metronomes, and their errors are *proportional*: someone aiming
for 0.3 s might produce 0.2-0.45 s, someone aiming for 0.9 s might produce
0.6-1.3 s. So all comparisons happen on a log scale, and thresholds sit at
geometric midpoints (sqrt(1 x 3) = 1.73 units between dot and dash, not 2).
The unit itself is learned from the pulses, so fast and slow senders both work.
"""

import math
from dataclasses import dataclass, field

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.", "G": "--.", "H": "....",
    "I": "..", "J": ".---", "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---", "P": ".--.",
    "Q": "--.-", "R": ".-.", "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..", "0": "-----", "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..", "9": "----.", ".": ".-.-.-", ",": "--..--",
    "?": "..--..", "!": "-.-.--", "/": "-..-.", "@": ".--.-.", "'": ".----.", "-": "-....-",
}
DECODE = {v: k for k, v in MORSE.items()}

DOT_DASH = math.sqrt(3)      # 1.73 units: geometric midpoint of 1 and 3
LETTER = math.sqrt(3)        # gaps: 1 (inside letter) vs 3 (between letters)
WORD = math.sqrt(21)         # 4.58 units: 3 (between letters) vs 7 (between words)


@dataclass
class Pulse:
    """One ON period of a signal, in seconds, with the detector's confidence."""
    start: float
    end: float
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Decoded:
    morse: str                      # e.g. "... --- ... / .... ..", letters split by space, words by " / "
    text: str                       # e.g. "SOS HI"
    marks: list[str] = field(default_factory=list)   # "." or "-" per pulse
    unit: float = 0.0               # estimated dot length in seconds
    unknown: int = 0                # letters that were not valid Morse


def encode(text: str) -> str:
    """'SOS HI' -> '... --- ... / .... ..' (unknown characters are skipped)."""
    words = []
    for word in text.upper().split():
        letters = [MORSE[c] for c in word if c in MORSE]
        if letters:
            words.append(" ".join(letters))
    return " / ".join(words)


def decode_morse(morse: str) -> tuple[str, int]:
    """'... --- ... / .... ..' -> ('SOS HI', 0). Unknown letters become '?'."""
    unknown, words = 0, []
    for word in morse.split("/"):
        letters = []
        for code in word.split():
            letters.append(DECODE.get(code, "?"))
            unknown += code not in DECODE
        if letters:
            words.append("".join(letters))
    return " ".join(words), unknown


def _two_means_log(values: list[float]) -> tuple[float, float]:
    """1-D k-means with k=2 on log(values). Returns the two centres (seconds)."""
    logs = sorted(math.log(v) for v in values)
    c1, c2 = logs[0], logs[-1]
    for _ in range(50):
        lo = [x for x in logs if abs(x - c1) <= abs(x - c2)]
        hi = [x for x in logs if abs(x - c1) > abs(x - c2)]
        n1 = sum(lo) / len(lo) if lo else c1
        n2 = sum(hi) / len(hi) if hi else c2
        if (n1, n2) == (c1, c2):
            break
        c1, c2 = n1, n2
    return math.exp(c1), math.exp(c2)


def estimate_unit(durations: list[float], default_split: float) -> float:
    """Dot length in seconds, learned from ON durations.

    If short and long pulses are both present (longest >= 2 x shortest),
    cluster them into dots and dashes on a log scale; the unit is the
    geometric mean of the dot centre and dash centre / 3, so both kinds of
    pulse inform it. Otherwise all pulses are the same kind, and the default
    split decides which kind.
    """
    lo, hi = min(durations), max(durations)
    if hi >= 2 * lo:
        dot, dash = _two_means_log(durations)
        return math.sqrt(dot * dash / 3)
    typical = math.exp(sum(math.log(d) for d in durations) / len(durations))
    return typical if typical < default_split else typical / 3


def decode_pulses(pulses: list[Pulse], default_split: float = 0.3,
                  unit: float | None = None) -> Decoded:
    """Turn a sequence of ON pulses into Morse and text.

    ON time decides dot vs dash (threshold 1.73 units). OFF time between
    pulses decides structure: under 1.73 units = same letter, under 4.58 =
    next letter, longer = next word. Pass `unit` to fix the speed instead of
    learning it (the live app does this until it has seen enough pulses).
    """
    pulses = sorted(pulses, key=lambda p: p.start)
    if not pulses:
        return Decoded("", "", [], unit or default_split / DOT_DASH)
    u = unit or estimate_unit([p.duration for p in pulses], default_split)
    marks = ["." if p.duration < DOT_DASH * u else "-" for p in pulses]

    parts = [marks[0]]
    for prev, cur, mark in zip(pulses, pulses[1:], marks[1:]):
        gap = cur.start - prev.end
        if gap >= WORD * u:
            parts.append(" / ")
        elif gap >= LETTER * u:
            parts.append(" ")
        parts.append(mark)
    morse = "".join(parts)
    text, unknown = decode_morse(morse)
    return Decoded(morse, text, marks, u, unknown)


def pulses_from_states(times: list[float], states: list[bool], confidences: list[float] | None = None,
                       min_on: float = 0.0, min_off: float = 0.0) -> list[Pulse]:
    """Per-frame ON/OFF states -> Pulses.

    min_on drops flickers shorter than a real signal (detector noise);
    min_off merges two pulses split by a tiny dropout (e.g. one missed frame).
    A pulse's confidence is the mean detector confidence over its frames.
    """
    confidences = confidences or [1.0] * len(times)
    raw, start, confs = [], None, []
    for t, on, c in zip(times, states, confidences):
        if on and start is None:
            start, confs = t, [c]
        elif on:
            confs.append(c)
        elif start is not None:
            raw.append(Pulse(start, t, sum(confs) / len(confs)))
            start = None
    if start is not None and times:
        raw.append(Pulse(start, times[-1], sum(confs) / len(confs)))

    merged: list[Pulse] = []
    for p in raw:
        if merged and p.start - merged[-1].end < min_off:
            last = merged[-1]
            merged[-1] = Pulse(last.start, p.end, (last.confidence + p.confidence) / 2)
        else:
            merged.append(p)
    return [p for p in merged if p.duration >= min_on]
