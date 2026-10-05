"""Sign language (ASL fingerspelling) from MediaPipe hand landmarks.

MediaPipe HandLandmarker gives 21 points per hand. A letter is a hand shape,
so the shape alone is the input: the points relative to the wrist, scaled by
the hand's size. A small neural network (63 inputs -> 64 -> 26 letters) maps
that to a letter. The same weights run here and in the browser (web/models/).

Typing works like holding a key: a letter held steadily for HOLD seconds is
typed once. To type the same letter twice, change shape or relax the hand in
between. Lowering the hand out of view for SPACE seconds types a space.
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
WEIGHTS = Path(__file__).resolve().parents[1] / "web" / "models" / "asl_letters.json"

HOLD = 0.4          # seconds a letter must be held before it is typed
MIN_CONF = 0.6      # below this the network is unsure: no letter
GRACE = 0.15        # a gap this short (a blurred frame) does not break a hold
SPACE = 1.0         # seconds with no hand that end a word


def features(points) -> np.ndarray:
    """21 (x, y, z) landmarks -> 63 numbers: relative to the wrist, scaled by palm length.

    Position in the frame and distance from the camera drop out; the shape and
    its orientation stay (orientation matters: H and U differ only by it).
    """
    p = np.asarray(points, dtype=np.float32).reshape(21, 3)
    p = p - p[0]
    size = np.linalg.norm(p[9, :2]) + 1e-6          # wrist -> middle-finger knuckle
    return (p / size).reshape(-1)


def landmarks_to_array(lms) -> np.ndarray:
    """MediaPipe landmark objects or dicts -> (21, 3) array."""
    if isinstance(lms[0], dict):
        return np.array([[q["x"], q["y"], q["z"]] for q in lms], dtype=np.float32)
    return np.array([[q.x, q.y, q.z] for q in lms], dtype=np.float32)


class LetterNet:
    """The trained network, forward pass only (numpy)."""

    def __init__(self, path: Path = WEIGHTS):
        w = json.loads(Path(path).read_text(encoding="utf-8"))
        self.mean, self.std = np.array(w["mean"], np.float32), np.array(w["std"], np.float32)
        self.layers = [(np.array(l["W"], np.float32), np.array(l["b"], np.float32)) for l in w["layers"]]
        self.letters = w["letters"]

    def probs(self, feats: np.ndarray) -> np.ndarray:
        h = (np.atleast_2d(feats) - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)
        h = h - h.max(axis=1, keepdims=True)
        e = np.exp(h)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, points) -> tuple[str, float]:
        p = self.probs(features(points))[0]
        i = int(p.argmax())
        return self.letters[i], float(p[i])


@dataclass
class Typer:
    """Per-frame (time, letter or None, confidence) -> typed letters and spaces."""
    hold: float = HOLD
    min_conf: float = MIN_CONF
    grace: float = GRACE
    space: float = SPACE
    current: str | None = None
    since: float = 0.0
    last_seen: float = -math.inf     # last time the current letter was seen
    fired: bool = False
    hand_gone: float | None = None   # when the hand left the view
    typed: list = field(default_factory=list)   # [(time, char, confidence)]

    def push(self, t: float, letter: str | None, conf: float, hand: bool = True) -> str | None:
        out = None
        if not hand:
            if self.hand_gone is None:
                self.hand_gone = t
            if (t - self.hand_gone >= self.space and self.typed and self.typed[-1][1] != " "):
                out = " "
        else:
            self.hand_gone = None
        cand = letter if (hand and letter and conf >= self.min_conf) else None
        if cand == self.current and cand is not None:
            self.last_seen = t
        elif cand is None and self.current is not None and t - self.last_seen <= self.grace:
            pass                                          # a short blur: keep holding
        elif cand != self.current:
            self.current, self.since, self.last_seen, self.fired = cand, t, t, False
        if self.current and cand == self.current and not self.fired and t - self.since >= self.hold:   # type on a sure frame
            self.fired = True
            out = self.current
        if out:
            self.typed.append((t, out, conf))
        return out

    @property
    def text(self) -> str:
        return "".join(c for _, c, _ in self.typed).strip()
