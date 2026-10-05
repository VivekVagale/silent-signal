"""Word-level Indian Sign Language: body + both hands over time -> a word.

A word sign is a short movement, so a sign is a sequence of frames. Per frame
(see frame_features) the body and both hands are described relative to the
signer's shoulders, so where they stand and how far from the camera drop out:
  - 7 body points (nose, shoulders, elbows, wrists): 14 numbers,
  - for each hand: where its wrist is on the body (2), its shape (21 points
    relative to its own wrist, scaled by palm length: 42) and whether it was
    seen at all (1): 45 numbers, 90 for both.
Only the frames where a hand is raised count: in INCLUDE, resting wrists hang
1.47-1.70 shoulder-widths below the shoulders, and every sign lifts a wrist to
0.94 or higher, so the line sits at 1.3.
Those frames are resampled to T = 16 and flattened: 16 x 104 inputs to a
small network (same numpy MLP as the letters, see train_words.py).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[1] / "web" / "models" / "isl_words.json"
T = 16                   # frames per sign after resampling
RAISED = 1.3             # a wrist above this many shoulder-widths below the shoulders = signing
MIN_SIGN = 0.3           # seconds: shorter raised stretches are not signs
END_GAP = 0.35           # seconds of lowered hands that end a sign
MIN_PROB = 0.45          # below this the word is shown as unsure


def body_frame(pose: np.ndarray, aspect: float):
    """Shoulder centre and width in aspect-corrected units (x scaled by width / height)."""
    p = pose[:, :2] * (aspect, 1.0)
    origin = (p[1] + p[2]) / 2
    width = max(float(np.hypot(*(p[1] - p[2]))), 1e-3)
    return p, origin, width


def raised(pose: np.ndarray, aspect: float) -> bool:
    if pose is None or np.isnan(pose[0, 0]):
        return False
    p, o, w = body_frame(pose, aspect)
    return bool(min(p[5, 1], p[6, 1]) - o[1] < RAISED * w)


def frame_features(pose: np.ndarray, hands: np.ndarray, aspect: float) -> np.ndarray:
    """One frame -> 104 numbers (all zeros if there is no body)."""
    out = np.zeros(104, np.float32)
    if pose is None or np.isnan(pose[0, 0]):
        return out
    p, o, w = body_frame(pose, aspect)
    out[:14] = ((p - o) / w).reshape(-1)
    for k in range(2):
        h = hands[k]
        if np.isnan(h[0, 0]):
            continue
        q = h[:, :2] * (aspect, 1.0)
        palm = max(float(np.hypot(*(q[9] - q[0]))), 1e-4)
        base = 14 + 45 * k
        out[base:base + 2] = (q[0] - o) / w
        out[base + 2:base + 44] = ((q - q[0]) / palm).reshape(-1)
        out[base + 44] = 1.0
    return out


def active_span(raised_flags) -> tuple[int, int] | None:
    idx = np.flatnonzero(raised_flags)
    return (int(idx[0]), int(idx[-1]) + 1) if len(idx) else None


def resample(F: np.ndarray, n: int = T) -> np.ndarray:
    """(frames, 104) -> (n, 104) by linear interpolation in time."""
    if len(F) == 1:
        return np.repeat(F, n, axis=0)
    x = np.linspace(0, len(F) - 1, n)
    i0 = np.floor(x).astype(int)
    i1 = np.minimum(i0 + 1, len(F) - 1)
    f = (x - i0)[:, None]
    return F[i0] * (1 - f) + F[i1] * f


def clip_features(poses, hands, aspect: float, span=None) -> np.ndarray | None:
    """A recorded sign (per-frame poses and hands) -> the T x 104 input, flattened."""
    if span is None:
        span = active_span([raised(p, aspect) for p in poses])
    if span is None:
        return None
    a, b = span
    F = np.stack([frame_features(poses[i], hands[i], aspect) for i in range(a, b)])
    return resample(F).reshape(-1)


class WordNet:
    def __init__(self, path: Path = WEIGHTS):
        w = json.loads(Path(path).read_text(encoding="utf-8"))
        self.mean, self.std = np.array(w["mean"], np.float32), np.array(w["std"], np.float32)
        self.layers = [(np.array(l["W"], np.float32), np.array(l["b"], np.float32)) for l in w["layers"]]
        self.words = w["words"]

    def probs(self, x: np.ndarray) -> np.ndarray:
        h = (np.atleast_2d(x) - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)
        h = h - h.max(axis=1, keepdims=True)
        e = np.exp(h)
        return e / e.sum(axis=1, keepdims=True)


@dataclass
class Segmenter:
    """Per-frame stream -> finished signs. A sign is a stretch of raised hands that
    ends when the hands have been down for END_GAP seconds."""
    frames: list = field(default_factory=list)     # (t, pose, hands) while a sign is going on
    last_up: float | None = None

    def push(self, t, pose, hands, aspect):
        up = raised(pose, aspect)
        if up:
            self.frames.append((t, pose, hands))
            self.last_up = t
            return None
        if self.frames and t - self.last_up >= END_GAP:
            done, self.frames = self.frames, []
            if done[-1][0] - done[0][0] >= MIN_SIGN:
                return done
        return None
