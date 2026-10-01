"""Analyze a recorded video: find the signal, decode the Morse.

Run:  python -m silent_signal.analyze clip.mp4 --channel flash
      python -m silent_signal.analyze clip.mp4 --channel blink --json out.json
"""

import argparse
import json
import sys
from dataclasses import asdict

import cv2
import numpy as np

from .detectors import get
from .morse import Pulse, decode_pulses, pulses_from_states

CUT = 30.0          # mean grey-level change (0-255) between frames that means a new shot
MAX_LOST = 0.25     # seconds without a face before we stop guessing the eye state


def calibrated_states(values: list[float], found: list[bool], sensitivity: float = 0.45) -> list[bool]:
    """Eye open/closed per frame, with thresholds learned from this video.

    Faces and cameras read differently (in an old 320x240 film the open eye
    already scores 0.44 closure), so fixed thresholds fail. "Open" is the 35th
    percentile of closure, "closed" the 99th; the switch-on point sits
    `sensitivity` of the way down from closed, switch-off 0.2 lower (hysteresis).
    Frames with no face keep the previous state.
    """
    seen = sorted(v for v, f in zip(values, found) if f)
    if len(seen) < 10:
        return [False] * len(values)
    q = lambda x: seen[min(len(seen) - 1, int(x * (len(seen) - 1)))]
    open_, closed = q(0.35), q(0.99)
    span = closed - open_
    on_at, off_at = closed - sensitivity * span, closed - (sensitivity + 0.2) * span
    out, on = [], False
    for v, f in zip(values, found):
        if f:
            on = v > off_at if on else v >= on_at
        out.append(on)
    return out


def frame_change(prev: np.ndarray | None, frame_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Small grey thumbnail of the frame, and how much it differs from the previous one."""
    small = cv2.cvtColor(cv2.resize(frame_bgr, (80, 60), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return small, 0.0 if prev is None else float(np.abs(small - prev).mean())


def blind_frames(times: list[float], found: list[bool], cuts: list[bool], max_lost: float = MAX_LOST) -> list[bool]:
    """Frames where we cannot see the signal: the first frame of a new shot, and
    every frame of a stretch longer than `max_lost` with nothing detected.

    News clips are edited: a blink can be cut off by a cutaway, and the pause
    across a cut is not a real pause. A short miss (a frame or two) is ordinary
    detector noise and is bridged instead.
    """
    blind = list(cuts)
    i = 0
    while i < len(found):
        if found[i]:
            i += 1
            continue
        j = i
        while j < len(found) and not found[j]:
            j += 1
        end = times[j] if j < len(times) else times[-1]
        if end - times[i] > max_lost:
            blind[i:j] = [True] * (j - i)
        i = j
    return blind


def drop_unseen(pulses: list[Pulse], times: list[float], blind: list[bool]) -> tuple[list[Pulse], list[float]]:
    """Remove pulses that touch a blind frame (their true length is unknown) and
    return the blind times, which split the message (see decode_pulses breaks)."""
    dt = (times[1] - times[0]) if len(times) > 1 else 0.0
    bt = [t for t, b in zip(times, blind) if b]
    keep = [p for p in pulses if not any(p.start - dt - 1e-6 <= t <= p.end + 1e-6 for t in bt)]
    return keep, bt


def analyze(path: str, channel: str, sensitivity: float = 0.45) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det = get(channel)
    times, states, confs, values, found, cuts = [], [], [], [], [], []
    i, prev = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        r = det.read(frame, t)
        prev, change = frame_change(prev, frame)
        times.append(t); states.append(r.on); confs.append(r.confidence); values.append(r.value); found.append(r.found)
        cuts.append(change > CUT)
        i += 1
    cap.release()
    det.close()

    if channel == "blink":
        states = calibrated_states(values, found, sensitivity)
    pulses = pulses_from_states(times, states, confs, min_on=det.min_on, min_off=det.min_off)
    blind = blind_frames(times, found, cuts)
    pulses, breaks = drop_unseen(pulses, times, blind)
    d = decode_pulses(pulses, default_split=det.default_split, learn_gaps=True, breaks=breaks)
    return {
        "channel": channel, "fps": fps, "frames": i, "duration_s": round(i / fps, 2),
        "cuts_s": [round(t, 2) for k, (t, c) in enumerate(zip(times, cuts)) if c and not (k and cuts[k - 1])],
        "blind_s": round(sum(blind) / fps, 2),
        "text": d.text, "morse": d.morse, "unit_s": round(d.unit, 3), "unknown_letters": d.unknown,
        "signals": [{"start_s": round(p.start, 3), "end_s": round(p.end, 3), "duration_s": round(p.duration, 3),
                     "mark": m, "confidence": round(p.confidence, 3)} for p, m in zip(pulses, d.marks)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--channel", choices=["flash", "blink", "tap"], default="flash")
    ap.add_argument("--json", help="also write the full result here")
    args = ap.parse_args()
    out = analyze(args.video, args.channel)
    for s in out["signals"]:
        print(f"{s['start_s']:8.2f}s  {s['mark']}  {s['duration_s']:.2f}s  conf {s['confidence']:.2f}")
    if out["cuts_s"]:
        print(f"\n{len(out['cuts_s'])} cut(s) at {out['cuts_s']} s, {out['blind_s']} s unseen: "
              "signals there are dropped and a cut always ends the word")
    print(f"\nmorse: {out['morse']}\ntext:  {out['text']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
