"""Analyze a recorded video: find the signal, decode the Morse.

Run:  python -m silent_signal.analyze clip.mp4 --channel flash
      python -m silent_signal.analyze clip.mp4 --channel blink --json out.json
"""

import argparse
import json
import sys
from dataclasses import asdict

import cv2

from .detectors import get
from .morse import Pulse, decode_pulses, pulses_from_states


def analyze(path: str, channel: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    det = get(channel)
    times, states, confs, values = [], [], [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        r = det.read(frame, t)
        times.append(t); states.append(r.on); confs.append(r.confidence); values.append(r.value)
        i += 1
    cap.release()
    det.close()

    pulses = pulses_from_states(times, states, confs, min_on=det.min_on, min_off=det.min_off)
    d = decode_pulses(pulses, default_split=det.default_split)
    return {
        "channel": channel, "fps": fps, "frames": i, "duration_s": round(i / fps, 2),
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
    print(f"\nmorse: {out['morse']}\ntext:  {out['text']}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
