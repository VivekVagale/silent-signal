"""Measure how well messages survive the trip: text -> timing -> (video) -> text.

Run:  python -m silent_signal.benchmark

Two synthetic tests, both with known ground truth:

1. Timing: 300 random messages sent at different speeds with human-like
   timing wobble (each on/off duration multiplied by random noise). Compares
   the adaptive dot/dash split against a fixed threshold.
2. Video: 40 rendered clips of a small torch flashing Morse in a noisy,
   slowly brightening scene at 15 and 30 fps, run through the real
   FlashDetector frame by frame.

Metric: character error rate (CER) = edit distance / message length.
Blink and finger-press detection need real people on camera; they are not
simulated here (see README).
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np

from .detectors import FlashDetector
from .morse import MORSE, Pulse, decode_pulses, pulses_from_states

ROOT = Path(__file__).resolve().parent.parent
WORDS = ("SOS HELP NEED WATER DOCTOR YES NO COME HERE SAFE DANGER CALL MOM OK STOP WAIT GO "
         "FIRE EXIT NORTH SOUTH EAST WEST HOSPITAL FOOD PAIN").split()


def cer(ref: str, hyp: str) -> float:
    """Levenshtein distance / len(ref), on characters."""
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / max(1, len(ref))


def message(r: random.Random) -> str:
    return " ".join(r.choice(WORDS) for _ in range(r.randint(1, 3)))


def timeline(text: str, unit: float, jitter: float, r: random.Random, pause: float = 1.0) -> list[Pulse]:
    """Ideal Morse timing for text, each duration scaled by lognormal noise.

    pause > 1 stretches every gap, like a person blinking on purpose who rests
    much longer between blinks than Morse's 1 / 3 / 7 units.
    """
    wobble = lambda d: d * float(np.exp(r.gauss(0, jitter)))
    gap = lambda d: pause * wobble(d)
    t, pulses = 0.5, []
    for wi, word in enumerate(text.split()):
        if wi:
            t += gap(7 * unit)          # gap between words
        for li, ch in enumerate(word):
            if li:
                t += gap(3 * unit)
            for si, sym in enumerate(MORSE[ch]):
                if si:
                    t += gap(unit)
                d = wobble(unit if sym == "." else 3 * unit)
                pulses.append(Pulse(t, t + d))
                t += d
    return pulses


def timing_test(r: random.Random) -> dict:
    out = {}
    for jitter in (0.0, 0.15, 0.3, 0.45):
        adaptive, fixed = [], []
        for _ in range(300):
            text = message(r)
            unit = r.uniform(0.12, 0.45)          # fast tapper ... slow deliberate blinker
            p = timeline(text, unit, jitter, r)
            adaptive.append(cer(text, decode_pulses(p, default_split=0.3).text))
            fixed.append(cer(text, _fixed_decode(p)))
        out[f"jitter_{jitter}"] = {"adaptive_cer": round(float(np.mean(adaptive)), 4),
                                   "fixed_cer": round(float(np.mean(fixed)), 4)}
        print(f"timing  jitter {jitter:.2f}:  adaptive CER {np.mean(adaptive):.3f}   fixed-threshold CER {np.mean(fixed):.3f}")
    return out


def pause_test(r: random.Random) -> dict:
    """Deliberate signallers: gaps stretched 2.5x. Standard gap thresholds vs gaps learned from the data."""
    std, learned = [], []
    for _ in range(300):
        text = message(r)
        p = timeline(text, r.uniform(0.15, 0.4), 0.15, r, pause=2.5)
        std.append(cer(text, decode_pulses(p, default_split=0.3).text))
        learned.append(cer(text, decode_pulses(p, default_split=0.3, learn_gaps=True).text))
    print(f"pauses 2.5x, jitter 0.15:  standard gaps CER {np.mean(std):.3f}   learned gaps CER {np.mean(learned):.3f}")
    return {"standard_gaps_cer": round(float(np.mean(std)), 4), "learned_gaps_cer": round(float(np.mean(learned)), 4)}


def _fixed_decode(pulses: list[Pulse]) -> str:
    """Baseline: the same thresholds, but for one fixed speed (0.25 s unit) instead of learning the sender's."""
    return decode_pulses(pulses, unit=0.25).text


def render(pulses: list[Pulse], fps: int, r: random.Random, path: Path) -> None:
    """A dim room with a slowly rising light level, sensor noise, and a small torch."""
    end = pulses[-1].end + 1.0
    w, h = 320, 240
    cx, cy = r.randint(40, w - 40), r.randint(40, h - 40)
    scene = np.clip(np.random.default_rng(r.randint(0, 10**6)).normal(60, 25, (h, w, 3)), 0, 255).astype(np.uint8)
    scene = cv2.GaussianBlur(scene, (0, 0), 6)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(r.randint(0, 10**6))
    for i in range(int(end * fps)):
        t = i / fps
        frame = scene.astype(np.float32) * (1 + 0.3 * t / end)            # room light drifts up 30%
        if any(p.start <= t < p.end for p in pulses):
            cv2.circle(frame, (cx, cy), 9, (255, 255, 240), -1)
            frame = cv2.GaussianBlur(frame, (0, 0), 2)
        frame += rng.normal(0, 6, frame.shape)                              # sensor noise
        vw.write(np.clip(frame, 0, 255).astype(np.uint8))
    vw.release()


def video_test(r: random.Random, tmp: Path) -> dict:
    tmp.mkdir(parents=True, exist_ok=True)
    out = {}
    for fps in (15, 30):
        errs, exact = [], 0
        for k in range(20):
            text = message(r)
            unit = r.uniform(0.2, 0.4)
            pulses = timeline(text, unit, 0.15, r)
            path = tmp / f"clip_{fps}_{k}.mp4"
            render(pulses, fps, r, path)

            cap, det = cv2.VideoCapture(str(path)), FlashDetector()
            times, states, confs, i = [], [], [], 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rd = det.read(frame, i / fps)
                times.append(i / fps); states.append(rd.on); confs.append(rd.confidence); i += 1
            cap.release()
            got = decode_pulses(pulses_from_states(times, states, confs, det.min_on, det.min_off),
                                default_split=det.default_split).text
            errs.append(cer(text, got))
            exact += got == text
        out[f"{fps}fps"] = {"cer": round(float(np.mean(errs)), 4), "exact": f"{exact}/20"}
        print(f"video   {fps} fps:  CER {np.mean(errs):.3f}   exact messages {exact}/20")
    return out


def main() -> None:
    r = random.Random(42)
    # each new test gets its own seed, so adding one does not change the clips the others draw
    results = {"timing": timing_test(r), "slow_pauses": pause_test(random.Random(7)), "video_flash": video_test(r, ROOT / "results" / "tmp_clips")}
    for f in (ROOT / "results" / "tmp_clips").glob("*.mp4"):
        f.unlink()
    (ROOT / "results" / "tmp_clips").rmdir()
    (ROOT / "results" / "benchmark.json").write_text(json.dumps(results, indent=2))
    print("\nsaved results/benchmark.json")


if __name__ == "__main__":
    main()
