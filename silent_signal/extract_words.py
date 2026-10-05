"""Turn INCLUDE videos into per-frame body + hand landmarks (one .npz per clip).

INCLUDE (Sridhar et al., ACM Multimedia 2020; CC BY 4.0) is Indian Sign Language:
263 word signs by Deaf students of St. Louis School for the Deaf, Chennai,
https://zenodo.org/records/4010759. This prototype uses two categories,
Greetings and Pronouns (17 words). Unzip them under INCLUDE_DIR/videos, then

    python -m silent_signal.extract_words V:/silent-signal-data/include

The videos are large (about 5 GB for the two categories); only the landmarks
(a few MB) are needed afterwards.
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

_H = None


def _init():
    global _H
    from .holistic import Holistic
    _H = Holistic()


def _one(job):
    src, dst = job
    if dst.exists():
        return dst, "skip"
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    poses, hands, i = [], [], 0
    base = _H.ts / 1000 + 1                                     # timestamps keep rising across clips
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        p, hd = _H.read(frame, base + i / fps)
        poses.append(p if p is not None else np.full((7, 3), np.nan, np.float32))
        hands.append(hd)
        i += 1
    cap.release()
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, pose=np.stack(poses), hands=np.stack(hands), fps=fps)
    return dst, i


def main():
    root = Path(sys.argv[1])
    jobs = []
    for v in sorted((root / "videos").glob("*/*/*")):
        if v.suffix.lower() in (".mov", ".mp4"):
            word = v.parent.name.split(". ", 1)[-1]
            jobs.append((v, root / "landmarks" / word / (v.stem + ".npz")))
    print(f"{len(jobs)} clips")
    with Pool(6, initializer=_init) as pool:
        for k, (dst, n) in enumerate(pool.imap_unordered(_one, jobs), 1):
            if k % 25 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)}  {dst.parent.name}/{dst.name}: {n}", flush=True)


if __name__ == "__main__":
    main()
