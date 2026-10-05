"""Cut the ISL sample clip from three held-out INCLUDE clips of one signer.

    python tools/isl_sample/make_sample.py V:/silent-signal-data/include web/samples/isl_hello_how_are_you_thank_you.mp4

Each clip is cropped around the upper body (4.2 shoulder-widths wide, 4:3, from
the landmarks), scaled to 640x480, credited on screen, and the three are joined.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

CLIPS = [("Greetings", "48. Hello", "Hello", "MVI_0029"),
         ("Greetings", "49. How are you", "How are you", "MVI_0034"),
         ("Greetings", "55. Thank you", "Thank you", "MVI_0061")]
CREDIT = "INCLUDE ISL dataset (Sridhar et al. 2020), CC BY 4.0 - held-out clips, never used in training"


def main(root: Path, out: Path):
    tmp = Path(tempfile.mkdtemp())
    parts = []
    for cat, folder, word, stem in CLIPS:
        d = np.load(root / "landmarks" / word / f"{stem}.npz")
        sh = np.nanmean(d["pose"][:, 1:3, :2], axis=(0, 1)) * (1920, 1080)
        sw = np.nanmean(np.hypot(*((d["pose"][:, 1, :2] - d["pose"][:, 2, :2]) * (1920, 1080)).T))
        cw = int(4.2 * sw); ch = int(cw * 3 / 4)
        x0 = int(np.clip(sh[0] - cw / 2, 0, 1920 - cw)); y0 = int(np.clip(sh[1] - 1.5 * sw, 0, 1080 - ch))
        cap = cv2.VideoCapture(str(root / "videos" / cat / folder / f"{stem}.MOV"))
        part = tmp / f"{stem}.avi"
        vw = cv2.VideoWriter(str(part), cv2.VideoWriter_fourcc(*"MJPG"), 25, (640, 480))
        while True:
            ok, f = cap.read()
            if not ok:
                break
            c = cv2.resize(f[y0:y0 + ch, x0:x0 + cw], (640, 480), interpolation=cv2.INTER_AREA)
            cv2.rectangle(c, (6, 454), (634, 474), (20, 20, 20), -1)
            cv2.putText(c, CREDIT, (12, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 220, 255), 1, cv2.LINE_AA)
            vw.write(c)
        vw.release()
        parts.append(part)
    (tmp / "list.txt").write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(tmp / "list.txt"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "25",
                    "-movflags", "+faststart", str(out)], check=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
