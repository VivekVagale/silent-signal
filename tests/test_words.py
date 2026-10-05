"""Word-level Indian Sign Language: segmenting signs, features, the network."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from silent_signal.train_words import features_of, mirror
from silent_signal.words import RAISED, Segmenter, WordNet, clip_features

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"V:\silent-signal-data\include\landmarks")


def body(wrist_drop, r=None):
    """A standing person (image coordinates, 4:3 frame) with both wrists `wrist_drop` shoulder-widths below the shoulders."""
    sw, sx, sy = 0.2, 0.5, 0.3                      # shoulder width and centre, in x units
    pose = np.array([[sx, sy - 0.15, 1], [sx + sw / 2, sy, 1], [sx - sw / 2, sy, 1],
                     [sx + 0.12, sy + 0.12, 1], [sx - 0.12, sy + 0.12, 1],
                     [sx + 0.1, sy + wrist_drop * sw, 1], [sx - 0.1, sy + wrist_drop * sw, 1]], np.float32)
    pose[:, 0] /= 4 / 3                              # back to image x (the frame is 4:3)
    hands = np.full((2, 21, 3), np.nan, np.float32)
    if r is not None:
        for k in range(2):
            hands[k] = np.c_[pose[5 + k, :2] + r.normal(0, 0.02, (21, 2)), np.zeros(21)]
    return pose, hands


def test_segmenter_splits_signs_at_lowered_hands():
    seg, out, t = Segmenter(), [], 0.0
    for drop, secs in [(1.6, 0.5), (0.3, 1.0), (1.6, 0.6), (0.2, 0.8), (1.6, 0.6), (0.2, 0.1), (1.6, 0.6)]:
        for _ in range(int(secs * 25)):
            done = seg.push(t, *body(drop), 4 / 3)
            if done:
                out.append(len(done))
            t += 1 / 25
    assert len(out) == 2                              # two signs; the 0.1 s twitch is not one
    assert all(18 <= n <= 26 for n in out)


def test_rest_and_signing_sit_either_side_of_the_line():
    from silent_signal.words import raised
    assert not raised(body(1.6)[0], 4 / 3) and raised(body(0.5)[0], 4 / 3)
    assert 0.94 < RAISED < 1.47                       # measured: highest sign 0.94, lowest rest 1.47


def test_mirroring_twice_changes_nothing():
    r = np.random.default_rng(0)
    P = np.stack([body(0.5, r)[0] for _ in range(5)]); H = np.stack([body(0.5, r)[1] for _ in range(5)])
    P2, H2 = mirror(*mirror(P, H))
    assert np.allclose(P, P2) and np.allclose(H, H2, equal_nan=True)


def test_held_out_clips():
    if not DATA.exists():                             # the dataset is downloaded, not committed
        pytest.skip("INCLUDE landmarks not downloaded")
    net, r = WordNet(), json.loads((ROOT / "results" / "words.json").read_text(encoding="utf-8"))
    hits = n = 0
    for w, stems in r["held_out_clips"].items():
        for s in stems:
            d = np.load(DATA / w / f"{s}.npz")
            hits += net.words[int(net.probs(features_of(d["pose"], d["hands"]))[0].argmax())] == w
            n += 1
    assert hits / n > 0.9


def test_browser_matches_python():
    """web/js/detectors.js computes the same input and the same word probabilities."""
    if not shutil.which("node"):
        pytest.skip("node not installed")
    r = np.random.default_rng(3)
    frames = [body(d, r) for d in np.linspace(1.0, 0.2, 12)]
    js_frames = [{"pose": f[0][:, :2].tolist(), "hands": [h[:, :2].tolist() for h in f[1]], "aspect": 4 / 3} for f in frames]
    js = """
      globalThis.document = { createElement: () => ({ getContext: () => ({}) }) };
      const fs = await import('fs');
      const { signInput, letterProbs } = await import('./web/js/detectors.js');
      const net = JSON.parse(fs.readFileSync('web/models/isl_words.json'));
      console.log(JSON.stringify(letterProbs(net, signInput(JSON.parse(process.argv[1])))));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", js, json.dumps(js_frames)], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    x = clip_features([f[0] for f in frames], [f[1] for f in frames], 4 / 3, (0, len(frames)))
    assert np.allclose(json.loads(out), WordNet().probs(x)[0], atol=1e-4)
