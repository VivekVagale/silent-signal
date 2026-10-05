"""Sign language: landmark features, the letter network and the hold-to-type rule."""
import json
from pathlib import Path

import numpy as np
import pytest

from silent_signal.signs import LETTERS, LetterNet, Typer, features

ROOT = Path(__file__).resolve().parents[1]


def frames(typer, letter, start, seconds, fps=20, conf=0.9, hand=True):
    for k in range(int(seconds * fps)):
        typer.push(start + k / fps, letter, conf, hand)
    return start + seconds


def test_features_ignore_position_and_size():
    r = np.random.default_rng(0)
    p = r.random((21, 3)).astype(np.float32)
    moved = p * 0.5 + np.array([0.2, 0.1, 0.0], np.float32)
    assert np.allclose(features(p), features(moved), atol=1e-5)


def test_held_letter_types_once():
    t = Typer()
    frames(t, "B", 0.0, 1.5)
    assert t.text == "B"


def test_brief_letter_is_not_typed():
    t = Typer()
    frames(t, "X", 0.0, 0.2)             # passing through a shape on the way to another
    frames(t, "B", 0.2, 1.0)
    assert t.text == "B"


def test_same_letter_twice_needs_a_change_in_between():
    t = Typer()
    end = frames(t, "L", 0.0, 1.0)
    end = frames(t, "L", end, 1.0)        # still holding: no second L
    assert t.text == "L"
    end = frames(t, None, end, 0.3, conf=0.0)
    frames(t, "L", end, 1.0)
    assert t.text == "LL"


def test_a_blurred_frame_does_not_break_a_hold():
    t = Typer()
    end = frames(t, "D", 0.0, 0.25)
    end = frames(t, "D", end, 0.1, conf=0.3)   # two unsure frames
    frames(t, "D", end, 0.25)
    assert t.text == "D"


def test_hand_down_makes_one_space():
    t = Typer()
    end = frames(t, "B", 0.0, 1.0)
    end = frames(t, None, end, 3.0, conf=0.0, hand=False)
    frames(t, "E", end, 1.0)
    assert t.text == "B E"


def test_network_reads_its_own_training_frames():
    net = LetterNet()
    assert net.letters == LETTERS
    data = ROOT / "data" / "asl-now"
    if not data.exists():                 # the dataset is downloaded, not committed
        pytest.skip("ASLNow! landmarks not downloaded")
    hits = total = 0
    for L in LETTERS:
        for f in sorted((data / L).glob("*.json"))[:5]:
            pts = [[q["x"], q["y"], q["z"]] for q in json.loads(f.read_text(encoding="utf-8"))]
            hits += net.predict(pts)[0] == L
            total += 1
    assert hits / total > 0.9


def test_browser_network_matches_python():
    """web/js/detectors.js runs the same weights: same letter, same confidence."""
    import shutil
    import subprocess
    if not shutil.which("node"):
        pytest.skip("node not installed")
    r = np.random.default_rng(1)
    pts = (r.random((21, 3)) * 0.3 + 0.3).tolist()
    js = """
      globalThis.document = { createElement: () => ({ getContext: () => ({}) }) };
      const fs = await import('fs');
      const { signFeatures, letterProbs } = await import('./web/js/detectors.js');
      const net = JSON.parse(fs.readFileSync('web/models/asl_letters.json'));
      const lm = JSON.parse(process.argv[1]).map(([x, y, z]) => ({ x, y, z }));
      const p = letterProbs(net, signFeatures(lm));
      console.log(JSON.stringify(p));
    """
    out = subprocess.run(["node", "--input-type=module", "-e", js, json.dumps(pts)], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    assert np.allclose(json.loads(out), LetterNet().probs(features(pts))[0], atol=1e-4)
