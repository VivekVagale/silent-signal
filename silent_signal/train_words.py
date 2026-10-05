"""Train the word-level ISL network on INCLUDE landmarks (see extract_words.py first).

    python -m silent_signal.train_words V:/silent-signal-data/include

Split: for each word, 5 clips are held out (fixed seed) and never trained on;
the demo's sample clip is cut from those. INCLUDE's official split file is no
longer downloadable, and the clips carry no signer id, so the same people may
appear in training and test: the test score is an upper bound for a stranger.

Augmentation (training clips only), on the landmark sequences:
  - start and end of the sign moved by up to 15% (people start and stop differently),
  - the whole body rotated up to 8 degrees, scaled 10%, stretched sideways 10%,
  - 10% of frames lose one hand (the hand model misses frames),
  - optionally mirrored, as if signed by a left-handed signer (compared below).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

from .train_signs import MLP
from .words import T, WEIGHTS, active_span, clip_features, raised

ASPECT = 16 / 9            # INCLUDE is 1920 x 1080
ROOT = Path(__file__).resolve().parents[1]


def load(root: Path):
    clips = []
    for f in sorted((root / "landmarks").glob("*/*.npz")):
        d = np.load(f)
        clips.append((f.parent.name, f.stem, d["pose"], d["hands"]))
    words = sorted({c[0] for c in clips})
    return clips, words


def mirror(poses, hands):
    p, h = poses.copy(), hands.copy()
    p[:, :, 0] = 1 - p[:, :, 0]
    h[:, :, :, 0] = 1 - h[:, :, :, 0]
    p[:, [1, 2, 3, 4, 5, 6]] = p[:, [2, 1, 4, 3, 6, 5]]
    return p, h[:, ::-1]


def jitter(poses, hands, r, mirrored):
    p, h = (mirror(poses, hands) if mirrored else (poses.copy(), hands.copy()))
    a = math.radians(r.uniform(-8, 8))
    s = r.uniform(0.9, 1.1)
    st = math.exp(r.uniform(-0.1, 0.1))
    c = np.nanmean(p[:, 1:3, :2], axis=(0, 1))
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], np.float32) * s
    p[:, :, :2] = (p[:, :, :2] - c) @ R.T * (st, 1) + c
    h[:, :, :, :2] = (h[:, :, :, :2] - c) @ R.T * (st, 1) + c
    drop = r.random(len(h)) < 0.1
    side = r.integers(0, 2, len(h))
    h[drop, side[drop]] = np.nan
    return p, h


def features_of(poses, hands, r=None, aug=False, mirrored=False):
    if aug:
        poses, hands = jitter(poses, hands, r, mirrored)
    span = active_span([raised(p, ASPECT) for p in poses])
    if span is None:
        return None
    if aug:
        a, b = span
        n = b - a
        a = int(np.clip(a + r.integers(-int(0.15 * n), int(0.15 * n) + 1), 0, len(poses) - 2))
        b = int(np.clip(b + r.integers(-int(0.15 * n), int(0.15 * n) + 1), a + 2, len(poses)))
        span = (a, b)
    return clip_features(poses, hands, ASPECT, span)


def build(clips, idx, words, r, copies, mirror_p):
    X, y = [], []
    for i in idx:
        w, _, P, H = clips[i]
        for k in range(copies):
            f = features_of(P, H, r, aug=k > 0, mirrored=k > 0 and r.random() < mirror_p)
            if f is not None:
                X.append(f); y.append(words.index(w))
    return np.array(X, np.float32), np.array(y)


def fit(X, y, n_out, r, epochs=120, hidden=128):
    mean, std = X.mean(0), X.std(0)
    std[std < 1e-3] = 1.0
    Xn = ((X - mean) / std).astype(np.float32)
    net = MLP(Xn.shape[1], hidden, n_out, r)
    for ep in range(epochs):
        lr = 1e-3 * 0.5 * (1 + math.cos(math.pi * ep / epochs))
        idx = r.permutation(len(Xn))
        for s in range(0, len(idx), 64):
            b = idx[s:s + 64]
            net.step(Xn[b], y[b], lr=lr, wd=1e-3, drop=0.4, r=r)
    return net, mean, std


def evaluate(net, mean, std, X, y):
    p = net.forward(((X - mean) / std).astype(np.float32))
    top3 = np.argsort(-p, axis=1)[:, :3]
    return float(np.mean(top3[:, 0] == y)), float(np.mean([y[i] in top3[i] for i in range(len(y))])), top3[:, 0]


def main():
    root = Path(sys.argv[1])
    clips, words = load(root)
    r = np.random.default_rng(0)
    test, train = [], []
    for w in words:
        idx = [i for i, c in enumerate(clips) if c[0] == w]
        idx = list(r.permutation(idx))
        test += idx[:5]; train += idx[5:]
    print(f"{len(words)} words, {len(train)} training clips, {len(test)} held-out clips")
    Xte, yte = build(clips, test, words, r, 1, 0)

    results = {}
    for name, copies, mp in [("no augmentation", 1, 0.0), ("augmentation", 20, 0.0), ("augmentation + mirror", 20, 0.3)]:
        Xtr, ytr = build(clips, train, words, r, copies, mp)
        net, mean, std = fit(Xtr, ytr, len(words), r)
        top1, top3, pred = evaluate(net, mean, std, Xte, yte)
        Xm = np.array([f for f in (features_of(*mirror(clips[i][2], clips[i][3])) for i in test)], np.float32)
        m1, _, _ = evaluate(net, mean, std, Xm, yte)
        results[name] = {"top1": round(top1, 4), "top3": round(top3, 4), "left_handed_top1": round(m1, 4)}
        print(f"{name:24s} top-1 {top1:.3f}  top-3 {top3:.3f}  mirrored (left-handed) top-1 {m1:.3f}")
        if name == "augmentation + mirror":
            final = (net, mean, std, pred)

    net, mean, std, pred = final
    confused = {}
    for t, p in zip(yte, pred):
        if t != p:
            k = f"{words[t]} -> {words[p]}"
            confused[k] = confused.get(k, 0) + 1
    worst = sorted(confused.items(), key=lambda kv: -kv[1])
    print("confused:", worst)

    rd = lambda a: np.round(a.astype(float), 4).tolist()      # 4 decimals: half the download, same answers
    WEIGHTS.write_text(json.dumps({
        "words": words, "T": T, "mean": rd(mean), "std": rd(std),
        "layers": [{"W": rd(net.W1), "b": rd(net.b1)}, {"W": rd(net.W2), "b": rd(net.b2)}],
        "source": "trained on INCLUDE (Sridhar et al. 2020, CC BY 4.0), Greetings + Pronouns, by silent_signal/train_words.py",
    }, separators=(",", ":")), encoding="utf-8")
    held = {w: [clips[i][1] for i in test if clips[i][0] == w] for w in words}
    (ROOT / "results" / "words.json").write_text(json.dumps({
        "dataset": "INCLUDE (ISL), categories Greetings + Pronouns", "words": words,
        "train_clips": len(train), "test_clips": len(test), "results": results, "confused": worst,
        "held_out_clips": held}, indent=2), encoding="utf-8")
    print(f"saved {WEIGHTS.relative_to(ROOT)} and results/words.json")


if __name__ == "__main__":
    main()
