"""Train the ASL fingerspelling letter network on hand landmarks.

Data: ASLNow! (sid220/asl-now-fingerspelling on Hugging Face, MIT licence):
2,122 single frames of MediaPipe web hand landmarks from several signers,
one JSON file per frame, one folder per letter. Download it with

    uv run --with huggingface_hub python -c "from huggingface_hub import snapshot_download; snapshot_download('sid220/asl-now-fingerspelling', repo_type='dataset', local_dir='data/asl-now')"

then run   python -m silent_signal.train_signs

Augmentation, applied to training frames only:
  - mirror left/right: the same letter signed with the other hand
  - rotate the hand in the image plane by up to 20 degrees
  - stretch x against y by up to 25% (cameras and phones have different aspect ratios)
  - jitter every point a little (landmark noise)
Evaluation: stratified 5-fold cross-validation. The frames carry no signer id,
so frames of the same signer appear on both sides of a split: the score is an
upper bound for a new person, and is reported as such.
"""

import glob
import json
import math
import os
from pathlib import Path

import numpy as np

from .signs import LETTERS, WEIGHTS, features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "asl-now"


def load():
    X, y = [], []
    for li, L in enumerate(LETTERS):
        for f in sorted(glob.glob(str(DATA / L / "*.json"))):
            pts = np.array([[q["x"], q["y"], q["z"]] for q in json.load(open(f, encoding="utf-8"))], np.float32)
            X.append(pts)
            y.append(li)
    return np.stack(X), np.array(y)


def augment(P: np.ndarray, r: np.random.Generator, copies: int = 8) -> np.ndarray:
    """(n, 21, 3) raw landmarks -> (n * copies, 21, 3) varied copies (copy 0 is the original)."""
    out = [P]
    for _ in range(copies - 1):
        Q = P.copy()
        c = Q.mean(axis=1, keepdims=True)
        Q = Q - c
        flip = r.random(len(Q)) < 0.5
        Q[flip, :, 0] *= -1
        a = np.radians(r.uniform(-20, 20, len(Q)))
        cos, sin = np.cos(a)[:, None], np.sin(a)[:, None]
        x, yy = Q[:, :, 0].copy(), Q[:, :, 1].copy()
        Q[:, :, 0], Q[:, :, 1] = cos * x - sin * yy, sin * x + cos * yy
        Q[:, :, 0] *= np.exp(r.uniform(-0.22, 0.22, len(Q)))[:, None]
        Q += r.normal(0, 0.004, Q.shape)
        out.append(Q + c)
    return np.concatenate(out)


def feats(P: np.ndarray) -> np.ndarray:
    return np.stack([features(p) for p in P])


class MLP:
    """63 -> hidden -> 26, ReLU, softmax cross-entropy, Adam, weight decay."""

    def __init__(self, n_in, n_hidden, n_out, r):
        self.W1 = r.normal(0, math.sqrt(2 / n_in), (n_in, n_hidden)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, np.float32)
        self.W2 = r.normal(0, math.sqrt(2 / n_hidden), (n_hidden, n_out)).astype(np.float32)
        self.b2 = np.zeros(n_out, np.float32)
        self.params = [self.W1, self.b1, self.W2, self.b2]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def forward(self, X):
        self.X = X
        self.h = np.maximum(X @ self.W1 + self.b1, 0)
        z = self.h @ self.W2 + self.b2
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def step(self, X, y, lr=2e-3, wd=1e-4, drop=0.2, r=None):
        n = len(X)
        h_pre = X @ self.W1 + self.b1
        h = np.maximum(h_pre, 0)
        mask = (r.random(h.shape) > drop).astype(np.float32) / (1 - drop)
        hd = h * mask
        z = hd @ self.W2 + self.b2
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        loss = -np.log(p[np.arange(n), y] + 1e-9).mean()
        g = p
        g[np.arange(n), y] -= 1
        g /= n
        gW2 = hd.T @ g + wd * self.W2
        gb2 = g.sum(0)
        gh = (g @ self.W2.T) * mask * (h_pre > 0)
        gW1 = X.T @ gh + wd * self.W1
        gb1 = gh.sum(0)
        self.t += 1
        for i, (prm, grd) in enumerate(zip(self.params, [gW1, gb1, gW2, gb2])):
            self.m[i] = 0.9 * self.m[i] + 0.1 * grd
            self.v[i] = 0.999 * self.v[i] + 0.001 * grd * grd
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            prm -= lr * mh / (np.sqrt(vh) + 1e-8)
        return loss


def train(P, y, r, epochs=60, hidden=64, copies=8):
    Pa = augment(P, r, copies)
    ya = np.tile(y, len(Pa) // len(P))
    X = feats(Pa)
    mean, std = X.mean(0), X.std(0)
    std[std < 1e-3] = 1.0                       # the wrist is always 0: constant features stay 0
    Xn = ((X - mean) / std).astype(np.float32)
    net = MLP(Xn.shape[1], hidden, len(LETTERS), r)
    for ep in range(epochs):
        idx = r.permutation(len(Xn))
        lr = 2e-3 * (0.5 * (1 + math.cos(math.pi * ep / epochs)))   # cosine decay
        for s in range(0, len(idx), 128):
            b = idx[s:s + 128]
            net.step(Xn[b], ya[b], lr=lr, r=r)
    return net, mean, std


def predict(net, mean, std, P):
    return net.forward(((feats(P) - mean) / std).astype(np.float32))


def main() -> None:
    P, y = load()
    print(f"{len(P)} frames, {len(LETTERS)} letters")
    r = np.random.default_rng(0)

    # stratified 5-fold cross-validation
    folds = np.zeros(len(y), int)
    for c in range(len(LETTERS)):
        idx = r.permutation(np.where(y == c)[0])
        folds[idx] = np.arange(len(idx)) % 5
    mirrored = P.copy()
    mirrored[:, :, 0] = 1 - mirrored[:, :, 0]      # the same frames as if signed with the other hand
    pred = np.zeros(len(y), int)
    hits = {"plain": [0, 0], "plain_mirrored": [0, 0], "augmented_mirrored": [0, 0]}
    for k in range(5):
        tr, te = folds != k, folds == k
        net, mean, std = train(P[tr], y[tr], r)
        pred[te] = predict(net, mean, std, P[te]).argmax(1)
        hits["augmented_mirrored"][0] += int(np.sum(predict(net, mean, std, mirrored[te]).argmax(1) == y[te]))
        plain = train(P[tr], y[tr], r, copies=1)     # baseline: no augmentation
        hits["plain"][0] += int(np.sum(predict(*plain, P[te]).argmax(1) == y[te]))
        hits["plain_mirrored"][0] += int(np.sum(predict(*plain, mirrored[te]).argmax(1) == y[te]))
        for h in hits.values():
            h[1] += int(te.sum())
        print(f"fold {k}: accuracy {np.mean(pred[te] == y[te]):.3f}")
    ablation = {k: round(a / n, 4) for k, (a, n) in hits.items()}
    print("no augmentation:", ablation["plain"], " other hand, no augmentation:", ablation["plain_mirrored"],
          " other hand, augmented:", ablation["augmented_mirrored"])
    acc = float(np.mean(pred == y))
    per = {L: float(np.mean(pred[y == i] == i)) for i, L in enumerate(LETTERS)}
    f1s = []
    for i in range(len(LETTERS)):
        tp = np.sum((pred == i) & (y == i)); fp = np.sum((pred == i) & (y != i)); fn = np.sum((pred != i) & (y == i))
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    conf = {}
    for t, p in zip(y, pred):
        if t != p:
            k = f"{LETTERS[t]}->{LETTERS[p]}"
            conf[k] = conf.get(k, 0) + 1
    worst = sorted(conf.items(), key=lambda kv: -kv[1])[:8]
    print(f"cross-validated accuracy {acc:.3f}, macro F1 {np.mean(f1s):.3f}")
    print("most confused:", worst)

    net, mean, std = train(P, y, r)            # final model on all frames
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    rd = lambda a: np.round(a.astype(float), 5).tolist()
    WEIGHTS.write_text(json.dumps({
        "letters": LETTERS, "mean": rd(mean), "std": rd(std),
        "layers": [{"W": rd(net.W1), "b": rd(net.b1)}, {"W": rd(net.W2), "b": rd(net.b2)}],
        "source": "trained on ASLNow! (sid220/asl-now-fingerspelling, MIT) by silent_signal/train_signs.py",
    }), encoding="utf-8")
    out = ROOT / "results" / "signs.json"
    out.write_text(json.dumps({"frames": int(len(P)), "cv_accuracy": round(acc, 4), "cv_macro_f1": round(float(np.mean(f1s)), 4),
                               "cv_accuracy_no_augmentation": ablation["plain"],
                               "other_hand_no_augmentation": ablation["plain_mirrored"],
                               "other_hand_augmented": ablation["augmented_mirrored"],
                               "per_letter": {k: round(v, 3) for k, v in per.items()},
                               "most_confused": worst}, indent=2), encoding="utf-8")
    print(f"saved {WEIGHTS.relative_to(ROOT)} ({os.path.getsize(WEIGHTS) // 1024} KB) and {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
