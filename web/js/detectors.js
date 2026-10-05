// Signal channels. Each one turns a video frame into {on, confidence, value}.
// Same contract as silent_signal/detectors in Python: to add a channel, write
// one object with init() and read(), add it to CHANNELS. Nothing else changes.

const MP = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1";
const MODELS = "https://storage.googleapis.com/mediapipe-models";
let vision = null, fileset = null;

async function mp() {
  if (!vision) {
    vision = await import(`${MP}/vision_bundle.mjs`);
    fileset = await vision.FilesetResolver.forVisionTasks(`${MP}/wasm`);
  }
  return vision;
}

// GPU where the browser supports it, otherwise the slower CPU path
async function create(Task, options) {
  try {
    return await Task.createFromOptions(fileset, { ...options, baseOptions: { ...options.baseOptions, delegate: "GPU" } });
  } catch {
    return await Task.createFromOptions(fileset, { ...options, baseOptions: { ...options.baseOptions, delegate: "CPU" } });
  }
}

// MediaPipe's VIDEO mode needs every timestamp to be later than the last one it saw.
// A second video starts again at 0 s, which used to make it throw and silently
// stop the analysis; this keeps timestamps increasing across videos and camera runs.
function nextTs(task, tMs) {
  task._ts = Math.max(Math.round(tMs), (task._ts ?? -1) + 1);
  return task._ts;
}

// A face that fills the frame (an extreme close-up of the eyes) is often missed by
// the face detector, which expects some head and background around it. Shrinking the
// frame into a black border fixed this on a test clip: faces found in 100% of frames
// instead of 50%. Returns the padded canvas and how to map points back.
const padCanvas = document.createElement("canvas"), padCtx = padCanvas.getContext("2d");
function padded(src, frac = 0.5) {
  const w = src.videoWidth || src.width, h = src.videoHeight || src.height;
  const W = Math.round(w * (1 + 2 * frac)), H = Math.round(h * (1 + 2 * frac));
  if (padCanvas.width !== W || padCanvas.height !== H) { padCanvas.width = W; padCanvas.height = H; }
  padCtx.fillStyle = "#000"; padCtx.fillRect(0, 0, W, H);
  padCtx.drawImage(src, Math.round(w * frac), Math.round(h * frac), w, h);
  return { canvas: padCanvas, unmap: (p) => ({ x: (p.x * W - w * frac) / w, y: (p.y * H - h * frac) / h }) };
}

// hysteresis: separate switch-on and switch-off levels stop the signal chattering near the threshold
function hysteresis(state, value, onAt, offAt, higherIsOn = true) {
  if (higherIsOn) return state ? value > offAt : value >= onAt;
  return state ? value < offAt : value <= onAt;
}

export const flash = {
  id: "flash", label: "Light flash", hint: "Flash a torch or phone light at the camera. Short flash = dot, long = dash.",
  defaultSplit: 0.35, minOn: 0.05, minOff: 0.05, valueLabel: "brightness", range: [0, 255],
  async init() {
    this.canvas = new OffscreenCanvas(160, 120);
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.reset();
  },
  reset() { this.dark = null; this.bright = null; this.on = false; },
  // brightest spot after a light blur, with dark/bright levels that track the room
  read(src) {
    const ctx = this.ctx;
    ctx.filter = "blur(1px)";
    ctx.drawImage(src, 0, 0, 160, 120);
    const d = ctx.getImageData(0, 0, 160, 120).data;
    let v = 0;
    for (let i = 0; i < d.length; i += 4) {
      const y = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      if (y > v) v = y;
    }
    if (this.dark === null) { this.dark = v; this.bright = v; }
    this.dark = v < this.dark ? v : 0.98 * this.dark + 0.02 * v;
    this.bright = v > this.bright ? v : 0.995 * this.bright + 0.005 * v;
    const span = this.bright - this.dark;
    if (span < 40) { this.on = false; return { on: false, confidence: 0, value: v, threshold: null }; }
    this.on = hysteresis(this.on, v, this.dark + 0.6 * span, this.dark + 0.4 * span);
    const mid = this.dark + 0.5 * span;
    return { on: this.on, confidence: Math.min(1, Math.abs(v - mid) / (0.5 * span)), value: v, threshold: mid };
  },
};

export const blink = {
  id: "blink", label: "Eye blink", hint: "Short deliberate blink = dot, long blink (about 1 s) = dash. Pause with eyes open between letters. Arm first so natural blinks are ignored.",
  defaultSplit: 0.4, minOn: 0.08, minOff: 0.06, valueLabel: "eye closure", range: [0, 1],
  async init() {
    const v = await mp();
    this.model = await create(v.FaceLandmarker, {
      baseOptions: { modelAssetPath: `${MODELS}/face_landmarker/face_landmarker/float16/1/face_landmarker.task` },
      runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true,
      // looser than the 0.5 defaults: old, small or blurry footage (a 320x240 film) still gets tracked
      minFaceDetectionConfidence: 0.3, minFacePresenceConfidence: 0.3, minTrackingConfidence: 0.3,
    });
    this.reset();
  },
  reset() { this.on = false; this.pad = false; },
  // MediaPipe blendshapes score eye closure 0-1; both eyes must close, so a wink does not count.
  // If no face is found, retry on a padded frame (close-ups), and stay padded while that works.
  read(src, tMs) {
    let res, unmap = (p) => p;
    const tryPad = () => { const pd = padded(src); unmap = pd.unmap; return this.model.detectForVideo(pd.canvas, nextTs(this.model, tMs)); };
    res = this.pad ? tryPad() : this.model.detectForVideo(src, nextTs(this.model, tMs));
    if (!res.faceBlendshapes?.length) {
      unmap = (p) => p;
      res = this.pad ? this.model.detectForVideo(src, nextTs(this.model, tMs)) : tryPad();
      if (res.faceBlendshapes?.length) this.pad = !this.pad;
    }
    if (!res.faceBlendshapes?.length) { this.on = false; return { on: false, confidence: 0, value: null, threshold: 0.425, found: false }; }
    const s = Object.fromEntries(res.faceBlendshapes[0].categories.map((c) => [c.categoryName, c.score]));
    const closure = Math.min(s.eyeBlinkLeft, s.eyeBlinkRight);
    this.on = hysteresis(this.on, closure, 0.5, 0.35);
    return { on: this.on, confidence: Math.min(1, Math.abs(closure - 0.425) / 0.425), value: closure, threshold: 0.425, found: true,
             points: res.faceLandmarks[0] ? [33, 133, 159, 145, 362, 263, 386, 374].map((i) => unmap(res.faceLandmarks[0][i])) : [] };
  },
};

export const tap = {
  id: "tap", label: "Finger press", hint: "Touch your index fingertip to your thumb. Short touch = dot, long touch = dash.",
  defaultSplit: 0.35, minOn: 0.06, minOff: 0.05, valueLabel: "pinch distance", range: [0, 1.2],
  async init() {
    const v = await mp();
    this.model = await create(v.HandLandmarker, {
      baseOptions: { modelAssetPath: `${MODELS}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` },
      runningMode: "VIDEO", numHands: 1,
    });
    this.reset();
  },
  reset() { this.on = false; },
  // thumb tip (4) to index tip (8), divided by hand size (wrist 0 to middle knuckle 9)
  read(src, tMs) {
    const res = this.model.detectForVideo(src, nextTs(this.model, tMs));
    if (!res.landmarks?.length) { this.on = false; return { on: false, confidence: 0, value: 1.2, threshold: 0.3 }; }
    const lm = res.landmarks[0];
    const dist = (a, b) => Math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y);
    const pinch = dist(4, 8) / (dist(0, 9) + 1e-6);
    this.on = hysteresis(this.on, pinch, 0.25, 0.35, false);
    return { on: this.on, confidence: Math.min(1, Math.abs(pinch - 0.3) / 0.3), value: pinch, threshold: 0.3,
             points: [lm[4], lm[8]] };
  },
};

// Not Morse: sign language. ASL fingerspelling, one letter per hand shape. MediaPipe
// HandLandmarker gives 21 points; a small network trained on real signers' landmarks
// (silent_signal/train_signs.py, same weights as the Python side) names the letter.
// Hold a letter HOLD seconds to type it; lower the hand for SPACE seconds for a space.
export const SIGN = { HOLD: 0.4, MIN_CONF: 0.6, GRACE: 0.15, SPACE: 1.0 };

// 21 landmarks -> 63 numbers: relative to the wrist, scaled by palm length (as in signs.py)
export function signFeatures(lm) {
  const w = lm[0], size = Math.hypot(lm[9].x - w.x, lm[9].y - w.y) + 1e-6, f = [];
  for (const q of lm) f.push((q.x - w.x) / size, (q.y - w.y) / size, (q.z - w.z) / size);
  return f;
}

export function letterProbs(net, f) {
  let h = f.map((v, i) => (v - net.mean[i]) / net.std[i]);
  net.layers.forEach((L, k) => {
    const out = L.b.slice();
    for (let i = 0; i < h.length; i++) { const hi = h[i]; if (hi) for (let j = 0; j < out.length; j++) out[j] += hi * L.W[i][j]; }
    h = k < net.layers.length - 1 ? out.map((v) => Math.max(0, v)) : out;
  });
  const m = Math.max(...h), e = h.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

// per-frame (time, letter or null, confidence) -> typed letters and spaces (as Typer in signs.py)
export class SignTyper {
  constructor(o = SIGN) { Object.assign(this, { hold: o.HOLD, minConf: o.MIN_CONF, grace: o.GRACE, space: o.SPACE }); this.reset(); }
  reset() { this.current = null; this.since = 0; this.lastSeen = -Infinity; this.fired = false; this.handGone = null; this.typed = []; }
  push(t, letter, conf, hand = true) {
    let out = null;
    if (!hand) {
      if (this.handGone === null) this.handGone = t;
      if (t - this.handGone >= this.space && this.typed.length && this.typed.at(-1).ch !== " ") out = " ";
    } else this.handGone = null;
    const cand = hand && letter && conf >= this.minConf ? letter : null;
    if (cand !== null && cand === this.current) this.lastSeen = t;
    else if (cand === null && this.current !== null && t - this.lastSeen <= this.grace) { /* a short blur: keep holding */ }
    else if (cand !== this.current) { this.current = cand; this.since = t; this.lastSeen = t; this.fired = false; }
    if (this.current && cand === this.current && !this.fired && t - this.since >= this.hold) { this.fired = true; out = this.current; }   // type on a sure frame
    if (out) this.typed.push({ t, ch: out, confidence: conf });
    return out;
  }
}

export const sign = {
  id: "sign", label: "Sign language", isSign: true,
  hint: "ASL fingerspelling: hold each letter steady for about half a second. Lower your hand for a second to start a new word. One hand, palm toward the camera.",
  valueLabel: "letter confidence", range: [0, 1],
  async init() {
    const v = await mp();
    this.model = await create(v.HandLandmarker, {
      baseOptions: { modelAssetPath: `${MODELS}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` },
      runningMode: "VIDEO", numHands: 1,
    });
    this.net = await (await fetch("models/asl_letters.json")).json();
    this.reset();
  },
  reset() { this.typer = new SignTyper(); },
  // returns {word} once when a letter (or a space) is typed
  read(src, tMs) {
    const t = tMs / 1000, res = this.model.detectForVideo(src, nextTs(this.model, tMs));
    const lm = res.landmarks?.[0];
    let letter = null, conf = 0;
    if (lm) {
      const p = letterProbs(this.net, signFeatures(lm));
      const i = p.indexOf(Math.max(...p));
      letter = this.net.letters[i]; conf = p[i];
    }
    const word = this.typer.push(t, letter, conf, !!lm);
    const ty = this.typer, held = ty.current ? Math.min(1, (t - ty.since) / ty.hold) : 0;
    return { on: !!(lm && conf >= SIGN.MIN_CONF), confidence: conf, value: conf, threshold: SIGN.MIN_CONF, found: !!lm,
             letter: lm ? letter : null, held, word, points: lm ?? [] };
  },
};

export const CHANNELS = { blink, tap, flash, sign };
