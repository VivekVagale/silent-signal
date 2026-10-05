import { CHANNELS } from "./detectors.js";
import { DOT_DASH, LETTER, MORSE, PulseTracker, WORD, assemble, decodeMorse, decodePulses, encode } from "./morse.js";

const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), octx = overlay.getContext("2d");
const scope = $("scope"), sctx = scope.getContext("2d");
const ICON = { blink: "i-eye", tap: "i-hand", flash: "i-flash", gesture: "i-sign" };
const SUB = { blink: "Face landmarks", tap: "Hand landmarks", flash: "Image brightness", gesture: "Word shortcuts" };
const SCOPE_S = 10;
const CUT = 30, MAX_LOST = 0.25;   // grey-level change that means a new shot; seconds without a face before we stop guessing
const thumb = Object.assign(document.createElement("canvas"), { width: 80, height: 60 });
const tctx = thumb.getContext("2d", { willReadFrequently: true });
let prevThumb = null;
// how much this frame differs from the previous one (mean grey-level change, 0-255): a cut is a big jump
function frameChange() {
  tctx.drawImage(video, 0, 0, 80, 60);
  const d = tctx.getImageData(0, 0, 80, 60).data, g = new Float32Array(80 * 60);
  for (let i = 0; i < g.length; i++) g[i] = 0.299 * d[4 * i] + 0.587 * d[4 * i + 1] + 0.114 * d[4 * i + 2];
  let sum = 0;
  if (prevThumb) for (let i = 0; i < g.length; i++) sum += Math.abs(g[i] - prevThumb[i]);
  const change = prevThumb ? sum / g.length : 0;
  prevThumb = g;
  return change;
}

const state = {
  mode: "live", channel: CHANNELS.blink, armed: false, running: false, stream: null,
  tracker: null, words: [], trace: [], t0: 0, lastReading: null, frames: 0, fpsT: 0,
  // uploaded-video analysis: every frame's reading, then a calibrated, user-correctable result
  series: [], result: null, edits: null, analyzing: 0, playing: false,
};
const freshEdits = () => ({ deleted: new Set(), flipped: new Set(), added: [] });

// ---------- UI setup ----------
for (const ch of Object.values(CHANNELS)) {
  const b = document.createElement("button");
  b.className = "channel"; b.dataset.id = ch.id;
  b.innerHTML = `<span class="name"><svg class="i"><use href="#${ICON[ch.id]}"/></svg>${ch.label}</span><span class="sub">${SUB[ch.id]}</span>`;
  b.onclick = () => selectChannel(ch.id);
  $("channels").append(b);
}
$("chart").innerHTML = Object.entries(MORSE).filter(([k]) => /[A-Z0-9]/.test(k))
  .map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("");

function setStatus(text, cls = "") { $("status").className = `status ${cls}`; $("statusText").textContent = text; }

// ---------- channels ----------
async function selectChannel(id) {
  const ch = CHANNELS[id];
  document.querySelectorAll(".channel").forEach((b) => b.classList.toggle("on", b.dataset.id === id));
  $("hint").textContent = ch.hint;
  $("speedRow").classList.toggle("hidden", !!ch.isGesture);
  $("scopeLabel").textContent = ch.valueLabel.toUpperCase();
  setStatus("LOADING MODEL");
  state.channel = ch;
  try {
    if (!ch.ready) { await ch.init(); ch.ready = true; }
  } catch (e) {
    setStatus("MODEL FAILED"); $("hint").textContent = `Could not load this channel: ${e.message}`; return;
  }
  ch.reset();
  clearSession();
  setStatus(state.running ? "READY" : "STANDBY", state.running ? "ready" : "");
  updateArm();
}

function clearSession() {
  const ch = state.channel;
  state.tracker = new PulseTracker({ minOn: ch.minOn ?? 0.05, minOff: ch.minOff ?? 0.04 });
  state.words = []; state.trace = [];
  state.series = []; state.result = null; state.edits = freshEdits();
  $("review").classList.add("hidden"); $("reviewNote").classList.add("hidden");
  scope.classList.remove("timeline"); $("scopeTitle").textContent = "SIGNAL SCOPE";
  if (ch.reset) ch.reset();
  render(0); renderLog();
}

// ---------- decoding + display (live) ----------
function currentUnit() { return $("autoSpeed").checked ? null : +$("unit").value; }

function render(now) {
  if (state.result) return renderResult();
  const ch = state.channel, pulses = state.tracker?.pulses ?? [];
  let text, morse, unit = null;
  if (ch.isGesture) {
    text = state.words.join(" "); morse = encode(text);
  } else {
    const d = decodePulses(pulses, ch.defaultSplit, currentUnit());
    unit = d.unit; morse = d.morse; text = d.text;
    // live gap state: after the last pulse, show whether we are between marks, letters or words
    const last = pulses[pulses.length - 1];
    let gapTxt = "";
    if (state.tracker?.active) gapTxt = "SIGNAL ON";
    else if (last && now) {
      const gap = now - last.end;
      if (gap >= WORD * unit) { gapTxt = "WORD BREAK"; text += " "; morse += " /"; }
      else if (gap >= LETTER * unit) gapTxt = "LETTER BREAK";
      else gapTxt = "LISTENING…";
    }
    $("gap").textContent = pulses.length || state.tracker?.active ? gapTxt : "";
    if ($("autoSpeed").checked && pulses.length >= 3) { $("unit").value = unit.toFixed(2); $("unitV").textContent = `dot ${unit.toFixed(2)} s`; }
  }
  showText(text, morse, state.tracker?.active);
  showStats(ch.isGesture ? state.words.length : pulses.length, unit, pulses.map((p) => p.confidence), pulses.length);
}

function showText(text, morse, pending = false) {
  $("text").innerHTML = `${escapeHtml(text)}<span class="cursor"></span>`;
  $("morse").innerHTML = escapeHtml(morse) + (pending ? `<span class="pending"> ▮</span>` : "");
}
function showStats(n, unit, confs, pulseCount) {
  $("sN").textContent = n;
  $("sUnit").textContent = unit && pulseCount >= 2 ? Math.round(unit * 1000) : "–";
  $("sWpm").textContent = unit && pulseCount >= 2 ? (1.2 / unit).toFixed(1) : "–";
  $("sConf").textContent = confs.length ? Math.round((100 * confs.reduce((a, b) => a + b, 0)) / confs.length) + "%" : "–";
}

function renderLog() {
  if (state.result) return renderResultLog();
  const ch = state.channel;
  const rows = ch.isGesture
    ? state.words.map((w, i) => `<tr><td>${i + 1}</td><td>${w.t.toFixed(2)}s</td><td>hold</td><td class="mark">${escapeHtml(w.word)}</td><td>${confBar(w.confidence)}</td><td></td></tr>`)
    : (() => {
        const d = decodePulses(state.tracker.pulses, ch.defaultSplit, currentUnit());
        return state.tracker.pulses.map((p, i) => `<tr><td>${i + 1}</td><td>${(p.start - state.t0).toFixed(2)}s</td>
          <td>${Math.round((p.end - p.start) * 1000)} ms</td><td class="mark">${d.marks[i] === "." ? "· dot" : "— dash"}</td><td>${confBar(p.confidence)}</td><td></td></tr>`);
      })();
  $("log").innerHTML = rows.length ? rows.reverse().join("") : `<tr><td colspan="6" class="empty">No signals yet.</td></tr>`;
}
const confBar = (c) => `<span class="conf"><i style="width:${Math.round(c * 100)}%"></i></span>${Math.round(c * 100)}%`;
const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------- one live frame ----------
function processFrame(src, t) {
  const ch = state.channel;
  if (!ch.ready) return;
  const r = ch.read(src, t * 1000);
  state.lastReading = r;
  state.trace.push({ t, v: r.value, on: r.on && state.armed, thr: r.threshold });
  while (state.trace.length && state.trace[0].t < t - SCOPE_S) state.trace.shift();

  if (state.armed) {
    if (ch.isGesture) {
      if (r.word) { state.words.push({ word: r.word, t: t - state.t0, confidence: r.confidence }); renderLog(); }
    } else if (state.tracker.push(t, r.on, r.confidence)) {
      renderLog();
    }
  }
  $("sig").classList.toggle("on", !!(r.on && state.armed));
  drawOverlay(r);
  drawScope(t);
  render(t);
}

function drawOverlay(r) {
  const w = overlay.clientWidth, h = overlay.clientHeight;
  if (overlay.width !== w) { overlay.width = w; overlay.height = h; }
  octx.clearRect(0, 0, w, h);
  if (!r) return;
  // map normalized video coords to the cover-fitted canvas
  const vw = video.videoWidth || 4, vh = video.videoHeight || 3, s = Math.max(w / vw, h / vh);
  const ox = (w - vw * s) / 2, oy = (h - vh * s) / 2;
  const P = (p) => [ox + p.x * vw * s, oy + p.y * vh * s];
  octx.fillStyle = r.on && (state.armed || state.analyzing) ? "#f59e0b" : "#818cf8";
  for (const p of r.points ?? []) { const [x, y] = P(p); octx.beginPath(); octx.arc(x, y, 3.5, 0, 7); octx.fill(); }
  if (r.gesture) {
    octx.font = "600 20px 'Share Tech Mono', monospace"; octx.fillStyle = "#f59e0b";
    const label = `${r.gesture.replace("_", " ").toUpperCase()}  ${Math.round(r.held * 100)}%`;
    const x = state.mode === "live" ? w - 16 : 16;
    octx.save();
    if (state.mode === "live") { octx.translate(w, 0); octx.scale(-1, 1); }   // canvas is mirrored in live mode
    octx.fillText(label, state.mode === "live" ? w - x : x, h - 20);
    octx.restore();
  }
}

function sizeScope() {
  const w = scope.clientWidth, h = scope.clientHeight;
  if (scope.width !== w * devicePixelRatio) { scope.width = w * devicePixelRatio; scope.height = h * devicePixelRatio; }
  sctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  sctx.clearRect(0, 0, w, h);
  return [w, h];
}

// the live scope: last 10 s, scrolling. Frames with no reading (no face) leave a gap in the line.
function drawScope(now) {
  const [w, h] = sizeScope();
  const [lo, hi] = state.channel.range, x = (t) => w - ((now - t) / SCOPE_S) * w, y = (v) => h - 8 - ((v - lo) / (hi - lo)) * (h - 16);
  sctx.strokeStyle = "rgba(148,163,184,.08)";
  for (let s = 0; s <= SCOPE_S; s++) { sctx.beginPath(); sctx.moveTo(x(now - s), 0); sctx.lineTo(x(now - s), h); sctx.stroke(); }
  sctx.fillStyle = "rgba(245,158,11,.16)";
  for (const p of state.trace) if (p.on) sctx.fillRect(x(p.t) - 2, 0, 4, h);
  const thr = state.trace.at(-1)?.thr;
  if (thr != null) { sctx.setLineDash([4, 4]); sctx.strokeStyle = "rgba(129,140,248,.6)"; sctx.beginPath(); sctx.moveTo(0, y(thr)); sctx.lineTo(w, y(thr)); sctx.stroke(); sctx.setLineDash([]); }
  plotLine(state.trace, x, y);
}
function plotLine(points, x, y) {
  sctx.strokeStyle = "#f59e0b"; sctx.lineWidth = 2; sctx.beginPath();
  let pen = false;
  for (const p of points) {
    if (p.v == null) { pen = false; continue; }
    if (pen) sctx.lineTo(x(p.t), y(p.v)); else sctx.moveTo(x(p.t), y(p.v));
    pen = true;
  }
  sctx.stroke(); sctx.lineWidth = 1;
}

// ---------- live camera ----------
async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
  } catch (e) { $("phText").textContent = `Camera not available: ${e.message}`; return; }
  video.srcObject = state.stream; video.classList.add("mirror"); overlay.classList.add("mirror");
  await video.play();
  $("placeholder").classList.add("hidden");
  state.running = true; state.t0 = performance.now() / 1000;
  setStatus("READY", "ready"); $("badge").textContent = "LIVE"; updateArm();
  const loop = () => {
    if (!state.running || state.mode !== "live") return;
    const t = performance.now() / 1000;
    processFrame(video, t);
    state.frames++;
    if (t - state.fpsT > 1) { $("fps").textContent = `${state.frames} FPS`; state.frames = 0; state.fpsT = t; }
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

function stopCamera() {
  state.running = false;
  state.stream?.getTracks().forEach((t) => t.stop()); state.stream = null;
  video.srcObject = null;
}

// ---------- uploaded video ----------
// Pass 1: read the clip frame by frame (see below). Pass 2: calibrate to this video and
// decode; the user can then replay it and correct the result.
async function analyzeVideo(file) {
  stopCamera(); stopPlayback();
  const token = ++state.analyzing;
  video.classList.remove("mirror"); overlay.classList.remove("mirror");
  video.srcObject = null; video.src = URL.createObjectURL(file); video.muted = true;
  try {
    await new Promise((resolve, reject) => { video.onloadedmetadata = resolve; video.onerror = () => reject(new Error("this browser cannot play that video format")); });
  } catch (e) { setStatus("CANNOT READ VIDEO"); $("phText").textContent = `Could not open the video: ${e.message}.`; return; }
  $("placeholder").classList.add("hidden"); $("pick2").classList.remove("hidden");
  const ch = state.channel; ch.reset(); clearSession();
  state.t0 = 0; updateArm();
  setStatus("ANALYZING", "armed"); $("badge").textContent = "READING FRAMES";
  $("progress").classList.remove("hidden");
  const dur = video.duration;
  prevThumb = null;

  const sample = (t) => {
    const r = ch.read(video, t * 1000);
    state.series.push({ t, v: r.value, on: r.on, conf: r.confidence, found: r.found !== false, cut: frameChange() > CUT });
    state.trace.push({ t, v: r.value, on: r.on, thr: r.threshold });
    while (state.trace.length && state.trace[0].t < t - SCOPE_S) state.trace.shift();
    $("sig").classList.toggle("on", !!r.on);
    drawOverlay(r); drawScope(t);
    $("progress").firstElementChild.style.width = `${(100 * t) / dur}%`;
    if (state.series.length % 3 === 0) showLive(t);
  };
  // Step through at a fixed 20 frames per second. After each jump, wait until the browser
  // reports that the frame for that time is actually on screen (requestVideoFrameCallback)
  // before reading it. Playing the clip and reading frames as they came lost most of them on
  // slower machines (97 of 616 frames); blindly seeking could read the previous frame.
  const STEP = 1 / 20, hasRVFC = "requestVideoFrameCallback" in HTMLVideoElement.prototype;
  const shown = () => new Promise((resolve) => {
    let done = false; const finish = () => { if (!done) { done = true; resolve(); } };
    if (hasRVFC) video.requestVideoFrameCallback(finish);
    // A tab that is hidden or covered never paints, so waiting only for a paint stalled every
    // frame until the timeout (2 s a frame). After "seeked" the frame is decoded: move on after
    // two paints or 60 ms, whichever comes first (at once when hidden: timers are slowed there).
    video.addEventListener("seeked", () => {
      if (document.hidden) return finish();
      requestAnimationFrame(() => requestAnimationFrame(finish));
      setTimeout(finish, 60);
    }, { once: true });
    setTimeout(finish, 1500);                   // never hang on a frame the decoder will not show
  });
  // Never run ahead of the clip's own clock, so the message forms at the speed it was sent,
  // as with the live camera. A slow detector just lags; a hidden tab goes flat out.
  const start = performance.now();
  for (let t = 0; t <= dur && token === state.analyzing; t += STEP) {
    const ready = shown();
    video.currentTime = t;
    await ready;
    if (token !== state.analyzing) return;
    sample(t);
    const ahead = t * 1000 - (performance.now() - start);
    if (ahead > 0 && !document.hidden) await new Promise((r) => setTimeout(r, ahead));
  }
  if (token !== state.analyzing) return;
  $("progress").firstElementChild.style.width = "100%";
  $("fps").textContent = `${state.series.length} frames read`;
  setStatus("REVIEW", "ready"); $("badge").textContent = "REVIEW";
  $("review").classList.remove("hidden");
  $("sensRow").classList.toggle("hidden", ch.id !== "blink");
  scope.classList.add("timeline"); $("scopeTitle").textContent = "TIMELINE · CLICK TO JUMP";
  recompute();
}

const pct = (arr, q) => { const s = [...arr].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))]; };

// Calibrate to this video, find the pulses, apply the user's corrections, decode.
function recompute() {
  const { result, note } = decodeSeries(state.series);
  state.result = result;
  finishRecompute(note);
}

// The frames read so far -> pulses and text. live = true while the clip is still being read:
// calibration then uses only the frames seen so far, and a signal still ON stays open.
function decodeSeries(S, live = false) {
  const ch = state.channel, note = [];
  let states, thr = null;
  if (ch.id === "blink") {
    // Every face and camera reads differently (in an old 320x240 film the open eye
    // already scores 0.44), so the thresholds come from this video: "open" is the
    // 35th percentile of eye closure, "closed" the 99th, and the switch points sit
    // between them at the chosen sensitivity, with hysteresis.
    const vals = S.filter((s) => s.found).map((s) => s.v);
    const seen = vals.length / Math.max(1, S.length);
    if (seen < 0.95) note.push(`Face found in ${Math.round(seen * 100)}% of frames; short gaps are bridged, longer ones count as unseen.`);
    if (vals.length < 10) { note.push("No face found, so no blinks could be read."); return { result: { pulses: [], marks: [], kinds: [], morse: "", text: "", thr: null, unit: null, key: (p) => p.start.toFixed(2) }, note, active: false }; }
    const open = pct(vals, 0.35), closed = pct(vals, 0.99), span = closed - open, sens = +$("sens").value;
    if (span < 0.12) note.push("The eyes barely change in this clip: no clear blinks to read.");
    const onAt = closed - sens * span, offAt = Math.min(onAt, closed - (sens + 0.2) * span);
    thr = onAt;
    let on = false;
    // live, before the first blink the eyes have barely moved: thresholds would sit in the noise
    states = live && span < 0.12 ? S.map(() => false) : S.map((s) => { if (s.found) on = on ? s.v > offAt : s.v >= onAt; return on; });
  } else {
    states = S.map((s) => s.on);               // flash and press detectors already adapt to the scene
  }
  // per-frame states -> pulses, using the same cleaning as live mode
  const tr = new PulseTracker({ minOn: ch.minOn ?? 0.05, minOff: ch.minOff ?? 0.04 });
  S.forEach((s, i) => tr.push(s.t, states[i], s.conf ?? 1));
  if (S.length && !live) tr.push(S.at(-1).t + 0.01, false, 0);
  // Edited clips: the first frame of each shot, and any long stretch with no face, are "blind".
  // A signal touching one was cut off, so its length is unknown: drop it. A cut always ends the word.
  const blind = S.map((s) => !!s.cut);
  if (ch.id === "blink") {
    for (let i = 0; i < S.length;) {
      if (S[i].found) { i++; continue; }
      let j = i; while (j < S.length && !S[j].found) j++;
      if ((j < S.length ? S[j].t : S.at(-1).t) - S[i].t > MAX_LOST) for (let k = i; k < j; k++) blind[k] = true;
      i = j;
    }
  }
  const dt = S.length > 1 ? S[1].t - S[0].t : 0, breaks = S.filter((_, i) => blind[i]).map((s) => s.t);
  const intact = (p) => !breaks.some((t) => p.start - dt - 1e-6 <= t && t <= p.end + 1e-6);
  const cuts = S.filter((s, i) => s.cut && !S[i - 1]?.cut).length, dropped = tr.pulses.filter((p) => !intact(p)).length;
  if (cuts) note.push(`Edited video: ${cuts} cut${cuts > 1 ? "s" : ""} found. Each cut ends the word${dropped ? `; ${dropped} signal${dropped > 1 ? "s" : ""} cut off by an edit ${dropped > 1 ? "were" : "was"} left out` : ""}.`);
  const key = (p) => p.start.toFixed(2);
  let pulses = tr.pulses.filter((p) => intact(p) && !state.edits.deleted.has(key(p)));
  pulses = [...pulses, ...state.edits.added.filter((p) => !state.edits.deleted.has(key(p)))].sort((a, b) => a.start - b.start);
  const d = decodePulses(pulses, ch.defaultSplit, currentUnit(), true, breaks);
  const marks = d.marks.map((m, i) => (state.edits.flipped.has(key(pulses[i])) ? (m === "." ? "-" : ".") : m));
  const morse = assemble(marks, d.kinds);
  return { result: { pulses, marks, kinds: d.kinds, morse, text: decodeMorse(morse).text, thr, unit: d.unit, key }, note, active: tr.active };
}

// While the clip is being read: show the message as it forms, like the live camera does.
function showLive(t) {
  const { result: R, active } = decodeSeries(state.series, true);
  const last = R.pulses.at(-1);
  let gapTxt = "";
  if (active) gapTxt = "SIGNAL ON";
  else if (last && R.unit) {
    const gap = t - last.end;
    gapTxt = gap >= WORD * R.unit ? "WORD BREAK" : gap >= LETTER * R.unit ? "LETTER BREAK" : "LISTENING…";
  }
  showText(R.text, R.morse, active);
  $("gap").textContent = gapTxt;
  showStats(R.pulses.length, R.unit, R.pulses.map((p) => p.confidence ?? 1), R.pulses.length);
  $("log").innerHTML = R.pulses.length
    ? R.pulses.map((p, i) => `<tr><td>${i + 1}</td><td>${p.start.toFixed(2)}s</td><td>${Math.round((p.end - p.start) * 1000)} ms</td>
        <td class="mark">${R.marks[i] === "." ? "· dot" : "— dash"}</td><td>${confBar(p.confidence ?? 1)}</td><td></td></tr>`).reverse().join("")
    : `<tr><td colspan="6" class="empty">No signals yet.</td></tr>`;
}

// Review playback: type the message out as the video reaches each signal.
function showUpTo(t) {
  const R = state.result, n = R.pulses.filter((p) => p.end <= t).length;
  const active = R.pulses.some((p) => p.start <= t && t < p.end);
  const morse = assemble(R.marks.slice(0, n), R.kinds.slice(0, Math.max(0, n - 1)));
  showText(decodeMorse(morse).text, morse, active);
}
function finishRecompute(note) {
  const ed = state.edits, n = ed.deleted.size + ed.flipped.size + ed.added.length;
  if (n) note.push(`${n} correction${n > 1 ? "s" : ""} applied.`);
  $("reviewNote").textContent = note.join(" ");
  $("reviewNote").classList.toggle("hidden", !note.length);
  renderResult(); renderResultLog(); drawTimeline();
}

function renderResult() {
  const R = state.result;
  showText(R.text, R.morse);
  $("gap").textContent = "";
  showStats(R.pulses.length, R.unit, R.pulses.map((p) => p.confidence), R.pulses.length);
}

function renderResultLog() {
  const R = state.result, ed = state.edits, now = video.currentTime;
  if (!R.pulses.length) { $("log").innerHTML = `<tr><td colspan="6" class="empty">No signals found. Play the video and use "Add blink here" to mark them.</td></tr>`; return; }
  $("log").innerHTML = R.pulses.map((p, i) => {
    const k = R.key(p), cls = [p.start <= now && now <= p.end ? "active" : "", ed.flipped.has(k) ? "edited" : "", ed.added.some((a) => R.key(a) === k) ? "added" : ""].join(" ");
    return `<tr class="${cls}" data-i="${i}"><td>${i + 1}</td><td>${p.start.toFixed(2)}s</td><td>${Math.round((p.end - p.start) * 1000)} ms</td>
      <td class="mark">${R.marks[i] === "." ? "· dot" : "— dash"}</td><td>${confBar(p.confidence ?? 1)}</td>
      <td><span class="fix"><button data-act="flip" data-k="${k}" title="Switch dot / dash">·/—</button><button data-act="del" data-k="${k}" title="Not a signal: remove">✕</button></span></td></tr>`;
  }).join("");
}
$("log").addEventListener("click", (e) => {
  const b = e.target.closest("button[data-act]");
  if (b && state.result) {
    const set = b.dataset.act === "flip" ? state.edits.flipped : state.edits.deleted;
    set.has(b.dataset.k) ? set.delete(b.dataset.k) : set.add(b.dataset.k);
    return recompute();
  }
  const row = e.target.closest("tr[data-i]");   // click a row: jump the video there
  if (row && state.result) { stopPlayback(); video.currentTime = state.result.pulses[+row.dataset.i].start; drawTimeline(); }
});

// the whole clip on one strip: reading, threshold, pulses (amber = dash, indigo = dot), playhead
function drawTimeline() {
  const [w, h] = sizeScope(), R = state.result, dur = video.duration || 1;
  const [lo, hi] = state.channel.range, x = (t) => (t / dur) * w, y = (v) => h - 8 - ((v - lo) / (hi - lo)) * (h - 16);
  if (!R) return;
  R.pulses.forEach((p, i) => {
    sctx.fillStyle = R.marks[i] === "-" ? "rgba(245,158,11,.28)" : "rgba(129,140,248,.30)";
    sctx.fillRect(x(p.start), 0, Math.max(2, x(p.end) - x(p.start)), h);
  });
  if (R.thr != null) { sctx.setLineDash([4, 4]); sctx.strokeStyle = "rgba(129,140,248,.7)"; sctx.beginPath(); sctx.moveTo(0, y(R.thr)); sctx.lineTo(w, y(R.thr)); sctx.stroke(); sctx.setLineDash([]); }
  plotLine(state.series, x, y);
  sctx.fillStyle = "#e2e8f0"; sctx.fillRect(x(video.currentTime) - 1, 0, 2, h);
}
scope.addEventListener("click", (e) => {
  if (!state.result) return;
  const r = scope.getBoundingClientRect();
  video.currentTime = ((e.clientX - r.left) / r.width) * (video.duration || 0);
  if (!state.playing) setTimeout(drawTimeline, 50);
});

// review playback at normal speed: the playhead, the active row and the signal light follow the video
function playLoop() {
  if (!state.playing) return;
  const t = video.currentTime, R = state.result;
  $("sig").classList.toggle("on", !!R?.pulses.some((p) => p.start <= t && t <= p.end));
  showUpTo(t);
  drawTimeline();
  document.querySelectorAll("#log tr[data-i]").forEach((tr) => {
    const p = R.pulses[+tr.dataset.i];
    tr.classList.toggle("active", p.start <= t && t <= p.end);
  });
  if (video.ended) return stopPlayback();
  requestAnimationFrame(playLoop);
}
function stopPlayback() {
  const was = state.playing;
  state.playing = false; video.pause();
  if (was && state.result) video.ended ? renderResult() : showUpTo(video.currentTime);   // paused: keep the text where the video is
  $("playBtn").querySelector("span").textContent = "Play with signals";
  $("playBtn").querySelector("use").setAttribute("href", "#i-play");
}
$("playBtn").onclick = () => {
  if (state.playing) return stopPlayback();
  if (video.ended || video.currentTime >= video.duration - 0.05) video.currentTime = 0;
  state.playing = true; video.muted = false; video.playbackRate = 1;
  showUpTo(video.currentTime);
  video.play();
  $("playBtn").querySelector("span").textContent = "Pause";
  $("playBtn").querySelector("use").setAttribute("href", "#i-pause");
  requestAnimationFrame(playLoop);
};
// a blink the detector missed: add one dot-length mark at the current point of the video
$("addMark").onclick = () => {
  if (!state.result) return;
  const t = video.currentTime, len = Math.max(0.12, state.result.unit || state.channel.defaultSplit / DOT_DASH);
  state.edits.added.push({ start: t, end: t + len, confidence: 1 });
  recompute();
};
$("resetEdits").onclick = () => { state.edits = freshEdits(); recompute(); };
$("sens").oninput = (e) => {
  $("sensV").textContent = (+e.target.value).toFixed(2);
  try { localStorage.setItem("ss.sens", e.target.value); } catch { /* storage blocked: fine */ }
  if (state.result) recompute();
};
try { const v = localStorage.getItem("ss.sens"); if (v) { $("sens").value = v; $("sensV").textContent = (+v).toFixed(2); } } catch { /* ignore */ }

// ---------- controls ----------
function updateArm() {
  const b = $("arm");
  b.disabled = !state.running;
  b.classList.toggle("armed", state.armed);
  b.querySelector("span").textContent = state.armed ? "Armed · Space to pause" : "Arm (Space)";
  if (state.running) setStatus(state.armed ? "ARMED" : "READY", state.armed ? "armed" : "ready");
}
function toggleArm() {
  if (!state.running) return;
  state.armed = !state.armed;
  if (state.armed && state.tracker.pulses.length === 0) state.t0 = performance.now() / 1000;
  if (!state.armed && state.tracker.active) state.tracker.push(performance.now() / 1000, false, 0);
  updateArm();
}

$("start").onclick = startCamera;
// sample clips, both synthetic: a CG face blinking (MakeHuman, CC0) and a rendered torch
for (const b of document.querySelectorAll("button.sample")) b.onclick = async () => {
  await selectChannel(b.dataset.ch);
  const blob = await (await fetch(b.dataset.src)).blob();
  analyzeVideo(new File([blob], b.dataset.src.split("/").pop(), { type: "video/mp4" }));
};
$("arm").onclick = toggleArm;
$("clear").onclick = () => { if (state.mode === "video") { state.edits = freshEdits(); if (state.series.length) return recompute(); } clearSession(); };
for (const id of ["file", "file2"]) $(id).onchange = (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) analyzeVideo(f); };
$("autoSpeed").onchange = (e) => { $("unit").disabled = e.target.checked; state.result ? recompute() : (render(), renderLog()); };
$("unit").oninput = (e) => { $("unitV").textContent = `dot ${(+e.target.value).toFixed(2)} s`; state.result ? recompute() : (render(), renderLog()); };
$("speak").onclick = () => {
  const t = $("text").textContent.trim();
  if (t) { speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(t.toLowerCase())); }
};
$("copy").onclick = () => navigator.clipboard?.writeText($("text").textContent.trim());
$("mLive").onclick = () => setMode("live");
$("mVideo").onclick = () => setMode("video");
document.addEventListener("keydown", (e) => {
  if (e.target instanceof Element && e.target.matches("input, textarea, select")) return;
  if (e.code === "Space") {
    e.preventDefault();
    if (state.mode === "video" && state.result) $("playBtn").click(); else toggleArm();
  }
  if (e.key === "c" || e.key === "C") $("clear").click();
});

function setMode(mode) {
  state.mode = mode; state.armed = false; state.analyzing++;
  $("mLive").classList.toggle("on", mode === "live"); $("mVideo").classList.toggle("on", mode === "video");
  $("mLive").setAttribute("aria-selected", mode === "live"); $("mVideo").setAttribute("aria-selected", mode === "video");
  stopPlayback(); stopCamera(); video.removeAttribute("src"); video.load();
  $("placeholder").classList.remove("hidden"); $("progress").classList.add("hidden"); $("pick2").classList.add("hidden");
  $("start").classList.toggle("hidden", mode !== "live"); $("pick").classList.toggle("hidden", mode !== "video"); document.querySelectorAll("button.sample").forEach((b) => b.classList.toggle("hidden", mode !== "video"));
  $("phText").textContent = mode === "live"
    ? "Your camera feed is processed on this device only. Nothing is uploaded."
    : "Pick a clip of someone blinking, pressing or flashing Morse. Every frame is read on this device, then you can replay it and fix any mistakes.";
  $("badge").textContent = "STANDBY"; $("fps").textContent = "";
  clearSession(); updateArm(); setStatus("STANDBY");
}

selectChannel("blink");
