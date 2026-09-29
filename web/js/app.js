import { CHANNELS } from "./detectors.js";
import { DOT_DASH, LETTER, MORSE, PulseTracker, WORD, decodePulses, encode } from "./morse.js";

const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), octx = overlay.getContext("2d");
const scope = $("scope"), sctx = scope.getContext("2d");
const ICON = { blink: "i-eye", tap: "i-hand", flash: "i-flash", gesture: "i-sign" };
const SUB = { blink: "Face landmarks", tap: "Hand landmarks", flash: "Image brightness", gesture: "Word shortcuts" };
const SCOPE_S = 10;

const state = {
  mode: "live", channel: CHANNELS.blink, armed: false, running: false, stream: null,
  tracker: null, words: [], trace: [], t0: 0, lastReading: null, frames: 0, fpsT: 0,
};

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
  if (ch.reset) ch.reset();
  render(0);
}

// ---------- decoding + display ----------
function currentUnit() { return $("autoSpeed").checked ? null : +$("unit").value; }

function render(now) {
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
  $("text").innerHTML = `${escapeHtml(text)}<span class="cursor"></span>`;
  $("morse").innerHTML = escapeHtml(morse) + (state.tracker?.active ? `<span class="pending"> ▮</span>` : "");
  const confs = pulses.map((p) => p.confidence);
  $("sN").textContent = ch.isGesture ? state.words.length : pulses.length;
  $("sUnit").textContent = unit && pulses.length >= 2 ? Math.round(unit * 1000) : "–";
  $("sWpm").textContent = unit && pulses.length >= 2 ? (1.2 / unit).toFixed(1) : "–";
  $("sConf").textContent = confs.length ? Math.round((100 * confs.reduce((a, b) => a + b, 0)) / confs.length) + "%" : "–";
}

function renderLog() {
  const ch = state.channel;
  const rows = ch.isGesture
    ? state.words.map((w, i) => `<tr><td>${i + 1}</td><td>${w.t.toFixed(2)}s</td><td>hold</td><td class="mark">${escapeHtml(w.word)}</td><td>${confBar(w.confidence)}</td></tr>`)
    : (() => {
        const d = decodePulses(state.tracker.pulses, ch.defaultSplit, currentUnit());
        return state.tracker.pulses.map((p, i) => `<tr><td>${i + 1}</td><td>${(p.start - state.t0).toFixed(2)}s</td>
          <td>${Math.round((p.end - p.start) * 1000)} ms</td><td class="mark">${d.marks[i] === "." ? "· dot" : "— dash"}</td><td>${confBar(p.confidence)}</td></tr>`);
      })();
  $("log").innerHTML = rows.length ? rows.reverse().join("") : `<tr><td colspan="5" class="empty">No signals yet.</td></tr>`;
}
const confBar = (c) => `<span class="conf"><i style="width:${Math.round(c * 100)}%"></i></span>${Math.round(c * 100)}%`;
const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ---------- one frame ----------
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
  // map normalized video coords to the cover-fitted canvas
  const vw = video.videoWidth || 4, vh = video.videoHeight || 3, s = Math.max(w / vw, h / vh);
  const ox = (w - vw * s) / 2, oy = (h - vh * s) / 2;
  const P = (p) => [ox + p.x * vw * s, oy + p.y * vh * s];
  octx.fillStyle = r.on && state.armed ? "#f59e0b" : "#818cf8";
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

function drawScope(now) {
  const w = scope.clientWidth, h = scope.clientHeight;
  if (scope.width !== w * devicePixelRatio) { scope.width = w * devicePixelRatio; scope.height = h * devicePixelRatio; }
  sctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  sctx.clearRect(0, 0, w, h);
  const [lo, hi] = state.channel.range, x = (t) => w - ((now - t) / SCOPE_S) * w, y = (v) => h - 8 - ((v - lo) / (hi - lo)) * (h - 16);
  sctx.strokeStyle = "rgba(148,163,184,.08)";
  for (let s = 0; s <= SCOPE_S; s++) { sctx.beginPath(); sctx.moveTo(x(now - s), 0); sctx.lineTo(x(now - s), h); sctx.stroke(); }
  sctx.fillStyle = "rgba(245,158,11,.16)";
  for (const p of state.trace) if (p.on) sctx.fillRect(x(p.t) - 2, 0, 4, h);
  const thr = state.trace.at(-1)?.thr;
  if (thr != null) { sctx.setLineDash([4, 4]); sctx.strokeStyle = "rgba(129,140,248,.6)"; sctx.beginPath(); sctx.moveTo(0, y(thr)); sctx.lineTo(w, y(thr)); sctx.stroke(); sctx.setLineDash([]); }
  sctx.strokeStyle = "#f59e0b"; sctx.lineWidth = 2; sctx.beginPath();
  state.trace.forEach((p, i) => (i ? sctx.lineTo(x(p.t), y(p.v)) : sctx.moveTo(x(p.t), y(p.v))));
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

// ---------- uploaded video: step through at a fixed rate, using video time ----------
async function analyzeVideo(file) {
  stopCamera(); state.running = false;
  video.classList.remove("mirror"); overlay.classList.remove("mirror");
  video.srcObject = null; video.src = URL.createObjectURL(file);
  await new Promise((r) => (video.onloadedmetadata = r));
  $("placeholder").classList.add("hidden"); $("pick2").classList.remove("hidden");
  const ch = state.channel; ch.reset(); clearSession();
  state.armed = true; updateArm(); state.t0 = 0;
  setStatus("ANALYZING", "armed"); $("badge").textContent = "ANALYSIS";
  $("progress").classList.remove("hidden");
  const step = 1 / 20, dur = video.duration;
  for (let t = 0; t <= dur; t += step) {
    video.currentTime = t;
    await new Promise((r) => (video.onseeked = r));
    processFrame(video, t + 0.001);   // MediaPipe needs strictly increasing timestamps
    $("progress").firstElementChild.style.width = `${(100 * t) / dur}%`;
  }
  state.tracker.push(dur + 0.01, false, 0);   // close a pulse still open at the end
  render(0); renderLog();
  $("progress").firstElementChild.style.width = "100%";
  setStatus("DONE", "ready"); state.armed = false; updateArm();
  $("fps").textContent = `${Math.round(dur / step)} frames at 20 fps`;
}

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
$("sample").onclick = async () => {
  await selectChannel("flash");
  const blob = await (await fetch("samples/flash_sos_help.mp4")).blob();
  analyzeVideo(new File([blob], "flash_sos_help.mp4", { type: "video/mp4" }));
};
$("arm").onclick = toggleArm;
$("clear").onclick = () => { clearSession(); renderLog(); };
for (const id of ["file", "file2"]) $(id).onchange = (e) => e.target.files[0] && analyzeVideo(e.target.files[0]);
$("autoSpeed").onchange = (e) => { $("unit").disabled = e.target.checked; render(); renderLog(); };
$("unit").oninput = (e) => { $("unitV").textContent = `dot ${(+e.target.value).toFixed(2)} s`; render(); renderLog(); };
$("speak").onclick = () => {
  const t = $("text").textContent.trim();
  if (t) { speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(t.toLowerCase())); }
};
$("copy").onclick = () => navigator.clipboard?.writeText($("text").textContent.trim());
$("mLive").onclick = () => setMode("live");
$("mVideo").onclick = () => setMode("video");
document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea")) return;
  if (e.code === "Space") { e.preventDefault(); toggleArm(); }
  if (e.key === "c" || e.key === "C") { clearSession(); renderLog(); }
});

function setMode(mode) {
  state.mode = mode; state.armed = false;
  $("mLive").classList.toggle("on", mode === "live"); $("mVideo").classList.toggle("on", mode === "video");
  $("mLive").setAttribute("aria-selected", mode === "live"); $("mVideo").setAttribute("aria-selected", mode === "video");
  stopCamera(); video.removeAttribute("src"); video.load();
  $("placeholder").classList.remove("hidden"); $("progress").classList.add("hidden"); $("pick2").classList.add("hidden");
  $("start").classList.toggle("hidden", mode !== "live"); $("pick").classList.toggle("hidden", mode !== "video"); $("sample").classList.toggle("hidden", mode !== "video");
  $("phText").textContent = mode === "live"
    ? "Your camera feed is processed on this device only. Nothing is uploaded."
    : "Pick a clip of someone blinking, pressing or flashing Morse. It is analysed frame by frame on this device.";
  $("badge").textContent = "STANDBY"; $("fps").textContent = "";
  clearSession(); renderLog(); updateArm(); setStatus("STANDBY");
}

selectChannel("blink");
