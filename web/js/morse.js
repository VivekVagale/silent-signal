// Morse engine: same logic as silent_signal/morse.py, so the browser and the
// Python benchmark behave identically. See that file for the reasoning.

export const MORSE = {
  A: ".-", B: "-...", C: "-.-.", D: "-..", E: ".", F: "..-.", G: "--.", H: "....", I: "..", J: ".---",
  K: "-.-", L: ".-..", M: "--", N: "-.", O: "---", P: ".--.", Q: "--.-", R: ".-.", S: "...", T: "-",
  U: "..-", V: "...-", W: ".--", X: "-..-", Y: "-.--", Z: "--..", 0: "-----", 1: ".----", 2: "..---",
  3: "...--", 4: "....-", 5: ".....", 6: "-....", 7: "--...", 8: "---..", 9: "----.", ".": ".-.-.-",
  ",": "--..--", "?": "..--..", "!": "-.-.--", "/": "-..-.", "@": ".--.-.", "'": ".----.", "-": "-....-",
};
const DECODE = Object.fromEntries(Object.entries(MORSE).map(([k, v]) => [v, k]));

export const DOT_DASH = Math.sqrt(3); // 1.73 units: geometric midpoint of 1 and 3
export const LETTER = Math.sqrt(3);   // gaps: 1 (inside letter) vs 3 (between letters)
export const WORD = Math.sqrt(21);    // 4.58 units: 3 (between letters) vs 7 (between words)

export function encode(text) {
  return text.toUpperCase().split(/\s+/).filter(Boolean)
    .map((w) => [...w].filter((c) => MORSE[c]).map((c) => MORSE[c]).join(" "))
    .filter(Boolean).join(" / ");
}

export function decodeMorse(morse) {
  let unknown = 0;
  const words = morse.split("/").map((w) => w.trim().split(/\s+/).filter(Boolean).map((code) => {
    if (!(code in DECODE)) unknown++;
    return DECODE[code] ?? "?";
  }).join("")).filter(Boolean);
  return { text: words.join(" "), unknown };
}

function twoMeansLog(values) {
  const logs = values.map(Math.log).sort((a, b) => a - b);
  let c1 = logs[0], c2 = logs[logs.length - 1];
  for (let i = 0; i < 50; i++) {
    const lo = logs.filter((x) => Math.abs(x - c1) <= Math.abs(x - c2));
    const hi = logs.filter((x) => Math.abs(x - c1) > Math.abs(x - c2));
    const n1 = lo.length ? lo.reduce((a, b) => a + b) / lo.length : c1;
    const n2 = hi.length ? hi.reduce((a, b) => a + b) / hi.length : c2;
    if (n1 === c1 && n2 === c2) break;
    c1 = n1; c2 = n2;
  }
  return [Math.exp(c1), Math.exp(c2)];
}

// Dot length in seconds, learned from ON durations (log-scale 2-means).
export function estimateUnit(durations, defaultSplit) {
  const lo = Math.min(...durations), hi = Math.max(...durations);
  if (hi >= 2 * lo) {
    const [dot, dash] = twoMeansLog(durations);
    return Math.sqrt((dot * dash) / 3);
  }
  const typical = Math.exp(durations.reduce((a, d) => a + Math.log(d), 0) / durations.length);
  return typical < defaultSplit ? typical : typical / 3;
}

// pulses: [{start, end, confidence}] in seconds. unit: fixed dot length, or null to learn it.
export function decodePulses(pulses, defaultSplit = 0.3, unit = null) {
  pulses = [...pulses].sort((a, b) => a.start - b.start);
  if (!pulses.length) return { morse: "", text: "", marks: [], unit: unit ?? defaultSplit / DOT_DASH, unknown: 0 };
  const u = unit ?? estimateUnit(pulses.map((p) => p.end - p.start), defaultSplit);
  const marks = pulses.map((p) => (p.end - p.start < DOT_DASH * u ? "." : "-"));
  let morse = marks[0];
  for (let i = 1; i < pulses.length; i++) {
    const gap = pulses[i].start - pulses[i - 1].end;
    if (gap >= WORD * u) morse += " / ";
    else if (gap >= LETTER * u) morse += " ";
    morse += marks[i];
  }
  const { text, unknown } = decodeMorse(morse);
  return { morse, text, marks, unit: u, unknown };
}

// Turns a stream of per-frame ON/OFF readings into pulses, live.
// minOn drops flickers, minOff merges pulses split by a dropout.
export class PulseTracker {
  constructor({ minOn = 0.05, minOff = 0.04 } = {}) {
    this.minOn = minOn; this.minOff = minOff;
    this.pulses = []; this.start = null; this.confs = []; this.lastT = 0;
  }
  push(t, on, confidence) {
    this.lastT = t;
    if (on) {
      if (this.start === null) {
        const prev = this.pulses[this.pulses.length - 1];
        if (prev && t - prev.end < this.minOff) {   // dropout: reopen the previous pulse
          this.pulses.pop(); this.start = prev.start; this.confs = [prev.confidence];
        } else { this.start = t; this.confs = []; }
      }
      this.confs.push(confidence);
      return null;
    }
    if (this.start === null) return null;
    const p = { start: this.start, end: t, confidence: this.confs.reduce((a, b) => a + b, 0) / this.confs.length };
    this.start = null;
    if (p.end - p.start < this.minOn) return null;
    this.pulses.push(p);
    return p;
  }
  get active() { return this.start !== null; }
  reset() { this.pulses = []; this.start = null; this.confs = []; }
}
