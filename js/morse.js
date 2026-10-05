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

// [dot/dash boundary, unit] in seconds, learned from ON durations (see mark_split in morse.py).
// Two clear groups: boundary at the geometric midpoint of their centres, so a
// sender whose dashes are only 2.5x their dots is still read right.
export function markSplit(durations, defaultSplit) {
  const lo = Math.min(...durations), hi = Math.max(...durations);
  if (hi >= 2 * lo) {
    const [dot, dash] = twoMeansLog(durations);
    return [Math.sqrt(dot * dash), Math.sqrt((dot * dash) / 3)];
  }
  const typical = Math.exp(durations.reduce((a, d) => a + Math.log(d), 0) / durations.length);
  const unit = typical < defaultSplit ? typical : typical / 3;
  return [DOT_DASH * unit, unit];
}
export const estimateUnit = (durations, defaultSplit) => markSplit(durations, defaultSplit)[1];

function kmeansLog(values, k) {
  const logs = values.map(Math.log).sort((a, b) => a - b);
  // start evenly spaced from min to max (log scale) so a rare group, like one word gap, still gets a centre
  let c = Array.from({ length: k }, (_, i) => logs[0] + (i * (logs[logs.length - 1] - logs[0])) / (k - 1));
  for (let it = 0; it < 100; it++) {
    const groups = c.map(() => []);
    for (const x of logs) {
      let j = 0;
      for (let m = 1; m < k; m++) if (Math.abs(x - c[m]) < Math.abs(x - c[j])) j = m;
      groups[j].push(x);
    }
    const n = groups.map((g, i) => (g.length ? g.reduce((a, b) => a + b) / g.length : c[i]));
    if (n.every((v, i) => v === c[i])) break;
    c = n;
  }
  return c.map(Math.exp).sort((a, b) => a - b);
}

// 0 = same letter, 1 = next letter, 2 = next word (see classify_gaps in morse.py).
// learn: cluster the pauses when even the shortest group is over 1.6 units,
// i.e. a person pausing far longer than textbook Morse.
export function classifyGaps(gaps, unit, learn = true) {
  const standard = gaps.map((g) => (g >= WORD * unit ? 2 : g >= LETTER * unit ? 1 : 0));
  const pos = gaps.filter((g) => g > 0);
  if (!learn || pos.length < 4) return standard;
  for (const k of [3, 2]) {
    if (pos.length < k + 2) continue;
    const c = kmeansLog(pos, k);
    if (c[0] < 1.6 * unit) return standard;
    if (c.slice(1).every((b, i) => b / c[i] >= 1.8)) {
      const cuts = c.slice(1).map((b, i) => Math.sqrt(c[i] * b));
      const [lo, hi] = k === 3 ? cuts : c[0] < 3 * unit ? [cuts[0], Infinity] : [0, cuts[0]];
      return gaps.map((g) => (g >= hi ? 2 : g >= lo ? 1 : 0));
    }
  }
  return standard;
}

// marks + gap kinds -> "... --- ..." (letters split by space, words by " / ")
export function assemble(marks, kinds) {
  if (!marks.length) return "";
  let morse = marks[0];
  for (let i = 1; i < marks.length; i++) morse += (kinds[i - 1] === 2 ? " / " : kinds[i - 1] === 1 ? " " : "") + marks[i];
  return morse;
}

// pulses: [{start, end, confidence}] in seconds. unit: fixed dot length, or null to learn it.
// breaks: times where the recording is interrupted (a cut in an edited video); a pause that
// contains one has unknown length, so it is not used for learning and always ends the word.
export function decodePulses(pulses, defaultSplit = 0.3, unit = null, learnGaps = true, breaks = []) {
  pulses = [...pulses].sort((a, b) => a.start - b.start);
  if (!pulses.length) return { morse: "", text: "", marks: [], kinds: [], unit: unit ?? defaultSplit / DOT_DASH, unknown: 0 };
  const [split, u] = unit ? [DOT_DASH * unit, unit] : markSplit(pulses.map((p) => p.end - p.start), defaultSplit);
  const marks = pulses.map((p) => (p.end - p.start < split ? "." : "-"));
  const broken = pulses.slice(1).map((p, i) => breaks.some((b) => pulses[i].end <= b && b <= p.start));
  const learned = classifyGaps(pulses.slice(1).map((p, i) => p.start - pulses[i].end).filter((_, i) => !broken[i]), u, learnGaps);
  const kinds = broken.map((x) => (x ? 2 : learned.shift()));
  const morse = assemble(marks, kinds);
  const { text, unknown } = decodeMorse(morse);
  return { morse, text, marks, kinds, unit: u, unknown };
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
