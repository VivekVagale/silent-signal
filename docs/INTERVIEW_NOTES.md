# Interview notes: Silent Signal

## The core idea

Every channel (eyes, fingers, light) is reduced to the same thing: a stream
of ON/OFF readings with a confidence. Morse decoding never knows where the
signal came from. That separation is the whole architecture: detectors are
swappable, the decoder is tested once.

## Decisions

**MediaPipe blendshapes for blinks, not the eye aspect ratio (EAR).** EAR is
the classic approach: distances between 6 eye landmarks. It depends on head
angle and eye shape and needs per-person tuning. FaceLandmarker's
`eyeBlinkLeft/Right` blendshape is a learned 0-1 closure score that already
handles pose. Using `min(left, right)` means a wink does not count.

**Pinch as the "press".** A finger tap on a table is hard to see from a
webcam (depth is ambiguous). Fingertip-to-thumb contact is visible in 2-D
and gives a clean ON/OFF. The distance is divided by hand size (wrist to
middle knuckle), so it works near or far from the camera.

**Flash: brightest spot, not average brightness.** A torch covers a tiny
part of the frame; the average barely moves. My first version used the
99th-percentile pixel and the benchmark scored **0 of 40 clips**: the torch
covered only 0.3% of the frame, so the top 1% of pixels was still mostly
background. Switching to blur-then-max fixed it: 38 of 40 exact. The blur
stops one hot pixel of sensor noise from counting as a flash.

**Adaptive dark/bright levels.** The threshold sits halfway between a
slowly updated dark level and bright level, so a room getting brighter does
not read as a permanent flash.

**Hysteresis everywhere.** One threshold makes a value hovering near it
flicker ON/OFF many times; two thresholds (ON at 0.5, OFF at 0.35) give one
clean pulse.

**Cleaning pulses.** Drop ON blips shorter than a real signal (detector
noise); merge two pulses split by a one-frame dropout.

## The timing problem (best talking point)

Morse is defined in units: dot 1, dash 3; gaps 1 / 3 / 7. Real people do
not know their unit and are not consistent.

1. **Learn the unit per sender.** Cluster ON durations into two groups
   (dots, dashes) with 1-D k-means; the unit follows from both centres.
2. **Work on a log scale.** First version: arithmetic k-means and midpoints
   (2 units between dot and dash). At 30% timing wobble it had CER 0.39.
   Human timing errors are proportional (aim for 0.3 s, get 0.2-0.45 s; aim
   for 0.9 s, get 0.6-1.3 s), so the right boundary between 1 and 3 is the
   geometric midpoint √3 = 1.73, and clustering belongs on log durations.
   Result at 15% wobble: CER 0.022 -> 0.005.
3. **Baseline for honesty.** Same thresholds, one fixed speed for everyone:
   CER 0.55 at 15% wobble vs 0.005 learned. Learning the speed is the
   feature that matters most.

**Remaining limit.** At 30%+ wobble, individual decisions flip often enough
that errors compound over a message. Next step: decode with a dictionary /
language model, choosing the most likely words given the uncertain marks
(beam search), the same trick phone keyboards use.

## Why a browser app

Target users (patients, carers) will not install Python. MediaPipe ships a
WebAssembly build, so the same models run in any modern browser with no
server, and video never leaves the device, which matters for a medical-ish
tool. The JavaScript engine mirrors the Python one line by line; the Python
side exists for batch analysis, the benchmark and tests.

## Honest limits

- Blink and press detection were checked by hand, not measured on a labeled
  dataset. Next: record 20 people signalling known phrases and report CER per
  channel.
- Natural blinks produce dots, hence the arm/pause switch. A smarter fix:
  require a start sequence, or ignore blinks shorter than ~150 ms.
- Gestures are 7 built-in MediaPipe shapes mapped to words, not sign
  language.
