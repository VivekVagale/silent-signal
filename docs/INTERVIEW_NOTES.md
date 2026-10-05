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

## Real footage broke my assumptions (second best talking point)

I tested the video analyzer on the 1966 clip of Jeremiah Denton blinking
TORTURE as a prisoner of war. Each failure taught something:

1. **Uploading a second video did nothing.** MediaPipe's VIDEO mode needs
   strictly increasing timestamps; a new clip restarted at 0. Fix: keep a
   counter that only goes up.
2. **Most frames were never read.** Playing the clip and reading frames as
   they arrived got 97 of 616 on a slow laptop. Fix: seek frame by frame and
   wait (`requestVideoFrameCallback`) until each frame is really shown.
3. **The eyes were not found in close-ups.** The face detector is tuned for
   faces that do not fill the frame. Fix: retry those frames shrunk into a
   black border (50% -> 100% of frames on a close-up test clip, which is an
   AI-generated video, not real Morse).
4. **Fixed thresholds failed.** In grainy 320x240 film an open eye already
   scores 0.44. Fix: calibrate per video (open = 35th percentile, closed =
   99th) with a sensitivity slider.
5. **Deliberate blinkers pause far longer than Morse says.** Denton paused
   0.5-1.4 s between letters (Morse: about 3 dot lengths). Fix: learn the pause groups with 3-cluster
   k-means on log pauses, used only when even the shortest pauses are much
   longer than a unit. A bug here: starting the clusters at quantiles put two
   centres on the common in-letter pause, so a lone word gap got no cluster
   ("SOSHELP"). Starting them evenly across the range fixed it.
6. **News clips are edited.** The "5.3 s dash" at the end was really 4 s of
   cutaway with the last eye state carried over. Fix: detect cuts (a big
   jump in the picture), drop signals cut off by an edit or by a face lost
   for over 0.25 s, and end the word at every cut.

Where it stands: 15 marks found (truth 15, was 18), T, O and T read
correctly, but not the whole word. His pause after T is as short as the
pauses inside O, and at 15 fps his short and long blinks for R and U
overlap. A person reading it knows the answer; the timing alone does not
say it. That is why the app has a review mode: replay, flip, delete or add
marks. I kept a strict `xfail` test for "TORTUR" so a real fix is noticed,
rather than tuning thresholds until this one clip passes.

## Sign language: letters from hand landmarks

The sign channel reads ASL fingerspelling. It replaced an earlier "gesture"
channel that only mapped MediaPipe's 7 built-in hand shapes to words.

- **Landmarks, not pixels.** MediaPipe HandLandmarker already finds 21 hand
  points; a letter is a hand shape, so those points are the input. That needs
  far less data than an image model (2,122 frames were enough), works in any
  lighting MediaPipe handles, and runs in the browser in microseconds.
- **Features.** Points relative to the wrist, divided by palm length (wrist
  to middle knuckle): position and distance from the camera drop out, shape
  and orientation stay. Orientation must stay: H and U, or K and P, are the
  same shape pointing different ways.
- **Model.** 63 -> 64 -> 26 network, ReLU, softmax, Adam, dropout, weight
  decay, written in numpy so the same weights (a 53 KB JSON) run in Python and
  in JavaScript; a test checks both give the same answer.
- **Augmentation was the biggest gain.** Mirror (the other hand signs the
  mirror image of the same letter), rotate ±20°, stretch x against y
  (cameras differ in aspect ratio), jitter. Accuracy 90.2% -> 96.6%; on
  mirrored frames (the other hand) 72.6% -> 96.0%.
- **Honest evaluation.** The data has no signer id, so 5-fold
  cross-validation mixes the same people across folds: 96.6% is an upper
  bound for a stranger. The real test is new signers, which I do not have.
- **Typing rule.** A letter is typed when held confidently for 0.4 s; the
  same letter again needs a change in between (a relaxed hand); a short
  unsure blur (under 0.15 s) does not break a hold; no hand for 1 s is a
  space. A letter is typed only on a confident frame: otherwise an unsure
  frame inside the grace period could type it.
- **The sample clip is synthetic**: a MakeHuman character posed letter by
  letter in Blender. Each shape was checked against the real pipeline; A and
  U never read correctly on the CG hand (U with fingers together still reads
  as V), so the phrase "BE BOLD" avoids them. Compression mattered: O scored
  0.93-0.99 on the rendered frames, 0.59-0.80 after a strong H.264
  compression, so the clip is encoded at higher quality.
- **Why not ISL?** Indian Sign Language fingerspelling mostly uses two hands
  and public landmark data is scarce. The channel interface takes it as soon
  as there is data: a new label set and a two-hand feature vector.

## Sign language, part 2: whole words (Indian Sign Language)

Fingerspelling is letters; Deaf people mostly sign whole words with both
hands, movement and position on the body. The ISL words channel is a
prototype for 17 words from INCLUDE (Deaf students, Chennai, CC BY 4.0).

- **What it reads per frame**: 7 body points (PoseLandmarker) and both hands
  (HandLandmarker), relative to the shoulder centre, scaled by shoulder
  width. For each hand: where its wrist is on the body, and its shape
  (relative to its own wrist). Left/right hand is decided by the nearest body
  wrist, not MediaPipe's own label, which flips.
- **Small signers**: INCLUDE is filmed full body at 1080p; hands are a few
  dozen pixels and the hand model misses them. Running it on a crop around
  the upper body (found from the pose) fixes that. The browser does the same.
- **Where a sign starts and stops**: a sign is the stretch where a wrist is
  raised. I first set the line at 1.8 shoulder-widths below the shoulders
  from one clip, measured without correcting for the 16:9 frame; resting
  hands actually sit at 1.47-1.70, so hands always counted as raised, the
  model trained on whole clips and live mode never split words. Measuring
  all 358 clips (rest 1.47-1.70, every sign reaches 0.94 or higher) put the
  line at 1.3. Accuracy went from 95.3% to 97.6% and the sample split into
  its three words.
- **Model**: the raised frames resampled to 16, flattened (16 x 104), into
  the same small numpy network as the letters. With about 16 clips per word,
  augmentation matters: time crop, rotation, scale, dropped hand frames, and
  mirrored copies (a left-handed signer: 12% -> 99% on mirrored clips).
- **Honest limits**: 17 words, not 263; same signers in train and test (the
  official split link is dead); one word at a time with hands lowered in
  between; output is the words in order, not English sentences.
- **Why a real sample here but a synthetic one for letters**: a letter is a
  static hand shape I can pose and check. Word signs are movements of a real
  language; animating them myself would be inventing signs. So the ISL
  sample uses real INCLUDE clips (CC BY 4.0, credited on screen) that were
  held out before training.

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
- Sign language covers ASL fingerspelling only, one frame at a time: J and
  Z are moving letters and are seen as single frames; whole-word ISL is a
  17-word prototype (see above). The 96.6% is not measured on new signers.
