"""Blinks and finger presses with MediaPipe's pre-trained landmark models.

Blink: MediaPipe FaceLandmarker outputs 52 "blendshapes", scores between 0 and
1 for facial actions. eyeBlinkLeft / eyeBlinkRight measure eye closure
directly, which is more robust than hand-computing an eye aspect ratio from
landmark distances. Both eyes must close, so a wink or a squint of one eye
does not count.

Finger press: HandLandmarker gives 21 points per hand. A "press" is the index
fingertip touching the thumb tip (a pinch). The distance is divided by the
hand's size (wrist to middle-finger knuckle) so it works near or far from the
camera.

Both use hysteresis: separate thresholds to switch ON and OFF.
"""

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .base import Detector, FrameReading, register

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}


def model_path(name: str) -> str:
    path = MODEL_DIR / name
    if not path.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MODELS[name], path)
    return str(path)


def to_mp_image(frame_bgr: np.ndarray) -> mp.Image:
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


@register
class BlinkDetector(Detector):
    name = "blink"
    default_split = 0.4    # short deliberate blink vs long blink
    min_on = 0.08          # faster than a real blink = landmark noise
    min_off = 0.06

    def __init__(self, close_at: float = 0.5, open_at: float = 0.35):
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path("face_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True,
            # looser than the 0.5 defaults, so old or low-resolution footage still gets tracked
            min_face_detection_confidence=0.3, min_face_presence_confidence=0.3, min_tracking_confidence=0.3)
        self.model = vision.FaceLandmarker.create_from_options(opts)
        self.close_at, self.open_at, self.on, self.pad, self.ts = close_at, open_at, False, False, -1

    def _detect(self, frame, t, pad):
        if pad:                                  # shrink the frame into a black border: close-ups get found
            h, w = frame.shape[:2]
            canvas = np.zeros((h * 2, w * 2, 3), np.uint8)
            canvas[h // 2:h // 2 + h, w // 2:w // 2 + w] = frame
            frame = canvas
        self.ts = max(int(t * 1000), self.ts + 1)    # MediaPipe needs strictly increasing timestamps
        return self.model.detect_for_video(to_mp_image(frame), self.ts)

    def read(self, frame_bgr: np.ndarray, t: float) -> FrameReading:
        res = self._detect(frame_bgr, t, self.pad)
        if not res.face_blendshapes:             # a face filling the frame is often missed: retry the other way
            res = self._detect(frame_bgr, t, not self.pad)
            if res.face_blendshapes:
                self.pad = not self.pad
        if not res.face_blendshapes:
            self.on = False
            return FrameReading(False, 0.0, 0.0, found=False)
        scores = {c.category_name: c.score for c in res.face_blendshapes[0]}
        closure = min(scores["eyeBlinkLeft"], scores["eyeBlinkRight"])   # both eyes
        self.on = closure >= self.close_at if not self.on else closure > self.open_at
        conf = min(1.0, abs(closure - 0.425) / 0.425)
        return FrameReading(self.on, conf, closure)

    def close(self) -> None:
        self.model.close()


@register
class TapDetector(Detector):
    name = "tap"
    default_split = 0.35
    min_on = 0.06
    min_off = 0.05

    def __init__(self, touch_at: float = 0.25, release_at: float = 0.35):
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO, num_hands=1)
        self.model = vision.HandLandmarker.create_from_options(opts)
        self.touch_at, self.release_at, self.on = touch_at, release_at, False

    def read(self, frame_bgr: np.ndarray, t: float) -> FrameReading:
        res = self.model.detect_for_video(to_mp_image(frame_bgr), int(t * 1000))
        if not res.hand_landmarks:
            self.on = False
            return FrameReading(False, 0.0, 1.0)
        lm = res.hand_landmarks[0]
        p = lambda i: np.array([lm[i].x, lm[i].y])
        hand_size = np.linalg.norm(p(0) - p(9)) + 1e-6      # wrist -> middle finger knuckle
        pinch = float(np.linalg.norm(p(4) - p(8)) / hand_size)   # thumb tip <-> index tip
        self.on = pinch <= self.touch_at if not self.on else pinch < self.release_at
        conf = min(1.0, abs(pinch - 0.3) / 0.3)
        return FrameReading(self.on, conf, pinch)

    def close(self) -> None:
        self.model.close()
