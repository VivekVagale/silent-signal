"""Light flashes (a torch, a phone flashlight, a screen) with plain OpenCV.

Idea: measure the brightness of the brightest spot in the frame, then decide
ON/OFF with a threshold that adapts to the room. A light blur followed by the
maximum finds a small torch (the first version used the 99th-percentile pixel
and missed torches covering less than 1% of the frame); the blur stops a
single hot pixel of sensor noise from counting as a flash.

The threshold sits halfway between a slowly-updated "dark level" and "bright
level", with hysteresis (separate switch-on and switch-off levels) so the
signal does not chatter when brightness hovers near the threshold.
"""

import cv2
import numpy as np

from .base import Detector, FrameReading, register


@register
class FlashDetector(Detector):
    name = "flash"
    default_split = 0.35
    min_on = 0.05
    min_off = 0.05

    def __init__(self, min_contrast: float = 40.0):
        self.dark: float | None = None
        self.bright: float | None = None
        self.on = False
        self.min_contrast = min_contrast   # brightness levels (0-255) needed to call it a flash

    def read(self, frame_bgr: np.ndarray, t: float) -> FrameReading:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (160, 120), interpolation=cv2.INTER_AREA)   # speed, and averages out noise
        v = float(cv2.GaussianBlur(gray, (5, 5), 0).max())

        if self.dark is None:
            self.dark = self.bright = v
        # track the dark and bright levels: fast towards new extremes, slow drift otherwise
        self.dark = v if v < self.dark else 0.98 * self.dark + 0.02 * v
        self.bright = v if v > self.bright else 0.995 * self.bright + 0.005 * v

        span = self.bright - self.dark
        if span < self.min_contrast:
            self.on = False
            return FrameReading(False, 0.0, v)
        on_level, off_level = self.dark + 0.6 * span, self.dark + 0.4 * span
        self.on = v >= on_level if not self.on else v > off_level
        # confidence: how far from the middle of the band, 0 at the threshold, 1 at an extreme
        conf = min(1.0, abs(v - (self.dark + 0.5 * span)) / (0.5 * span))
        return FrameReading(self.on, conf, v)
