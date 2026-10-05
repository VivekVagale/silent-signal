"""Body + both hands per frame, for word-level signs (Indian Sign Language).

A word sign is a movement: where the hands go relative to the body, and the
shape of each hand along the way. Per frame we keep
  - 7 upper-body points from MediaPipe PoseLandmarker (nose, shoulders, elbows, wrists),
  - 21 points for each hand from HandLandmarker, assigned to the body's left or
    right side by the nearest pose wrist (MediaPipe's own left/right label flips),
all as image coordinates (0-1). Features are made later (see words.py).

The signer can be small in the picture (a full-body recording), so the hand model
runs on a crop around the upper body, found from the pose: hands of 20 pixels are
missed, the same hands in a crop are found.
"""

import cv2
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .detectors.landmarks import model_path, to_mp_image

POSE_IDS = [0, 11, 12, 13, 14, 15, 16]     # nose, shoulders L/R, elbows L/R, wrists L/R
L_WRIST, R_WRIST = 5, 6                    # their positions inside POSE_IDS


class Holistic:
    def __init__(self):
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path("pose_landmarker_lite.task")),
            running_mode=vision.RunningMode.VIDEO, num_poses=1))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path("hand_landmarker.task")),
            running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4))
        self.ts = -1

    def read(self, frame_bgr: np.ndarray, t: float):
        """-> pose (7, 3) x, y, visibility or None; hands (2, 21, 3) with NaN where a hand is missing."""
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (640, round(640 * h / w)))
        self.ts = max(int(t * 1000), self.ts + 1)
        res = self.pose.detect_for_video(to_mp_image(small), self.ts)
        hands = np.full((2, 21, 3), np.nan, np.float32)
        if not res.pose_landmarks:
            return None, hands
        lm = res.pose_landmarks[0]
        pose = np.array([[lm[i].x, lm[i].y, lm[i].visibility] for i in POSE_IDS], np.float32)
        x0, y0, x1, y1 = body_crop(pose, w, h)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return pose, hands
        hr = self.hands.detect(to_mp_image(crop))
        for pts in hr.hand_landmarks:
            a = np.array([[(x0 + q.x * (x1 - x0)) / w, (y0 + q.y * (y1 - y0)) / h, q.z] for q in pts], np.float32)
            d = [np.hypot(*(a[0, :2] - pose[i, :2])) for i in (L_WRIST, R_WRIST)]
            side = int(np.argmin(d))
            if np.isnan(hands[side, 0, 0]):
                hands[side] = a
            else:
                hands[1 - side] = a
        return pose, hands

    def close(self):
        self.pose.close()
        self.hands.close()


def body_crop(pose: np.ndarray, w: int, h: int):
    """Square around the upper body, 4 shoulder-widths wide, in pixels, clipped to the frame."""
    ls, rs = pose[1, :2] * (w, h), pose[2, :2] * (w, h)
    sw = max(np.hypot(*(ls - rs)), 0.08 * w)
    cx, cy = (ls + rs) / 2
    half = 2.0 * sw
    x0, x1 = int(max(0, cx - half)), int(min(w, cx + half))
    y0, y1 = int(max(0, cy - 1.6 * sw)), int(min(h, cy + 2.4 * sw))
    return x0, y0, x1, y1
