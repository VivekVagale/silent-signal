"""The contract every signal channel follows.

A detector looks at one video frame at a time and says whether its signal is
ON, with a confidence. That is all. Timing, Morse and text are shared code, so
adding a new channel (head nods, a tongue click on audio, ...) means writing
one small class and registering it; nothing else changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameReading:
    on: bool
    confidence: float     # 0-1, how sure the detector is about this frame
    value: float          # the raw measurement (eye closure, pinch distance, brightness), for plots
    found: bool = True    # False when the detector saw nothing to measure (no face, no hand)


class Detector(ABC):
    name: str = "base"
    default_split: float = 0.3   # seconds: dot/dash boundary before enough data to learn one
    min_on: float = 0.05         # ignore ON blips shorter than this
    min_off: float = 0.04        # merge pulses split by a shorter OFF dropout

    @abstractmethod
    def read(self, frame_bgr: np.ndarray, t: float) -> FrameReading:
        """Look at one frame (BGR, as OpenCV gives it) at time t seconds."""

    def close(self) -> None:
        pass


REGISTRY: dict[str, type[Detector]] = {}


def register(cls: type[Detector]) -> type[Detector]:
    REGISTRY[cls.name] = cls
    return cls
