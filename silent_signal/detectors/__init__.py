from .base import REGISTRY, Detector, FrameReading, register
from .flash import FlashDetector


def get(name: str) -> Detector:
    """Create a detector by name. MediaPipe ones are imported only when asked for."""
    if name in ("blink", "tap", "sign"):
        from . import landmarks  # noqa: F401  (registers BlinkDetector, TapDetector, SignDetector)
    if name not in REGISTRY:
        raise KeyError(f"unknown channel {name!r}; available: flash, blink, tap, sign")
    return REGISTRY[name]()


__all__ = ["Detector", "FrameReading", "FlashDetector", "REGISTRY", "get", "register"]
