"""Frame reader for a video file or camera stream."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


class VideoReader:
    """Reads frames in order and reports fps and frame size. Use as a context manager."""

    def __init__(self, source: str | Path) -> None:
        self._source = str(source)
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open video: {self._source}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self.frame_size = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        if self.fps <= 0:
            raise RuntimeError(f"video reports no fps: {self._source}")
        log.info("opened %s: %dx%d at %.2f fps", self._source, *self.frame_size, self.fps)

    def frames(self, max_frames: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
        idx = 0
        while max_frames is None or idx < max_frames:
            ok, frame = self._cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1

    def read_frame(self, index: int) -> np.ndarray:
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"could not read frame {index} of {self._source}")
        return frame

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
