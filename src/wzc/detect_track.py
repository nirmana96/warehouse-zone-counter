"""YOLO26 detection + BoT-SORT tracking, wrapped so the rest of the code never sees Ultralytics."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from wzc.config import Detector

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Track:
    track_id: int
    xyxy: tuple[float, float, float, float]
    conf: float


class Tracker:
    """Runs detection and tracking one frame at a time.

    Track IDs are anonymous integers from the tracker. Appearance (ReID) features live
    inside Ultralytics' memory only; this class never reads or stores them.
    """

    def __init__(self, cfg: Detector, classes: list[str]) -> None:
        from ultralytics import YOLO  # heavy import, kept out of module load so tests stay fast

        if not cfg.tracker.is_file():
            raise FileNotFoundError(f"tracker config not found: {cfg.tracker}")
        self._model = YOLO(cfg.model)
        name_to_id = {name: idx for idx, name in self._model.names.items()}
        missing = [c for c in classes if c not in name_to_id]
        if missing:
            raise ValueError(f"model {cfg.model} has no class {missing}")
        self._class_ids = [name_to_id[c] for c in classes]
        self._cfg = cfg
        log.info("tracker ready: model=%s tracker=%s classes=%s", cfg.model, cfg.tracker, classes)

    def update(self, frame: np.ndarray) -> list[Track]:
        results = self._model.track(
            frame,
            persist=True,  # keep track state between calls
            tracker=str(self._cfg.tracker),
            classes=self._class_ids,
            conf=self._cfg.conf,
            imgsz=self._cfg.imgsz,
            device=self._cfg.device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []
        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        return [
            Track(track_id=i, xyxy=(b[0], b[1], b[2], b[3]), conf=c)
            for i, b, c in zip(ids, xyxy, confs, strict=True)
        ]
