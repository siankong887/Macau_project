from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import os

import numpy as np
from oa_sort_core import OASortConfig, OASortTracker
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace


DEFAULT_TRACKER_BACKEND = "bytetrack"


@dataclass(frozen=True, slots=True)
class TrackedObject:
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int
    conf: float
    cls: int


@dataclass(frozen=True, slots=True)
class ByteTrackConfig:
    track_high_thresh: float = 0.4
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.4
    track_buffer: int = 30
    match_thresh: float = 0.8
    fuse_score: bool = True

    def to_namespace(self) -> IterableSimpleNamespace:
        return IterableSimpleNamespace(**asdict(self))


class TrackerBackend(ABC):
    name: str

    @abstractmethod
    def update_frame(self, det_np: np.ndarray) -> list[TrackedObject]:
        """Update tracker state with detections from one frame."""

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state after a recoverable failure."""


class _DetResult:
    """Minimal results adapter required by Ultralytics BYTETracker.update()."""

    __slots__ = ("conf", "xywh", "cls")

    def __init__(self, conf: np.ndarray, xywh: np.ndarray, cls: np.ndarray):
        self.conf = np.atleast_1d(np.asarray(conf, dtype=np.float32))
        self.xywh = np.asarray(xywh, dtype=np.float32)
        self.cls = np.atleast_1d(np.asarray(cls, dtype=np.float32))

        if self.xywh.ndim == 1:
            self.xywh = self.xywh[None, :]

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_DetResult":
        conf = self.conf[idx]
        xywh = self.xywh[idx]
        cls = self.cls[idx]

        if np.isscalar(conf):
            conf = np.asarray([conf], dtype=np.float32)
            xywh = np.asarray([xywh], dtype=np.float32)
            cls = np.asarray([cls], dtype=np.float32)

        return _DetResult(conf=conf, xywh=xywh, cls=cls)


class ByteTrackBackend(TrackerBackend):
    name = "bytetrack"

    def __init__(self, frame_rate: int = 30, config: ByteTrackConfig | None = None):
        self.frame_rate = int(frame_rate)
        self.config = config or ByteTrackConfig()
        self._tracker = BYTETracker(args=self.config.to_namespace(), frame_rate=self.frame_rate)

    def update_frame(self, det_np: np.ndarray) -> list[TrackedObject]:
        det_np = np.asarray(det_np, dtype=np.float32)
        if det_np.size == 0:
            det_np = np.empty((0, 6), dtype=np.float32)
        elif det_np.ndim == 1:
            det_np = det_np[None, :]

        xywh = _xyxy_to_xywh(det_np)
        wrapped = _DetResult(conf=det_np[:, 4], xywh=xywh, cls=det_np[:, 5])
        tracks = self._tracker.update(wrapped, None)
        return [_track_row_to_object(track_row) for track_row in tracks]

    def reset(self) -> None:
        self._tracker = BYTETracker(args=self.config.to_namespace(), frame_rate=self.frame_rate)


class OASortBackend(TrackerBackend):
    name = "oasort"

    def __init__(self, frame_rate: int = 30, config: OASortConfig | None = None):
        self.frame_rate = int(frame_rate)
        self.config = config or OASortConfig()
        self._tracker = OASortTracker(config=self.config)

    def update_frame(self, det_np: np.ndarray) -> list[TrackedObject]:
        det_np = np.asarray(det_np, dtype=np.float32)
        if det_np.size == 0:
            det_np = np.empty((0, 6), dtype=np.float32)
        elif det_np.ndim == 1:
            det_np = det_np[None, :]

        tracks = self._tracker.update(det_np)
        return [_oa_track_output_to_object(track_row) for track_row in tracks]

    def reset(self) -> None:
        self._tracker = OASortTracker(config=self.config)


def _xyxy_to_xywh(det_np: np.ndarray) -> np.ndarray:
    if det_np.size == 0:
        return np.empty((0, 4), dtype=np.float32)

    return np.column_stack(
        [
            (det_np[:, 0] + det_np[:, 2]) / 2,
            (det_np[:, 1] + det_np[:, 3]) / 2,
            det_np[:, 2] - det_np[:, 0],
            det_np[:, 3] - det_np[:, 1],
        ]
    ).astype(np.float32, copy=False)


def _track_row_to_object(track_row: np.ndarray) -> TrackedObject:
    return TrackedObject(
        x1=float(track_row[0]),
        y1=float(track_row[1]),
        x2=float(track_row[2]),
        y2=float(track_row[3]),
        track_id=int(track_row[4]),
        conf=float(track_row[5]),
        cls=int(track_row[6]),
    )


def _oa_track_output_to_object(track_row) -> TrackedObject:
    return TrackedObject(
        x1=float(track_row.x1),
        y1=float(track_row.y1),
        x2=float(track_row.x2),
        y2=float(track_row.y2),
        track_id=int(track_row.track_id),
        conf=float(track_row.conf),
        cls=int(track_row.cls),
    )


_TRACKER_BACKENDS: dict[str, type[TrackerBackend]] = {
    ByteTrackBackend.name: ByteTrackBackend,
    OASortBackend.name: OASortBackend,
}


def normalize_tracker_backend_name(backend_name: str | None = None) -> str:
    raw_name = backend_name if backend_name is not None else os.getenv("TRACKER_BACKEND", DEFAULT_TRACKER_BACKEND)
    normalized = (raw_name or DEFAULT_TRACKER_BACKEND).strip().lower()
    if normalized not in _TRACKER_BACKENDS:
        available = ", ".join(sorted(_TRACKER_BACKENDS))
        raise ValueError(f"Unknown TRACKER_BACKEND '{raw_name}'. Available backends: {available}")
    return normalized


def get_configured_tracker_backend_name() -> str:
    return normalize_tracker_backend_name()


def build_tracker_backend(backend_name: str, frame_rate: int) -> TrackerBackend:
    normalized = normalize_tracker_backend_name(backend_name)
    backend_cls = _TRACKER_BACKENDS[normalized]
    return backend_cls(frame_rate=frame_rate)
