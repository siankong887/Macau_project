from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import scipy.optimize

from oa_sort_kalman import OASortKalmanFilter, box_from_state, measurement_from_box
from oa_sort_occlusion import (
    apply_occlusion_aware_offset,
    blend_boxes_xyxy,
    compute_bias_aware_momentum,
    compute_occlusion_coefficients,
    ensure_boxes_xyxy,
    pairwise_iou_xyxy,
)


_MISSING_OBSERVATION = np.asarray([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
_EPS = 1e-6


class TrackState(str, Enum):
    TRACKED = "tracked"
    LOST = "lost"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class OASortConfig:
    track_high_thresh: float = 0.4
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.4
    match_thresh: float = 0.3
    track_buffer: int = 30
    min_hits: int = 1
    delta_t: int = 3
    inertia: float = 0.05
    tcm_first_step: bool = True
    tcm_byte_step: bool = True
    tcm_first_step_weight: float = 1.0
    tcm_byte_step_weight: float = 1.0
    tau: float = 0.15
    occ_trigger_thresh: float = 5.0
    gm_kx: float = 3.0 * math.sqrt(2.0)
    gm_ky: float = 3.0


def _normalize_detection_row(detection: np.ndarray) -> np.ndarray:
    arr = np.asarray(detection, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] < 6:
        raise ValueError(f"Expected one detection row [x1,y1,x2,y2,conf,cls], got shape {arr.shape}")
    return arr[:6].astype(np.float32, copy=False)


def _normalize_observation(observation: np.ndarray) -> np.ndarray:
    arr = np.asarray(observation, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] < 5:
        raise ValueError(f"Expected one observation row [x1,y1,x2,y2,score], got shape {arr.shape}")
    return arr[:5].astype(np.float32, copy=False)


def _normalized_speed(point_a: tuple[float, float], point_b: tuple[float, float]) -> np.ndarray:
    dy = float(point_b[1] - point_a[1])
    dx = float(point_b[0] - point_a[0])
    norm = float(np.hypot(dy, dx)) + _EPS
    return np.asarray([dy / norm, dx / norm], dtype=np.float32)


def _corner_velocities(previous_box: np.ndarray, current_box: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prev = _normalize_observation(previous_box)
    curr = _normalize_observation(current_box)
    prev_lt = (float(prev[0]), float(prev[1]))
    prev_rt = (float(prev[0]), float(prev[3]))
    prev_lb = (float(prev[2]), float(prev[1]))
    prev_rb = (float(prev[2]), float(prev[3]))

    curr_lt = (float(curr[0]), float(curr[1]))
    curr_rt = (float(curr[0]), float(curr[3]))
    curr_lb = (float(curr[2]), float(curr[1]))
    curr_rb = (float(curr[2]), float(curr[3]))

    return (
        _normalized_speed(prev_lt, curr_lt),
        _normalized_speed(prev_rt, curr_rt),
        _normalized_speed(prev_lb, curr_lb),
        _normalized_speed(prev_rb, curr_rb),
    )


def k_previous_observation(observations: dict[int, np.ndarray], current_age: int, delta_t: int) -> np.ndarray:
    if not observations:
        return _MISSING_OBSERVATION.copy()

    for i in range(int(delta_t)):
        age_key = current_age - (int(delta_t) - i)
        if age_key in observations:
            return observations[age_key].copy()

    return observations[max(observations)].copy()


@dataclass(slots=True)
class TrackOutputRow:
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int
    conf: float
    cls: int


class OATrack:
    _next_track_id = 1

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._next_track_id = 1

    @classmethod
    def next_id(cls) -> int:
        track_id = cls._next_track_id
        cls._next_track_id += 1
        return track_id

    def __init__(
        self,
        detection: np.ndarray,
        frame_id: int,
        kalman_filter: OASortKalmanFilter,
        delta_t: int = 3,
    ):
        det = _normalize_detection_row(detection)
        self.track_id = self.next_id()
        self.kalman_filter = kalman_filter
        self.delta_t = int(delta_t)

        self.mean, self.covariance = self.kalman_filter.initiate(measurement_from_box(det[:5]))
        self.state = TrackState.TRACKED

        self.cls = int(det[5])
        self.frame_id = int(frame_id)
        self.start_frame = int(frame_id)

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

        self.latest_score = float(det[4])
        self.previous_score: float | None = None
        self.last_occlusion_coefficient = 0.0

        observation = det[:5].copy()
        self.last_observation = observation
        self.last_observation_save = observation.copy()
        self.observations: dict[int, np.ndarray] = {0: observation.copy()}
        self.history_observations: list[np.ndarray] = [observation.copy()]

        self.velocity_lt: np.ndarray | None = None
        self.velocity_rt: np.ndarray | None = None
        self.velocity_lb: np.ndarray | None = None
        self.velocity_rb: np.ndarray | None = None

    def predict(self) -> np.ndarray:
        if (self.mean[2] + self.mean[7]) <= 0.0:
            self.mean[7] = 0.0

        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.predicted_observation()

    def _update_velocity_state(self, observation: np.ndarray) -> None:
        if self.last_observation.sum() < 0:
            return

        previous_box = None
        agg_lt = agg_rt = agg_lb = agg_rb = None
        for i in range(self.delta_t):
            age_key = self.age - i - 1
            if age_key not in self.observations:
                continue
            candidate = self.observations[age_key]
            previous_box = candidate
            vel_lt, vel_rt, vel_lb, vel_rb = _corner_velocities(candidate, observation)
            if agg_lt is None:
                agg_lt, agg_rt, agg_lb, agg_rb = vel_lt, vel_rt, vel_lb, vel_rb
            else:
                agg_lt += vel_lt
                agg_rt += vel_rt
                agg_lb += vel_lb
                agg_rb += vel_rb

        if previous_box is None:
            previous_box = self.last_observation
            agg_lt, agg_rt, agg_lb, agg_rb = _corner_velocities(previous_box, observation)

        self.velocity_lt = agg_lt
        self.velocity_rt = agg_rt
        self.velocity_lb = agg_lb
        self.velocity_rb = agg_rb

    def update_with_detection(
        self,
        detection: np.ndarray,
        frame_id: int,
        observation_box: np.ndarray | None = None,
        occlusion_coefficient: float | None = None,
    ) -> None:
        det = _normalize_detection_row(detection)
        if observation_box is None:
            observation_xyxy = det[:4]
        else:
            observation_xyxy = ensure_boxes_xyxy(observation_box)[0]

        observation = np.asarray([*observation_xyxy.tolist(), float(det[4])], dtype=np.float32)
        self._update_velocity_state(observation)

        self.last_observation = observation
        self.last_observation_save = observation.copy()
        self.observations[self.age] = observation.copy()
        self.history_observations.append(observation.copy())

        measurement_box = np.asarray([*observation_xyxy.tolist(), float(det[4])], dtype=np.float32)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean,
            self.covariance,
            measurement_from_box(measurement_box),
        )

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.state = TrackState.TRACKED
        self.frame_id = int(frame_id)

        self.previous_score = self.latest_score
        self.latest_score = float(det[4])
        self.cls = int(det[5])
        if occlusion_coefficient is not None:
            self.last_occlusion_coefficient = float(occlusion_coefficient)

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED

    def set_last_occlusion_coefficient(self, coefficient: float) -> None:
        self.last_occlusion_coefficient = float(coefficient)

    def predicted_observation(self) -> np.ndarray:
        return box_from_state(self.mean).astype(np.float32, copy=False)

    def predicted_box_xyxy(self) -> np.ndarray:
        return self.predicted_observation()[:4].copy()

    def output_box_xyxy(self) -> np.ndarray:
        if self.last_observation.sum() >= 0:
            return self.last_observation[:4].copy()
        return self.predicted_box_xyxy()

    def latest_observation_or_missing(self) -> np.ndarray:
        return self.last_observation.copy()

    def previous_observation(self) -> np.ndarray:
        return k_previous_observation(self.observations, self.age, self.delta_t)

    def association_scores(self, track_thresh: float) -> tuple[float, float]:
        predicted = self.predicted_observation()
        predicted_score = float(np.clip(predicted[4], float(track_thresh), 1.0))
        if self.previous_score is None:
            simple_score = float(np.clip(self.latest_score, 0.1, float(track_thresh)))
        else:
            simple_score = float(
                np.clip(self.latest_score - (self.previous_score - self.latest_score), 0.1, float(track_thresh))
            )
        return predicted_score, simple_score

    def association_track_vector(self, track_thresh: float) -> np.ndarray:
        predicted = self.predicted_box_xyxy()
        predicted_score, simple_score = self.association_scores(track_thresh)
        return np.asarray([predicted[0], predicted[1], predicted[2], predicted[3], predicted_score, simple_score])

    def track_length(self) -> int:
        return int(self.frame_id - self.start_frame)

    def ready_for_output(self, min_hits: int) -> bool:
        return self.state == TrackState.TRACKED and self.time_since_update == 0 and self.hit_streak >= int(min_hits)

    def to_output_row(self) -> TrackOutputRow:
        box = self.output_box_xyxy()
        return TrackOutputRow(
            x1=float(box[0]),
            y1=float(box[1]),
            x2=float(box[2]),
            y2=float(box[3]),
            track_id=int(self.track_id),
            conf=float(self.latest_score),
            cls=int(self.cls),
        )


def joint_tracks(primary: list[OATrack], secondary: list[OATrack]) -> list[OATrack]:
    seen: dict[int, OATrack] = {}
    result: list[OATrack] = []

    for track in primary + secondary:
        if track.track_id in seen:
            continue
        seen[track.track_id] = track
        result.append(track)
    return result


def subtract_tracks(primary: list[OATrack], secondary: list[OATrack]) -> list[OATrack]:
    secondary_ids = {track.track_id for track in secondary}
    return [track for track in primary if track.track_id not in secondary_ids]


def remove_duplicate_tracks(
    tracked: list[OATrack],
    lost: list[OATrack],
    iou_threshold: float = 0.85,
) -> tuple[list[OATrack], list[OATrack]]:
    if not tracked or not lost:
        return tracked, lost

    tracked_boxes = np.asarray([track.output_box_xyxy() for track in tracked], dtype=np.float32)
    lost_boxes = np.asarray([track.output_box_xyxy() for track in lost], dtype=np.float32)
    iou = pairwise_iou_xyxy(tracked_boxes, lost_boxes)
    duplicate_pairs = np.argwhere(iou > float(iou_threshold))

    tracked_drop: set[int] = set()
    lost_drop: set[int] = set()
    for tracked_idx, lost_idx in duplicate_pairs:
        tracked_len = tracked[tracked_idx].track_length()
        lost_len = lost[lost_idx].track_length()
        if tracked_len > lost_len:
            lost_drop.add(int(lost_idx))
        else:
            tracked_drop.add(int(tracked_idx))

    kept_tracked = [track for idx, track in enumerate(tracked) if idx not in tracked_drop]
    kept_lost = [track for idx, track in enumerate(lost) if idx not in lost_drop]
    return kept_tracked, kept_lost


def _hmiou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.empty((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    yy11 = np.maximum(a[..., 1], b[..., 1])
    yy12 = np.minimum(a[..., 3], b[..., 3])
    yy21 = np.minimum(a[..., 1], b[..., 1])
    yy22 = np.maximum(a[..., 3], b[..., 3])

    vertical = np.maximum(0.0, yy12 - yy11) / np.maximum(yy22 - yy21, _EPS)
    return (vertical * pairwise_iou_xyxy(boxes_a, boxes_b)).astype(np.float32, copy=False)


def _score_difference_matrix(detections: np.ndarray, track_vectors: np.ndarray) -> np.ndarray:
    if len(detections) == 0 or len(track_vectors) == 0:
        return np.empty((len(detections), len(track_vectors)), dtype=np.float32)
    det_scores = detections[:, 4][:, None]
    track_scores = track_vectors[:, 5][None, :]
    return np.abs(det_scores - track_scores).astype(np.float32, copy=False)


def _speed_direction_batch(dets: np.ndarray, tracks: np.ndarray, point: str) -> tuple[np.ndarray, np.ndarray]:
    tracks = np.asarray(tracks, dtype=np.float32)[..., np.newaxis]
    dets = np.asarray(dets, dtype=np.float32)

    if point == "lt":
        cx1, cy1 = dets[:, 0], dets[:, 1]
        cx2, cy2 = tracks[:, 0], tracks[:, 1]
    elif point == "rt":
        cx1, cy1 = dets[:, 0], dets[:, 3]
        cx2, cy2 = tracks[:, 0], tracks[:, 3]
    elif point == "lb":
        cx1, cy1 = dets[:, 2], dets[:, 1]
        cx2, cy2 = tracks[:, 2], tracks[:, 1]
    elif point == "rb":
        cx1, cy1 = dets[:, 2], dets[:, 3]
        cx2, cy2 = tracks[:, 2], tracks[:, 3]
    else:
        raise ValueError(f"Unknown corner point: {point}")

    dx = cx1 - cx2
    dy = cy1 - cy2
    norm = np.sqrt(dx * dx + dy * dy) + _EPS
    return (dy / norm).astype(np.float32, copy=False), (dx / norm).astype(np.float32, copy=False)


def _corner_motion_cost(
    detections: np.ndarray,
    previous_obs: np.ndarray,
    velocities: np.ndarray,
    weight: float,
    point: str,
) -> np.ndarray:
    if len(detections) == 0 or len(previous_obs) == 0:
        return np.empty((len(detections), len(previous_obs)), dtype=np.float32)

    y_dir, x_dir = _speed_direction_batch(detections, previous_obs, point)
    inertia_y = velocities[:, 0][:, None]
    inertia_x = velocities[:, 1][:, None]
    diff_angle_cos = np.clip(inertia_x * x_dir + inertia_y * y_dir, -1.0, 1.0)
    diff_angle = (np.pi / 2.0 - np.abs(np.arccos(diff_angle_cos))) / np.pi

    valid_mask = np.ones((previous_obs.shape[0], 1), dtype=np.float32)
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0.0
    scores = detections[:, 4][:, None]
    return ((valid_mask * diff_angle).T * float(weight) * scores).astype(np.float32, copy=False)


def _combined_motion_cost(
    detections: np.ndarray,
    previous_obs: np.ndarray,
    velocities_lt: np.ndarray,
    velocities_rt: np.ndarray,
    velocities_lb: np.ndarray,
    velocities_rb: np.ndarray,
    weight: float,
) -> np.ndarray:
    if len(detections) == 0 or len(previous_obs) == 0:
        return np.empty((len(detections), len(previous_obs)), dtype=np.float32)

    return (
        _corner_motion_cost(detections, previous_obs, velocities_lt, weight, "lt")
        + _corner_motion_cost(detections, previous_obs, velocities_rt, weight, "rt")
        + _corner_motion_cost(detections, previous_obs, velocities_lb, weight, "lb")
        + _corner_motion_cost(detections, previous_obs, velocities_rb, weight, "rb")
    ).astype(np.float32, copy=False)


def _linear_assignment_from_similarity(
    similarity: np.ndarray,
    minimum_similarity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    similarity = np.asarray(similarity, dtype=np.float32)
    num_rows, num_cols = similarity.shape if similarity.ndim == 2 else (0, 0)
    if num_rows == 0 or num_cols == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(num_rows, dtype=int),
            np.arange(num_cols, dtype=int),
        )

    row_idx, col_idx = scipy.optimize.linear_sum_assignment(-similarity)
    matches = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    threshold = float(minimum_similarity)
    for r, c in zip(row_idx.tolist(), col_idx.tolist()):
        if similarity[r, c] < threshold:
            continue
        matches.append([r, c])
        used_rows.add(r)
        used_cols.add(c)

    unmatched_rows = np.asarray([idx for idx in range(num_rows) if idx not in used_rows], dtype=int)
    unmatched_cols = np.asarray([idx for idx in range(num_cols) if idx not in used_cols], dtype=int)
    if matches:
        return np.asarray(matches, dtype=int), unmatched_rows, unmatched_cols
    return np.empty((0, 2), dtype=int), unmatched_rows, unmatched_cols


class OASortTracker:
    def __init__(self, config: OASortConfig | None = None):
        self.config = config or OASortConfig()
        self.kalman_filter = OASortKalmanFilter()
        self.reset()

    def reset(self) -> None:
        self.frame_id = 0
        self.tracks: list[OATrack] = []
        self.removed_tracks: list[OATrack] = []
        OATrack.reset_id_counter()

    def _live_tracks(self) -> list[OATrack]:
        return [track for track in self.tracks if track.state != TrackState.REMOVED]

    def _tracked_tracks(self) -> list[OATrack]:
        return [track for track in self.tracks if track.state == TrackState.TRACKED]

    def _lost_tracks(self) -> list[OATrack]:
        return [track for track in self.tracks if track.state == TrackState.LOST]

    def _predict_live_tracks(self) -> None:
        for track in self._live_tracks():
            track.predict()

    def _build_track_context(self, track_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(track_indices) == 0:
            empty_track = np.empty((0, 6), dtype=np.float32)
            empty_obs = np.empty((0, 5), dtype=np.float32)
            empty_vel = np.empty((0, 2), dtype=np.float32)
            return empty_track, empty_obs, empty_vel, empty_vel, empty_vel, empty_vel

        selected_tracks = [self.tracks[int(idx)] for idx in track_indices]
        track_vectors = np.asarray(
            [track.association_track_vector(self.config.track_high_thresh) for track in selected_tracks],
            dtype=np.float32,
        )
        previous_obs = np.asarray([track.previous_observation() for track in selected_tracks], dtype=np.float32)
        velocities_lt = np.asarray(
            [track.velocity_lt if track.velocity_lt is not None else np.zeros((2,), dtype=np.float32) for track in selected_tracks],
            dtype=np.float32,
        )
        velocities_rt = np.asarray(
            [track.velocity_rt if track.velocity_rt is not None else np.zeros((2,), dtype=np.float32) for track in selected_tracks],
            dtype=np.float32,
        )
        velocities_lb = np.asarray(
            [track.velocity_lb if track.velocity_lb is not None else np.zeros((2,), dtype=np.float32) for track in selected_tracks],
            dtype=np.float32,
        )
        velocities_rb = np.asarray(
            [track.velocity_rb if track.velocity_rb is not None else np.zeros((2,), dtype=np.float32) for track in selected_tracks],
            dtype=np.float32,
        )
        return track_vectors, previous_obs, velocities_lt, velocities_rt, velocities_lb, velocities_rb

    def _first_stage_association(self, detections: np.ndarray, track_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(detections) == 0 or len(track_indices) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(detections), dtype=int),
                np.arange(len(track_indices), dtype=int),
            )

        track_vectors, previous_obs, velocities_lt, velocities_rt, velocities_lb, velocities_rb = self._build_track_context(track_indices)
        det_obs = detections[:, :5]
        pred_boxes = track_vectors[:, :4]

        base_similarity = _hmiou(det_obs, pred_boxes)
        occ_coeff = compute_occlusion_coefficients(
            pred_boxes,
            occ_trigger_thresh=self.config.occ_trigger_thresh,
            gm_kx=self.config.gm_kx,
            gm_ky=self.config.gm_ky,
        )
        similarity = apply_occlusion_aware_offset(base_similarity, occ_coeff, tau=self.config.tau)
        similarity += _combined_motion_cost(
            det_obs,
            previous_obs,
            velocities_lt,
            velocities_rt,
            velocities_lb,
            velocities_rb,
            self.config.inertia,
        )
        if self.config.tcm_first_step:
            similarity -= _score_difference_matrix(det_obs, track_vectors) * float(self.config.tcm_first_step_weight)

        return _linear_assignment_from_similarity(similarity, self.config.match_thresh)

    def _second_stage_association(self, detections: np.ndarray, track_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(detections) == 0 or len(track_indices) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(detections), dtype=int),
                np.arange(len(track_indices), dtype=int),
            )

        track_vectors, _, _, _, _, _ = self._build_track_context(track_indices)
        det_obs = detections[:, :5]
        similarity = _hmiou(det_obs, track_vectors[:, :4])
        if self.config.tcm_byte_step:
            similarity -= _score_difference_matrix(det_obs, track_vectors) * float(self.config.tcm_byte_step_weight)
        return _linear_assignment_from_similarity(similarity, self.config.match_thresh)

    def _third_stage_association(self, detections: np.ndarray, track_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(detections) == 0 or len(track_indices) == 0:
            return (
                np.empty((0, 2), dtype=int),
                np.arange(len(detections), dtype=int),
                np.arange(len(track_indices), dtype=int),
            )

        candidate_tracks = [self.tracks[int(idx)] for idx in track_indices]
        last_boxes = np.asarray([track.output_box_xyxy() for track in candidate_tracks], dtype=np.float32)
        similarity = _hmiou(detections[:, :5], last_boxes)
        return _linear_assignment_from_similarity(similarity, self.config.match_thresh)

    def _refresh_tracked_occlusion_coefficients(self) -> None:
        tracked = self._tracked_tracks()
        if not tracked:
            return

        boxes = np.asarray([track.output_box_xyxy() for track in tracked], dtype=np.float32)
        coeffs = compute_occlusion_coefficients(
            boxes,
            occ_trigger_thresh=self.config.occ_trigger_thresh,
            gm_kx=self.config.gm_kx,
            gm_ky=self.config.gm_ky,
        )
        for track, coeff in zip(tracked, coeffs.tolist()):
            track.set_last_occlusion_coefficient(coeff)

    def _prune_dead_tracks(self) -> None:
        kept_tracks: list[OATrack] = []
        for track in self.tracks:
            if track.state != TrackState.REMOVED and track.time_since_update > int(self.config.track_buffer):
                track.mark_removed()
                self.removed_tracks.append(track)
            if track.state != TrackState.REMOVED:
                kept_tracks.append(track)
        self.tracks = kept_tracks

    def _deduplicate_live_tracks(self) -> None:
        tracked = self._tracked_tracks()
        lost = self._lost_tracks()
        tracked, lost = remove_duplicate_tracks(tracked, lost)
        tracked_ids = {track.track_id for track in tracked}
        lost_ids = {track.track_id for track in lost}
        self.tracks = [track for track in self.tracks if track.track_id in tracked_ids or track.track_id in lost_ids]

    def _outputs(self) -> list[TrackOutputRow]:
        return [track.to_output_row() for track in self._tracked_tracks() if track.ready_for_output(self.config.min_hits)]

    def update(self, detections: np.ndarray) -> list[TrackOutputRow]:
        det_np = np.asarray(detections, dtype=np.float32)
        if det_np.size == 0:
            det_np = np.empty((0, 6), dtype=np.float32)
        elif det_np.ndim == 1:
            det_np = det_np[None, :]

        self.frame_id += 1
        self._predict_live_tracks()

        det_scores = det_np[:, 4] if len(det_np) else np.empty((0,), dtype=np.float32)
        high_mask = det_scores > float(self.config.track_high_thresh)
        low_mask = (det_scores > float(self.config.track_low_thresh)) & ~high_mask

        detections_high = det_np[high_mask]
        detections_low = det_np[low_mask]

        all_track_indices = np.asarray(
            [idx for idx, track in enumerate(self.tracks) if track.state != TrackState.REMOVED],
            dtype=int,
        )

        high_matches, unmatched_high_det_idx, unmatched_track_local = self._first_stage_association(
            detections_high,
            all_track_indices,
        )
        matched_track_indices: set[int] = set()
        for det_local, track_local in high_matches.tolist():
            track_idx = int(all_track_indices[track_local])
            self.tracks[track_idx].update_with_detection(detections_high[det_local], frame_id=self.frame_id)
            matched_track_indices.add(track_idx)

        remaining_track_indices = np.asarray(
            [int(all_track_indices[idx]) for idx in unmatched_track_local.tolist()],
            dtype=int,
        )
        low_matches, _, unmatched_low_track_local = self._second_stage_association(detections_low, remaining_track_indices)
        low_matched_track_indices: set[int] = set()
        for det_local, track_local in low_matches.tolist():
            track_idx = int(remaining_track_indices[track_local])
            track = self.tracks[track_idx]
            det_row = detections_low[det_local]
            predicted_box = track.predicted_box_xyxy()
            iou_value = float(pairwise_iou_xyxy(det_row[None, :4], predicted_box[None, :])[0, 0])
            bam = float(compute_bias_aware_momentum(iou_value, track.last_occlusion_coefficient))
            blended_box = blend_boxes_xyxy(det_row[:4], predicted_box, bam)[0]
            track.update_with_detection(det_row, frame_id=self.frame_id, observation_box=blended_box)
            matched_track_indices.add(track_idx)
            low_matched_track_indices.add(track_idx)

        remaining_after_low = np.asarray(
            [int(remaining_track_indices[idx]) for idx in unmatched_low_track_local.tolist()],
            dtype=int,
        )
        remaining_high_detections = detections_high[unmatched_high_det_idx]
        third_matches, unmatched_high_after_third, unmatched_track_after_third_local = self._third_stage_association(
            remaining_high_detections,
            remaining_after_low,
        )
        for det_local, track_local in third_matches.tolist():
            track_idx = int(remaining_after_low[track_local])
            self.tracks[track_idx].update_with_detection(remaining_high_detections[det_local], frame_id=self.frame_id)
            matched_track_indices.add(track_idx)

        unmatched_track_indices = np.asarray(
            [int(remaining_after_low[idx]) for idx in unmatched_track_after_third_local.tolist()],
            dtype=int,
        )
        for track_idx in unmatched_track_indices.tolist():
            self.tracks[track_idx].mark_lost()

        new_detections = remaining_high_detections[unmatched_high_after_third]
        for det_row in new_detections:
            if float(det_row[4]) < float(self.config.new_track_thresh):
                continue
            self.tracks.append(
                OATrack(
                    det_row,
                    frame_id=self.frame_id,
                    kalman_filter=self.kalman_filter,
                    delta_t=self.config.delta_t,
                )
            )

        self._prune_dead_tracks()
        self._refresh_tracked_occlusion_coefficients()
        self._deduplicate_live_tracks()
        return self._outputs()
