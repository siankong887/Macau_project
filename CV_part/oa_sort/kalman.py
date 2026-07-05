from __future__ import annotations

import numpy as np
import scipy.linalg


_EPS = 1e-6


def measurement_from_box(box: np.ndarray) -> np.ndarray:
    """Convert one xyxy(+score) box into OA/Hybrid-SORT measurement space."""
    arr = np.asarray(box, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] < 5:
        raise ValueError(f"Expected one box with at least 5 values, got shape {arr.shape}")

    x1, y1, x2, y2, score = arr[:5]
    width = max(float(x2 - x1), _EPS)
    height = max(float(y2 - y1), _EPS)
    center_x = float(x1 + width / 2.0)
    center_y = float(y1 + height / 2.0)
    area = float(width * height)
    aspect_ratio = float(width / height)
    return np.asarray([center_x, center_y, area, float(score), aspect_ratio], dtype=np.float32)


def box_from_state(mean: np.ndarray) -> np.ndarray:
    """Convert one OA/Hybrid-SORT state vector into xyxy(+score)."""
    arr = np.asarray(mean, dtype=np.float32).reshape(-1)
    if arr.shape[0] < 5:
        raise ValueError(f"Expected state vector with at least 5 values, got shape {arr.shape}")

    center_x, center_y, area, score, aspect_ratio = arr[:5]
    area = max(float(area), _EPS)
    aspect_ratio = max(float(aspect_ratio), _EPS)
    width = np.sqrt(area * aspect_ratio)
    height = max(area / max(width, _EPS), _EPS)

    return np.asarray(
        [
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
            score,
        ],
        dtype=np.float32,
    )


class OASortKalmanFilter:
    """Minimal 9D linear Kalman filter aligned with Hybrid-SORT's state layout."""

    dim_x = 9
    dim_z = 5

    def __init__(self):
        self._motion_mat = np.eye(self.dim_x, dtype=np.float32)
        for idx in range(4):
            self._motion_mat[idx, idx + self.dim_z] = 1.0

        self._update_mat = np.eye(self.dim_z, self.dim_x, dtype=np.float32)
        self._base_process_cov = self._build_process_covariance()
        self._base_measurement_cov = self._build_measurement_covariance()
        self._base_initial_cov = self._build_initial_covariance()

    @staticmethod
    def _build_process_covariance() -> np.ndarray:
        cov = np.eye(9, dtype=np.float32)
        cov[5:, 5:] *= 0.01
        cov[7, 7] *= 0.01
        cov[8, 8] *= 0.01
        return cov

    @staticmethod
    def _build_measurement_covariance() -> np.ndarray:
        cov = np.eye(5, dtype=np.float32)
        cov[2:, 2:] *= 10.0
        return cov

    @staticmethod
    def _build_initial_covariance() -> np.ndarray:
        cov = np.eye(9, dtype=np.float32)
        cov[5:, 5:] *= 1000.0
        cov *= 10.0
        return cov

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        measurement = np.asarray(measurement, dtype=np.float32).reshape(self.dim_z)
        mean = np.zeros((self.dim_x,), dtype=np.float32)
        mean[: self.dim_z] = measurement
        covariance = self._base_initial_cov.copy()
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(mean, dtype=np.float32).reshape(self.dim_x)
        covariance = np.asarray(covariance, dtype=np.float32).reshape(self.dim_x, self.dim_x)

        predicted_mean = self._motion_mat @ mean
        predicted_cov = self._motion_mat @ covariance @ self._motion_mat.T
        predicted_cov = predicted_cov + self._base_process_cov
        return predicted_mean.astype(np.float32, copy=False), predicted_cov.astype(np.float32, copy=False)

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(mean, dtype=np.float32).reshape(self.dim_x)
        covariance = np.asarray(covariance, dtype=np.float32).reshape(self.dim_x, self.dim_x)

        projected_mean = self._update_mat @ mean
        projected_cov = self._update_mat @ covariance @ self._update_mat.T
        projected_cov = projected_cov + self._base_measurement_cov
        return projected_mean.astype(np.float32, copy=False), projected_cov.astype(np.float32, copy=False)

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(mean, dtype=np.float32).reshape(self.dim_x)
        covariance = np.asarray(covariance, dtype=np.float32).reshape(self.dim_x, self.dim_x)
        measurement = np.asarray(measurement, dtype=np.float32).reshape(self.dim_z)

        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self._update_mat.T).T,
            check_finite=False,
        ).T

        innovation = measurement - projected_mean
        updated_mean = mean + kalman_gain @ innovation
        updated_cov = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return updated_mean.astype(np.float32, copy=False), updated_cov.astype(np.float32, copy=False)
