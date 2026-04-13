from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import scipy.sparse as sp

logger = logging.getLogger(__name__)

N_ZONES = 23


class UpdateMode(Enum):
    BATCH = "batch"
    SEQUENTIAL = "sequential"
    ERROR_FREE = "error_free"


@dataclass
class BayesianODResult:
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    posterior_variance: np.ndarray
    confidence_intervals_95: np.ndarray
    od_matrix: np.ndarray
    od_matrix_lower: np.ndarray
    od_matrix_upper: np.ndarray
    prior_mean: np.ndarray
    vehicle_type: str = ""
    update_mode: UpdateMode = UpdateMode.BATCH
    n_clipped: int = 0
    info: dict[str, float] = field(default_factory=dict)


def batch_update(
    mu0: np.ndarray,
    V0: np.ndarray,
    H: np.ndarray,
    g: np.ndarray,
    Sigma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if sp.issparse(H):
        H = H.toarray()
    v0_diag = np.diag(V0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        V0H_T = v0_diag[:, None] * H.T
        M = Sigma + H @ V0H_T
        residual = g - H @ mu0
        M_inv_residual = np.linalg.solve(M, residual)
        HV0 = H * v0_diag[None, :]
        M_inv_HV0 = np.linalg.solve(M, HV0)
        mu1 = mu0 + V0H_T @ M_inv_residual
        V1 = V0 - V0H_T @ M_inv_HV0
    if not (
        np.isfinite(M).all()
        and np.isfinite(residual).all()
        and np.isfinite(mu1).all()
        and np.isfinite(V1).all()
    ):
        raise FloatingPointError("Non-finite values encountered during batch_update")
    return mu1, V1


def sequential_update(
    mu0: np.ndarray,
    V0: np.ndarray,
    H: np.ndarray,
    g: np.ndarray,
    sigma_diag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if sp.issparse(H):
        H = H.toarray()
    mu = mu0.copy()
    V = V0.copy()
    for idx in range(len(g)):
        h = H[idx, :]
        tau_sq = sigma_diag[idx]
        obs = g[idx]
        S = V @ h
        T = float(h @ S)
        denom = tau_sq + T
        if denom < 1e-12:
            continue
        v = float(h @ mu)
        factor = 1.0 / denom
        mu = mu + S * factor * (obs - v)
        V = V - factor * np.outer(S, S)
    return mu, V


def error_free_sequential(
    mu0: np.ndarray,
    V0: np.ndarray,
    H: np.ndarray,
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return sequential_update(mu0, V0, H, g, np.zeros(len(g)))


def compute_confidence_intervals(
    mu: np.ndarray,
    V: np.ndarray,
    confidence: float = 0.95,
) -> np.ndarray:
    from scipy.stats import norm

    z = norm.ppf(1 - (1 - confidence) / 2)
    std = np.sqrt(np.maximum(np.diag(V), 0.0))
    lower = np.maximum(mu - z * std, 0.0)
    upper = mu + z * std
    return np.column_stack([lower, upper])


def reshape_to_od_matrix(
    vector: np.ndarray,
    od_pairs: list[tuple[int, int]],
    n_zones: int = N_ZONES,
) -> np.ndarray:
    matrix = np.zeros((n_zones, n_zones))
    for idx, (origin, dest) in enumerate(od_pairs):
        if idx < len(vector):
            matrix[origin, dest] = vector[idx]
    return matrix


def bayesian_update(
    mu0: np.ndarray,
    V0: np.ndarray,
    H: np.ndarray,
    g: np.ndarray,
    Sigma: np.ndarray,
    od_pairs: list[tuple[int, int]],
    mode: UpdateMode = UpdateMode.BATCH,
    vehicle_type: str = "",
    n_zones: int = N_ZONES,
) -> BayesianODResult:
    if mode == UpdateMode.BATCH:
        mu1, V1 = batch_update(mu0, V0, H, g, Sigma)
    elif mode == UpdateMode.SEQUENTIAL:
        mu1, V1 = sequential_update(mu0, V0, H, g, np.diag(Sigma))
    elif mode == UpdateMode.ERROR_FREE:
        mu1, V1 = error_free_sequential(mu0, V0, H, g)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    n_clipped = int(np.sum(mu1 < 0))
    if n_clipped:
        mu1 = np.maximum(mu1, 0.0)

    ci = compute_confidence_intervals(mu1, V1)
    od_matrix = reshape_to_od_matrix(mu1, od_pairs, n_zones=n_zones)
    od_lower = reshape_to_od_matrix(ci[:, 0], od_pairs, n_zones=n_zones)
    od_upper = reshape_to_od_matrix(ci[:, 1], od_pairs, n_zones=n_zones)
    prior_total = float(np.sum(mu0))
    posterior_total = float(np.sum(mu1))
    mean_cv = float(np.mean(np.sqrt(np.maximum(np.diag(V1), 0.0)) / np.maximum(mu1, 1e-10)))
    return BayesianODResult(
        posterior_mean=mu1,
        posterior_covariance=V1,
        posterior_variance=np.diag(V1),
        confidence_intervals_95=ci,
        od_matrix=od_matrix,
        od_matrix_lower=od_lower,
        od_matrix_upper=od_upper,
        prior_mean=mu0.copy(),
        vehicle_type=vehicle_type,
        update_mode=mode,
        n_clipped=n_clipped,
        info={
            "prior_total_flow": prior_total,
            "posterior_total_flow": posterior_total,
            "mean_cv": mean_cv,
        },
    )
