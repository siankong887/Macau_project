from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def build_uniform_prior(
    g: np.ndarray,
    n_od_pairs: int,
    beta: float = 100.0,
) -> tuple[np.ndarray, np.ndarray]:
    total_flow = float(np.sum(g))
    if total_flow <= 0:
        total_flow = float(n_od_pairs)
        logger.warning("Observed total flow is non-positive; falling back to mean 1.0")
    mean_per_od = total_flow / float(n_od_pairs)
    mu0 = np.full(n_od_pairs, mean_per_od)
    V0 = np.diag(np.full(n_od_pairs, beta * mean_per_od))
    return mu0, V0
