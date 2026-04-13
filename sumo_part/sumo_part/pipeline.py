from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .assignment import build_assignment as _build_assignment
from .assignment import prepare_gates as _prepare_gates
from .bayes import BayesianODResult, UpdateMode, bayesian_update
from .network import prepare_network as _prepare_network
from .observations import VEHICLE_TYPES, prepare_observations as _prepare_observations
from .paths import ProjectPaths, default_paths
from .prior import build_uniform_prior
from .sumo_backend import build_observed_turning_sumo, ensure_sumo_assets, simulate_posterior_sumo as _simulate_posterior_sumo
from .types import SumoAssignmentConfig, SumoGateMapping, SumoSimulationResult

logger = logging.getLogger(__name__)


def prepare_network(
    paths: ProjectPaths | None = None,
    allow_online: bool = False,
) -> tuple[object, dict[int, str]]:
    return _prepare_network(paths=paths, allow_online=allow_online)


def prepare_gates(
    G: object,
    paths: ProjectPaths | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
) -> tuple[list, dict[int, tuple[str, str, str] | None]]:
    return _prepare_gates(G, paths=paths, use_cache=use_cache, write_cache=write_cache)


def prepare_observations(
    gates: list,
    vehicle_type: str,
    paths: ProjectPaths | None = None,
):
    return _prepare_observations(
        gates=gates,
        vehicle_type=vehicle_type,
        paths=paths,
    )


def prepare_sumo_assets(
    gates: list,
    vehicle_type: str,
    paths: ProjectPaths | None = None,
    sumo_config: SumoAssignmentConfig | None = None,
):
    return ensure_sumo_assets(
        gates=gates,
        vehicle_type=vehicle_type,
        paths=paths,
        sumo_config=sumo_config,
    )


def build_assignment(
    G: object,
    zone_node_map: dict[int, str],
    gates: list,
    K: int = 3,
    theta: float = 0.1,
    paths: ProjectPaths | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
) -> tuple[sp.csr_matrix, list[tuple[int, int]]]:
    return _build_assignment(
        G,
        zone_node_map,
        gates,
        K=K,
        theta=theta,
        paths=paths,
        use_cache=use_cache,
        write_cache=write_cache,
    )


def simulate_posterior_sumo(
    gates: list,
    vehicle_type: str,
    zone_ids: list[int] | None = None,
    posterior_od_matrix: np.ndarray | None = None,
    od_csv_path: Path | None = None,
    paths: ProjectPaths | None = None,
    sumo_config: SumoAssignmentConfig | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
) -> SumoSimulationResult:
    return _simulate_posterior_sumo(
        gates=gates,
        vehicle_type=vehicle_type,
        zone_ids=zone_ids,
        posterior_od_matrix=posterior_od_matrix,
        od_csv_path=od_csv_path,
        paths=paths,
        sumo_config=sumo_config,
        use_cache=use_cache,
        write_cache=write_cache,
    )


def _zone_ids_from_od_pairs(od_pairs: list[tuple[int, int]]) -> list[int]:
    zone_ids = sorted({value for pair in od_pairs for value in pair})
    if len(zone_ids) != 23:
        logger.warning("Expected 23 zones but found %d in od_pairs", len(zone_ids))
    return zone_ids


def estimate_posterior_od(
    edge_observations,
    H: sp.csr_matrix,
    od_pairs: list[tuple[int, int]],
    beta: float = 100.0,
    mode: str = "batch",
    vehicle_type: str = "",
) -> BayesianODResult:
    zone_ids = _zone_ids_from_od_pairs(od_pairs)
    zone_idx_map = {zone_id: idx for idx, zone_id in enumerate(zone_ids)}
    od_pairs_idx = [(zone_idx_map[o], zone_idx_map[d]) for o, d in od_pairs]
    mu0, V0 = build_uniform_prior(edge_observations.g, n_od_pairs=len(od_pairs), beta=beta)
    H_obs = H[edge_observations.gate_indices, :]
    if sp.issparse(H_obs):
        H_obs = H_obs.toarray()
    update_mode = UpdateMode(mode)
    return bayesian_update(
        mu0=mu0,
        V0=V0,
        H=H_obs,
        g=edge_observations.g,
        Sigma=edge_observations.sigma,
        od_pairs=od_pairs_idx,
        mode=update_mode,
        vehicle_type=vehicle_type,
        n_zones=len(zone_ids),
    )


def save_od_matrix(result: BayesianODResult, zone_ids: list[int], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(zone_id) for zone_id in zone_ids]
    outputs = {
        "mean": (output_dir / f"od_{result.vehicle_type}.csv", result.od_matrix),
        "lower": (output_dir / f"od_{result.vehicle_type}_lower.csv", result.od_matrix_lower),
        "upper": (output_dir / f"od_{result.vehicle_type}_upper.csv", result.od_matrix_upper),
    }
    for path, matrix in outputs.values():
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["origin_zona", *labels])
            for zone_id, row in zip(labels, matrix):
                writer.writerow([zone_id, *[f"{value:.6f}" for value in row]])
    return {name: path for name, (path, _) in outputs.items()}


def save_turning_summary(turning_observations, vehicle_type: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"turning_{vehicle_type}.json"
    payload = {
        "vehicle_type": vehicle_type,
        "n_cameras": turning_observations.n_cameras,
        "total_observations": turning_observations.total_observations,
        "total_independent": turning_observations.total_independent,
        "cameras": [
            {
                "camera_id": camera.camera_id,
                "n_supergates": camera.n_supergates,
                "edge_keys": [list(edge) for edge in camera.edge_keys],
                "observed_counts": camera.observed_counts.tolist(),
                "observed_proportions": camera.observed_proportions.tolist(),
                "gate_indices_per_supergate": camera.gate_indices_per_supergate,
                "total_count": camera.total_count,
            }
            for camera in turning_observations.cameras
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def save_observed_turning_sumo(
    gate_mappings: dict[int, SumoGateMapping],
    raw_obs: dict[str, dict[int, list[float]]],
    vehicle_type: str,
    output_dir: Path,
) -> Path:
    """将观测侧转向数据映射到 SUMO edge 体系并保存为 JSON。

    产出文件格式与 SUMO 仿真产出的 turning_{vehicle}__posterior_sumo.json 一致，
    方便两者按 (camera_id, gate_id) 逐条对比。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"turning_{vehicle_type}__observed.json"
    payload = build_observed_turning_sumo(gate_mappings, raw_obs, vehicle_type)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    logger.info(
        "Saved observed turning (SUMO edges) for %s: %d matched movements → %s",
        vehicle_type,
        payload["matched_movements"],
        path,
    )
    return path


def run_vehicle_pipeline(
    vehicle_type: str,
    K: int = 3,
    theta: float = 0.1,
    beta: float = 100.0,
    mode: str = "batch",
    paths: ProjectPaths | None = None,
    use_assignment_cache: bool = True,
    sumo_config: SumoAssignmentConfig | None = None,
) -> dict[str, object]:
    paths = paths or default_paths()
    G, zone_node_map = prepare_network(paths=paths, allow_online=False)
    gates, gate_edge_mapping = prepare_gates(G, paths=paths)
    prepared = prepare_observations(gates, vehicle_type=vehicle_type, paths=paths)
    H, od_pairs = build_assignment(
        G,
        zone_node_map,
        gates,
        K=K,
        theta=theta,
        paths=paths,
        use_cache=use_assignment_cache,
    )
    result = estimate_posterior_od(
        prepared.edge_observations,
        H,
        od_pairs,
        beta=beta,
        mode=mode,
        vehicle_type=vehicle_type,
    )
    zone_ids = sorted(zone_node_map.keys())
    od_paths = save_od_matrix(result, zone_ids, paths.output_dir)
    turning_path = save_turning_summary(prepared.turning_observations, vehicle_type, paths.output_dir)
    sumo_simulation = simulate_posterior_sumo(
        gates=gates,
        vehicle_type=vehicle_type,
        zone_ids=zone_ids,
        od_csv_path=od_paths["mean"],
        paths=paths,
        sumo_config=sumo_config,
        use_cache=use_assignment_cache,
    )
    observed_turning_path = save_observed_turning_sumo(
        gate_mappings=sumo_simulation.mappings.gate_mappings,
        raw_obs=prepared.raw_observations,
        vehicle_type=vehicle_type,
        output_dir=paths.output_dir,
    )
    return {
        "network": G,
        "zone_node_map": zone_node_map,
        "gates": gates,
        "gate_edge_mapping": gate_edge_mapping,
        "gate_observations": prepared.gate_observations,
        "edge_observations": prepared.edge_observations,
        "turning_observations": prepared.turning_observations,
        "H": H,
        "od_pairs": od_pairs,
        "result": result,
        "od_paths": od_paths,
        "turning_path": turning_path,
        "observed_turning_path": observed_turning_path,
        "sumo_simulation": sumo_simulation,
    }


def run_all(
    vehicle_types: list[str] | None = None,
    K: int = 3,
    theta: float = 0.1,
    beta: float = 100.0,
    mode: str = "batch",
    paths: ProjectPaths | None = None,
    sumo_config: SumoAssignmentConfig | None = None,
) -> dict[str, BayesianODResult]:
    paths = paths or default_paths()
    if vehicle_types is None:
        vehicle_types = list(VEHICLE_TYPES)
    G, zone_node_map = prepare_network(paths=paths, allow_online=False)
    gates, _ = prepare_gates(G, paths=paths)
    zone_ids = sorted(zone_node_map.keys())
    results: dict[str, BayesianODResult] = {}
    H, od_pairs = build_assignment(G, zone_node_map, gates, K=K, theta=theta, paths=paths)
    for vehicle_type in vehicle_types:
        prepared = prepare_observations(gates, vehicle_type=vehicle_type, paths=paths)
        result = estimate_posterior_od(
            prepared.edge_observations,
            H,
            od_pairs,
            beta=beta,
            mode=mode,
            vehicle_type=vehicle_type,
        )
        od_paths = save_od_matrix(result, zone_ids, paths.output_dir)
        save_turning_summary(prepared.turning_observations, vehicle_type, paths.output_dir)
        sumo_result = simulate_posterior_sumo(
            gates=gates,
            vehicle_type=vehicle_type,
            zone_ids=zone_ids,
            od_csv_path=od_paths["mean"],
            paths=paths,
            sumo_config=sumo_config,
        )
        save_observed_turning_sumo(
            gate_mappings=sumo_result.mappings.gate_mappings,
            raw_obs=prepared.raw_observations,
            vehicle_type=vehicle_type,
            output_dir=paths.output_dir,
        )
        results[vehicle_type] = result
    return results
