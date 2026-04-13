from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class GateInfo:
    camera_id: str
    gate_id: str
    gate_index: int
    camera_lon: float
    camera_lat: float
    origin_direction: str
    dest_direction: str
    origin_road: str = ""
    dest_road: str = ""
    edge: Optional[tuple[str, str, str]] = None


@dataclass
class PathCache:
    od_pairs: list[tuple[int, int]]
    paths: dict[int, list[tuple[list[str], float]]]
    edge_to_gates: dict[tuple[str, str], list[int]]
    n_gates: int


@dataclass
class ObservationData:
    g: np.ndarray
    sigma: np.ndarray
    gate_indices: list[int]
    vehicle_type: str
    n_time_periods: int
    total_flow: float
    group_keys: list[Any] = field(default_factory=list)


@dataclass
class CameraTurningObs:
    camera_id: str
    n_supergates: int
    edge_keys: list[tuple[str, str]]
    observed_counts: np.ndarray
    observed_proportions: np.ndarray
    gate_indices_per_supergate: list[list[int]]
    total_count: float


@dataclass
class TurningData:
    cameras: list[CameraTurningObs]
    n_cameras: int
    total_observations: int
    total_independent: int
    vehicle_type: str


@dataclass
class PreparedObservations:
    gate_observations: ObservationData
    edge_observations: ObservationData
    turning_observations: TurningData
    raw_observations: dict[str, dict[int, list[float]]] = field(default_factory=dict)


@dataclass(frozen=True)
class SumoAssignmentConfig:
    period_name: str = "08_00_09_00"
    begin: int = 0
    end: int = 3600
    due_iterations: int = 8
    route_choice_model: str = "gawron"
    gawron_beta: float = 0.3
    gawron_a: float = 0.5
    seed: int = 42
    aggregation_freq: int = 3600
    target_total_trips: Optional[int] = None


@dataclass
class SumoZoneMapping:
    zone_id: int
    taz_id: str
    edge_ids: list[str]
    centroid_xy: tuple[float, float]
    used_centroid_fallback: bool = False


@dataclass
class SumoGateMapping:
    gate_index: int
    camera_id: str
    gate_id: str
    from_edge_id: Optional[str] = None
    to_edge_id: Optional[str] = None
    status: str = "unmapped"
    from_distance: Optional[float] = None
    to_distance: Optional[float] = None


@dataclass
class SumoNativeMappings:
    zone_mappings: dict[int, SumoZoneMapping]
    gate_mappings: dict[int, SumoGateMapping]
    taz_file: Optional[Path] = None
    gate_mapping_file: Optional[Path] = None
    n_zone_fallbacks: int = 0
    n_gate_from_mapped: int = 0
    n_gate_turning_mapped: int = 0


@dataclass
class SumoAssignmentArtifacts:
    vehicle_type: str
    period_name: str
    output_dir: Path
    sumo_version: str = ""
    tool_paths: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    gap_history: list[float] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class SumoSimulationResult:
    vehicle_type: str
    od_pairs: list[tuple[int, int]]
    H: Any
    mappings: SumoNativeMappings
    artifacts: SumoAssignmentArtifacts
