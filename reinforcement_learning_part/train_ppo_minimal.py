from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:
    gym = None
    spaces = None

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
except ModuleNotFoundError:
    PPO = None

    class BaseCallback:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

try:
    import torch
except ModuleNotFoundError:
    torch = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_SUMO_ROOT = REPO_ROOT / "sumo_part"
RUNS_DIR = SCRIPT_DIR / "runs"
DEFAULT_EPS = 1.0
DEFAULT_TIMESTEPS = 8

if str(DEFAULT_SUMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_SUMO_ROOT))

from sumo_part.paths import ProjectPaths
from sumo_part.pipeline import (
    prepare_gates,
    prepare_network,
    prepare_observations,
    prepare_sumo_assets,
    simulate_posterior_sumo,
)
from sumo_part.sumo_backend import build_observed_turning_sumo
from sumo_part.types import SumoAssignmentConfig


MovementKey = tuple[str, str, str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal PPO OD-delta trainer for SUMO turning rewards")
    parser.add_argument("--sumo-root", default=str(DEFAULT_SUMO_ROOT), help="Root directory of the sumo_part project")
    parser.add_argument("--vehicle", default="car", help="Vehicle type to optimize")
    parser.add_argument("--base-od-csv", default=None, help="Baseline OD CSV path")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS, help="Total PPO timesteps")
    parser.add_argument("--delta-scale", type=float, default=1.0, help="Scaling factor applied to action deltas")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default=None, help="Directory for training outputs")
    parser.add_argument("--smoke-episode", action="store_true", help="Run one zero-action episode without training")
    return parser.parse_args()


def check_required_dependencies() -> None:
    missing: list[str] = []
    for module_name in ("stable_baselines3", "gymnasium", "torch"):
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(module_name)
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            "Missing RL dependencies: "
            f"{missing_text}. Install them in the project venv first, for example: "
            "./.venv/bin/pip install stable-baselines3 gymnasium torch"
        )


def resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (RUNS_DIR / timestamp).resolve()


def load_od_matrix_csv(path: Path) -> tuple[list[int], np.ndarray]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"OD CSV is missing a valid header: {path}")

    zone_ids = [int(value) for value in rows[0][1:]]
    matrix_rows: list[list[float]] = []
    row_zone_ids: list[int] = []
    for row in rows[1:]:
        if not row:
            continue
        row_zone_ids.append(int(row[0]))
        matrix_rows.append([float(value) for value in row[1:]])

    if row_zone_ids != zone_ids:
        raise ValueError(f"OD CSV row/column ordering mismatch in {path}")

    matrix = np.asarray(matrix_rows, dtype=np.float64)
    expected_shape = (len(zone_ids), len(zone_ids))
    if matrix.shape != expected_shape:
        raise ValueError(f"OD matrix shape {matrix.shape} does not match expected {expected_shape}")
    return zone_ids, matrix


def off_diagonal_mask(size: int) -> np.ndarray:
    return ~np.eye(size, dtype=bool)


def matrix_to_offdiag_vector(matrix: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"OD matrix must be square, got shape {matrix.shape}")
    effective_mask = mask if mask is not None else off_diagonal_mask(matrix.shape[0])
    return np.asarray(matrix[effective_mask], dtype=np.float64)


def offdiag_vector_to_matrix(vector: np.ndarray, size: int, mask: np.ndarray | None = None) -> np.ndarray:
    effective_mask = mask if mask is not None else off_diagonal_mask(size)
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    expected_size = int(np.count_nonzero(effective_mask))
    if vector.size != expected_size:
        raise ValueError(f"Off-diagonal vector length {vector.size} does not match expected {expected_size}")
    matrix = np.zeros((size, size), dtype=np.float64)
    matrix[effective_mask] = vector
    np.fill_diagonal(matrix, 0.0)
    return matrix


def stable_softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = np.sum(exp_values)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Softmax normalization failed due to invalid total")
    return exp_values / total


def candidate_offdiag_from_action(
    base_off_diag: np.ndarray,
    action: np.ndarray,
    delta_scale: float,
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    base_off_diag = np.asarray(base_off_diag, dtype=np.float64).reshape(-1)
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if base_off_diag.shape != action.shape:
        raise ValueError(
            f"Action shape {action.shape} does not match base off-diagonal shape {base_off_diag.shape}"
        )

    base_total = float(np.sum(base_off_diag))
    if base_total <= 0.0:
        raise ValueError("Baseline OD total demand must be positive")

    base_logits = np.log(base_off_diag + float(eps))
    delta_scaled = action * float(delta_scale)
    shares = stable_softmax(base_logits + delta_scaled)
    return base_total * shares


def candidate_matrix_from_action(
    base_matrix: np.ndarray,
    action: np.ndarray,
    delta_scale: float,
    eps: float = DEFAULT_EPS,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    base_matrix = np.asarray(base_matrix, dtype=np.float64)
    effective_mask = mask if mask is not None else off_diagonal_mask(base_matrix.shape[0])
    candidate_off_diag = candidate_offdiag_from_action(
        base_off_diag=matrix_to_offdiag_vector(base_matrix, effective_mask),
        action=action,
        delta_scale=delta_scale,
        eps=eps,
    )
    return offdiag_vector_to_matrix(candidate_off_diag, size=base_matrix.shape[0], mask=effective_mask)


def movement_key(movement: dict[str, Any]) -> MovementKey:
    return (
        str(movement.get("camera_id", "")),
        str(movement.get("gate_id", "")),
        str(movement.get("from_edge_id", "")),
        str(movement.get("to_edge_id", "")),
    )


def normalize_turning_payload(payload: dict[str, Any]) -> dict[MovementKey, dict[str, float]]:
    normalized: dict[MovementKey, dict[str, float]] = {}
    for camera in payload.get("cameras", []):
        for movement in camera.get("movements", []):
            key = movement_key(movement)
            normalized[key] = {
                "count": float(movement.get("count", 0.0)),
                "proportion": float(movement.get("proportion", 0.0)),
            }
    return normalized


def compute_turning_errors(
    observed: dict[MovementKey, dict[str, float]],
    simulated: dict[MovementKey, dict[str, float]],
) -> tuple[float, float, int]:
    keys = sorted(set(observed) | set(simulated))
    if not keys:
        raise ValueError("No turning movements are available for reward computation")

    obs_counts = np.array([observed.get(key, {}).get("count", 0.0) for key in keys], dtype=np.float64)
    sim_counts = np.array([simulated.get(key, {}).get("count", 0.0) for key in keys], dtype=np.float64)
    obs_prop = np.array([observed.get(key, {}).get("proportion", 0.0) for key in keys], dtype=np.float64)
    sim_prop = np.array([simulated.get(key, {}).get("proportion", 0.0) for key in keys], dtype=np.float64)

    count_error = float(np.mean(np.abs(np.log1p(sim_counts) - np.log1p(obs_counts))))
    proportion_error = float(np.mean(np.abs(sim_prop - obs_prop)))
    return count_error, proportion_error, len(keys)


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def json_ready(value: Any) -> Any:
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_ready(item) for item in value]
    return value


class SumoOdDeltaEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        sumo_root: Path,
        vehicle: str,
        base_od_matrix: np.ndarray,
        zone_ids: list[int],
        delta_scale: float,
        sumo_config: SumoAssignmentConfig,
        seed: int | None = None,
    ) -> None:
        if gym is None or spaces is None:
            raise RuntimeError("gymnasium is required to instantiate SumoOdDeltaEnv")
        super().__init__()

        self.paths = ProjectPaths.from_root(sumo_root)
        self.vehicle = vehicle
        self.base_od_matrix = np.asarray(base_od_matrix, dtype=np.float64)
        self.zone_ids = [int(zone_id) for zone_id in zone_ids]
        self.delta_scale = float(delta_scale)
        self.sumo_config = sumo_config
        self.matrix_mask = off_diagonal_mask(self.base_od_matrix.shape[0])
        self.base_off_diag = matrix_to_offdiag_vector(self.base_od_matrix, self.matrix_mask)
        self.base_total_demand = float(np.sum(self.base_off_diag))
        self.base_observation = np.log1p(self.base_off_diag).astype(np.float32)
        if self.base_total_demand <= 0.0:
            raise ValueError("Baseline OD matrix has zero total off-diagonal demand")

        self.observation_space = spaces.Box(
            low=0.0,
            high=np.finfo(np.float32).max,
            shape=(self.base_observation.size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.base_observation.size,),
            dtype=np.float32,
        )

        self._seed = seed
        self._last_info: dict[str, Any] = {}
        self._last_candidate_matrix = self.base_od_matrix.copy()

        self.network, _ = prepare_network(paths=self.paths, allow_online=False)
        self.gates, _ = prepare_gates(self.network, paths=self.paths)
        self.prepared_observations = prepare_observations(self.gates, vehicle_type=self.vehicle, paths=self.paths)
        self.mappings, self.asset_artifacts = prepare_sumo_assets(
            self.gates,
            vehicle_type=self.vehicle,
            paths=self.paths,
            sumo_config=self.sumo_config,
        )
        observed_payload = build_observed_turning_sumo(
            gate_mappings=self.mappings.gate_mappings,
            raw_obs=self.prepared_observations.raw_observations,
            vehicle_type=self.vehicle,
        )
        self.observed_turning = normalize_turning_payload(observed_payload)
        if not self.observed_turning:
            raise ValueError("Observed turning payload is empty, reward cannot be computed")

    @property
    def observed_movement_count(self) -> int:
        return len(self.observed_turning)

    @property
    def last_info(self) -> dict[str, Any]:
        return dict(self._last_info)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
        self._last_info = {}
        self._last_candidate_matrix = self.base_od_matrix.copy()
        info = {
            "vehicle": self.vehicle,
            "base_total_demand": self.base_total_demand,
            "observed_movement_count": self.observed_movement_count,
        }
        return self.base_observation.copy(), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        candidate_matrix = candidate_matrix_from_action(
            base_matrix=self.base_od_matrix,
            action=action_array,
            delta_scale=self.delta_scale,
            eps=DEFAULT_EPS,
            mask=self.matrix_mask,
        )
        self._last_candidate_matrix = candidate_matrix

        sumo_result = simulate_posterior_sumo(
            gates=self.gates,
            vehicle_type=self.vehicle,
            zone_ids=self.zone_ids,
            posterior_od_matrix=candidate_matrix,
            paths=self.paths,
            sumo_config=self.sumo_config,
            use_cache=True,
            write_cache=True,
        )
        turning_file = Path(sumo_result.artifacts.files["turning_file"])
        if not turning_file.exists():
            raise FileNotFoundError(f"SUMO turning file was not produced: {turning_file}")

        simulated_payload = read_json(turning_file)
        simulated_turning = normalize_turning_payload(simulated_payload)
        count_error, proportion_error, compared_movements = compute_turning_errors(
            observed=self.observed_turning,
            simulated=simulated_turning,
        )
        reward = -(count_error + proportion_error)

        candidate_total = float(np.sum(candidate_matrix[self.matrix_mask]))
        info = {
            "reward": float(reward),
            "turning_count_error": count_error,
            "turning_proportion_error": proportion_error,
            "sumo_turning_file": str(turning_file),
            "base_total_demand": self.base_total_demand,
            "candidate_total_demand": candidate_total,
            "compared_movements": compared_movements,
        }
        self._last_info = info
        observation = np.log1p(matrix_to_offdiag_vector(candidate_matrix, self.matrix_mask)).astype(np.float32)
        return observation, float(reward), True, False, info

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


class EpisodeMetricsCallback(BaseCallback):
    def __init__(self, metrics_path: Path, summary_path: Path) -> None:
        super().__init__()
        self.metrics_path = metrics_path
        self.summary_path = summary_path
        self.best_entry: dict[str, Any] | None = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done or "reward" not in info:
                continue
            entry = {
                "num_timesteps": int(self.num_timesteps),
                "reward": float(info["reward"]),
                "turning_count_error": float(info["turning_count_error"]),
                "turning_proportion_error": float(info["turning_proportion_error"]),
                "sumo_turning_file": str(info["sumo_turning_file"]),
                "base_total_demand": float(info["base_total_demand"]),
                "candidate_total_demand": float(info["candidate_total_demand"]),
                "compared_movements": int(info["compared_movements"]),
            }
            with open(self.metrics_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if self.best_entry is None or entry["reward"] > self.best_entry["reward"]:
                self.best_entry = entry
                with open(self.summary_path, "w", encoding="utf-8") as handle:
                    json.dump(entry, handle, indent=2, ensure_ascii=False)
        return True


def build_run_config(
    *,
    args: argparse.Namespace,
    env: SumoOdDeltaEnv,
    output_dir: Path,
    base_od_csv: Path,
) -> dict[str, Any]:
    return {
        "sumo_root": str(env.paths.root),
        "vehicle": args.vehicle,
        "base_od_csv": str(base_od_csv),
        "timesteps": int(args.timesteps),
        "delta_scale": float(args.delta_scale),
        "seed": int(args.seed),
        "output_dir": str(output_dir),
        "smoke_episode": bool(args.smoke_episode),
        "base_total_demand": env.base_total_demand,
        "observed_movement_count": env.observed_movement_count,
        "sumo_config": asdict(env.sumo_config),
    }


def run_smoke_episode(env: SumoOdDeltaEnv) -> dict[str, Any]:
    observation, reset_info = env.reset(seed=env._seed)
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    next_observation, reward, terminated, truncated, info = env.step(zero_action)
    candidate_matrix = env._last_candidate_matrix
    mask = env.matrix_mask
    summary = {
        "reset_info": reset_info,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "turning_count_error": float(info["turning_count_error"]),
        "turning_proportion_error": float(info["turning_proportion_error"]),
        "sumo_turning_file": str(info["sumo_turning_file"]),
        "base_total_demand": float(info["base_total_demand"]),
        "candidate_total_demand": float(info["candidate_total_demand"]),
        "candidate_non_negative": bool(np.all(candidate_matrix >= 0.0)),
        "candidate_diagonal_zero": bool(np.allclose(np.diag(candidate_matrix), 0.0)),
        "candidate_total_matches_base": bool(np.isclose(np.sum(candidate_matrix[mask]), env.base_total_demand)),
        "observation_shape": list(observation.shape),
        "next_observation_shape": list(next_observation.shape),
    }
    return summary


def train_model(args: argparse.Namespace, env: SumoOdDeltaEnv, output_dir: Path) -> None:
    if PPO is None or torch is None:
        raise RuntimeError("stable_baselines3 and torch must be installed before training")

    n_steps = max(2, min(int(args.timesteps), 8))
    callback = EpisodeMetricsCallback(
        metrics_path=output_dir / "episode_metrics.jsonl",
        summary_path=output_dir / "best_reward_summary.json",
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=int(args.seed),
        device="auto",
        n_steps=n_steps,
        batch_size=n_steps,
        gamma=1.0,
        gae_lambda=1.0,
        n_epochs=1,
    )
    model.learn(total_timesteps=int(args.timesteps), callback=callback, progress_bar=False)
    model.save(str(output_dir / "final_model"))

    if callback.best_entry is None and env.last_info:
        with open(output_dir / "best_reward_summary.json", "w", encoding="utf-8") as handle:
            json.dump(json_ready(env.last_info), handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    check_required_dependencies()

    sumo_root = Path(args.sumo_root).resolve()
    if not sumo_root.exists():
        raise FileNotFoundError(f"SUMO project root does not exist: {sumo_root}")

    base_od_csv = Path(args.base_od_csv).resolve() if args.base_od_csv else sumo_root / "output" / f"od_{args.vehicle}.csv"
    if not base_od_csv.exists():
        raise FileNotFoundError(f"Baseline OD CSV does not exist: {base_od_csv}")

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zone_ids, base_od_matrix = load_od_matrix_csv(base_od_csv)
    env = SumoOdDeltaEnv(
        sumo_root=sumo_root,
        vehicle=args.vehicle,
        base_od_matrix=base_od_matrix,
        zone_ids=zone_ids,
        delta_scale=args.delta_scale,
        sumo_config=SumoAssignmentConfig(),
        seed=args.seed,
    )

    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(
            build_run_config(args=args, env=env, output_dir=output_dir, base_od_csv=base_od_csv),
            handle,
            indent=2,
            ensure_ascii=False,
        )

    if args.smoke_episode:
        summary = run_smoke_episode(env)
        print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False))
        return

    train_model(args, env, output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "final_model": str(output_dir / "final_model.zip"),
                "metrics": str(output_dir / "episode_metrics.jsonl"),
                "best_summary": str(output_dir / "best_reward_summary.json"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
