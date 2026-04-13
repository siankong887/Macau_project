from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .assignment import DEFAULT_K_PATHS, DEFAULT_LOGIT_THETA
from .observations import VEHICLE_TYPES
from .paths import ProjectPaths
from .pipeline import (
    build_assignment,
    estimate_posterior_od,
    prepare_gates,
    prepare_network,
    prepare_observations,
    prepare_sumo_assets,
    run_all,
    save_observed_turning_sumo,
    save_od_matrix,
    save_turning_summary,
)
from .types import SumoAssignmentConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Macau OD pipeline for sumo_part")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent.parent),
        help="Root directory of the sumo_part package",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level")

    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_prepare_network = subparsers.add_parser("prepare-network", help="Load local graph and zone nodes")
    parser_prepare_network.add_argument("--allow-online", action="store_true", help="Allow online OSM fallback")

    parser_prepare_obs = subparsers.add_parser(
        "prepare-observations",
        help="Build gate, edge, and turning observations for one vehicle type",
    )
    parser_prepare_obs.add_argument("--vehicle", default="car", choices=VEHICLE_TYPES)

    parser_prepare_sumo = subparsers.add_parser(
        "prepare-sumo-assets",
        help="Build and validate SUMO-native assets for one vehicle type",
    )
    parser_prepare_sumo.add_argument("--vehicle", default="car", choices=VEHICLE_TYPES)

    parser_obs_turning = subparsers.add_parser(
        "save-observed-turning",
        help="Save observation turning data mapped to SUMO edges for one vehicle type",
    )
    parser_obs_turning.add_argument("--vehicle", default="car", choices=VEHICLE_TYPES)

    parser_build_h = subparsers.add_parser("build-h", help="Build the analytic assignment matrix H")
    parser_build_h.add_argument("--K", type=int, default=DEFAULT_K_PATHS)
    parser_build_h.add_argument("--theta", type=float, default=DEFAULT_LOGIT_THETA)
    parser_build_h.add_argument("--no-cache", action="store_true", help="Ignore cached H matrices")

    parser_estimate = subparsers.add_parser("estimate-od", help="Estimate posterior OD for one vehicle type")
    parser_estimate.add_argument("--vehicle", default="car", choices=VEHICLE_TYPES)
    parser_estimate.add_argument("--K", type=int, default=DEFAULT_K_PATHS)
    parser_estimate.add_argument("--theta", type=float, default=DEFAULT_LOGIT_THETA)
    parser_estimate.add_argument("--beta", type=float, default=100.0)
    parser_estimate.add_argument("--mode", default="batch", choices=["batch", "sequential", "error_free"])
    parser_estimate.add_argument("--no-cache", action="store_true", help="Ignore cached H matrices")

    parser_run_all = subparsers.add_parser(
        "run-all",
        help="Run analytic OD estimation for all vehicles and then simulate posterior OD in SUMO",
    )
    parser_run_all.add_argument("--K", type=int, default=DEFAULT_K_PATHS)
    parser_run_all.add_argument("--theta", type=float, default=DEFAULT_LOGIT_THETA)
    parser_run_all.add_argument("--beta", type=float, default=100.0)
    parser_run_all.add_argument("--mode", default="batch", choices=["batch", "sequential", "error_free"])
    return parser


def _paths_from_args(args: argparse.Namespace) -> ProjectPaths:
    return ProjectPaths.from_root(args.root)


def _sumo_config_from_args(_: argparse.Namespace) -> SumoAssignmentConfig:
    return SumoAssignmentConfig()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        paths = _paths_from_args(args)

        if args.command == "prepare-network":
            G, zone_node_map = prepare_network(paths=paths, allow_online=args.allow_online)
            print(
                json.dumps(
                    {
                        "nodes": G.number_of_nodes(),
                        "edges": G.number_of_edges(),
                        "zones": len(zone_node_map),
                    },
                    indent=2,
                )
            )
            return

        G, zone_node_map = prepare_network(paths=paths, allow_online=False)
        gates, gate_edge_mapping = prepare_gates(G, paths=paths)
        sumo_config = _sumo_config_from_args(args)

        if args.command == "prepare-observations":
            prepared = prepare_observations(gates, vehicle_type=args.vehicle, paths=paths)
            print(
                json.dumps(
                    {
                        "vehicle_type": args.vehicle,
                        "gate_observations": len(prepared.gate_observations.gate_indices),
                        "edge_observations": len(prepared.edge_observations.gate_indices),
                        "turning_cameras": prepared.turning_observations.n_cameras,
                        "gate_total_flow": prepared.gate_observations.total_flow,
                        "edge_total_flow": prepared.edge_observations.total_flow,
                    },
                    indent=2,
                )
            )
            return

        if args.command == "prepare-sumo-assets":
            mappings, artifacts = prepare_sumo_assets(
                gates,
                vehicle_type=args.vehicle,
                paths=paths,
                sumo_config=sumo_config,
            )
            print(
                json.dumps(
                    {
                        "vehicle_type": args.vehicle,
                        "period_name": artifacts.period_name,
                        "sumo_version": artifacts.sumo_version,
                        "taz_file": artifacts.files.get("taz_file"),
                        "gate_mapping_file": artifacts.files.get("gate_mapping_file"),
                        "n_zone_fallbacks": mappings.n_zone_fallbacks,
                        "n_gate_from_mapped": mappings.n_gate_from_mapped,
                        "n_gate_turning_mapped": mappings.n_gate_turning_mapped,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return

        if args.command == "save-observed-turning":
            prepared = prepare_observations(gates, vehicle_type=args.vehicle, paths=paths)
            mappings, artifacts = prepare_sumo_assets(
                gates,
                vehicle_type=args.vehicle,
                paths=paths,
                sumo_config=sumo_config,
            )
            observed_turning_path = save_observed_turning_sumo(
                gate_mappings=mappings.gate_mappings,
                raw_obs=prepared.raw_observations,
                vehicle_type=args.vehicle,
                output_dir=paths.output_dir,
            )
            print(
                json.dumps(
                    {
                        "vehicle_type": args.vehicle,
                        "output": str(observed_turning_path),
                    },
                    indent=2,
                )
            )
            return

        H, od_pairs = build_assignment(
            G,
            zone_node_map,
            gates,
            K=getattr(args, "K", DEFAULT_K_PATHS),
            theta=getattr(args, "theta", DEFAULT_LOGIT_THETA),
            paths=paths,
            use_cache=not getattr(args, "no_cache", False),
        )

        if args.command == "build-h":
            print(
                json.dumps(
                    {
                        "shape": list(H.shape),
                        "nnz": int(H.nnz),
                        "od_pairs": len(od_pairs),
                        "mapped_gates": sum(1 for edge in gate_edge_mapping.values() if edge is not None),
                    },
                    indent=2,
                )
            )
            return

        if args.command == "estimate-od":
            prepared = prepare_observations(gates, vehicle_type=args.vehicle, paths=paths)
            result = estimate_posterior_od(
                prepared.edge_observations,
                H,
                od_pairs,
                beta=args.beta,
                mode=args.mode,
                vehicle_type=args.vehicle,
            )
            od_paths = save_od_matrix(result, sorted(zone_node_map.keys()), paths.output_dir)
            turning_path = save_turning_summary(prepared.turning_observations, args.vehicle, paths.output_dir)
            print(
                json.dumps(
                    {
                        "vehicle_type": args.vehicle,
                        "posterior_total_flow": result.info["posterior_total_flow"],
                        "mean_cv": result.info["mean_cv"],
                        "od_path": str(od_paths["mean"]),
                        "turning_path": str(turning_path),
                    },
                    indent=2,
                )
            )
            return

        if args.command == "run-all":
            results = run_all(
                K=args.K,
                theta=args.theta,
                beta=args.beta,
                mode=args.mode,
                paths=paths,
                sumo_config=sumo_config,
            )
            print(
                json.dumps(
                    {
                        "vehicles": list(results.keys()),
                        "posterior_total_flow": {
                            vehicle: result.info["posterior_total_flow"]
                            for vehicle, result in results.items()
                        },
                        "posterior_sumo_output_dir": str(paths.sumo_output_dir),
                    },
                    indent=2,
                )
            )
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
