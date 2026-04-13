from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_DEPENDENCY_ERROR = None
try:
    import networkx as nx
    import numpy as np
    import scipy.sparse as sp
    from shapely.geometry import Polygon
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local environment
    _DEPENDENCY_ERROR = exc

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if _DEPENDENCY_ERROR is None:
    from sumo_part.assignment import build_assignment
    from sumo_part.paths import ProjectPaths
    from sumo_part.pipeline import run_all
    from sumo_part.sumo_backend import (
        _SumoEdgeRecord,
        _build_h_from_sumo_routes,
        _build_turning_output,
        _prepare_posterior_demand_metadata,
        _write_tazrelation_od,
        build_observed_turning_sumo,
        build_zone_mappings_from_records,
        discover_sumo_tools,
        map_gates_to_sumo_records,
    )
    from sumo_part.types import GateInfo, SumoAssignmentConfig, SumoGateMapping, SumoZoneMapping


@unittest.skipIf(_DEPENDENCY_ERROR is not None, f"Optional test dependency is unavailable: {_DEPENDENCY_ERROR}")
class PosteriorSumoHelpersTest(unittest.TestCase):
    def test_zone_mappings_use_centroid_fallback_when_polygon_misses_edges(self) -> None:
        edge_records = [
            _SumoEdgeRecord("edge_a", "n1", "n2", [(0.0, 0.0), (10.0, 0.0)], name="A"),
            _SumoEdgeRecord("edge_b", "n2", "n3", [(20.0, 0.0), (30.0, 0.0)], name="B"),
        ]
        zone_shapes = [
            (1, Polygon([(-1, -1), (11, -1), (11, 1), (-1, 1)])),
            (2, Polygon([(39, -1), (41, -1), (41, 1), (39, 1)])),
        ]
        centroid_lookup = {1: (5.0, 0.0), 2: (40.0, 0.0)}

        mappings = build_zone_mappings_from_records(zone_shapes, edge_records, centroid_lookup)

        self.assertEqual(mappings[1].edge_ids, ["edge_a"])
        self.assertFalse(mappings[1].used_centroid_fallback)
        self.assertEqual(mappings[2].edge_ids, ["edge_b"])
        self.assertTrue(mappings[2].used_centroid_fallback)

    def test_gate_mapping_prefers_direction_and_road_name(self) -> None:
        gate = GateInfo(
            camera_id="cam_1",
            gate_id="gate_1",
            gate_index=0,
            camera_lon=0.0,
            camera_lat=0.0,
            origin_direction="south",
            dest_direction="east",
            origin_road="Road In",
            dest_road="Road Out",
        )
        edge_records = [
            _SumoEdgeRecord("edge_in", "n1", "n2", [(0.0, -10.0), (0.0, 0.0)], name="Road In"),
            _SumoEdgeRecord("edge_out", "n2", "n3", [(0.0, 0.0), (10.0, 0.0)], name="Road Out"),
            _SumoEdgeRecord("edge_noise", "n4", "n5", [(-10.0, 0.0), (0.0, 0.0)], name="Noise"),
        ]

        mappings = map_gates_to_sumo_records([gate], edge_records, {"cam_1": (0.0, 0.0)})
        mapping = mappings[0]

        self.assertEqual(mapping.from_edge_id, "edge_in")
        self.assertEqual(mapping.to_edge_id, "edge_out")
        self.assertEqual(mapping.status, "mapped_turning")

    def test_prepare_posterior_demand_metadata_rounds_and_clips(self) -> None:
        metadata = _prepare_posterior_demand_metadata(
            zone_ids=[1, 2, 3],
            posterior_od_matrix=np.array(
                [
                    [0.0, 1.2, -2.0],
                    [2.5, 0.0, 3.6],
                    [4.49, 5.5, 0.0],
                ]
            ),
            vehicle_type="car",
            period_name="08_00_09_00",
        )

        self.assertEqual(metadata["od_pairs"], [[1, 2], [1, 3], [2, 1], [2, 3], [3, 1], [3, 2]])
        self.assertEqual(metadata["raw_counts"], [1.2, 0.0, 2.5, 3.6, 4.49, 5.5])
        self.assertEqual(metadata["rounded_counts"], [1, 0, 3, 4, 4, 6])
        self.assertEqual(metadata["total_rounded_trips"], 18)

    def test_write_tazrelation_od_skips_zero_count_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "posterior.xml"
            _write_tazrelation_od(
                path=path,
                zone_mappings={
                    1: SumoZoneMapping(1, "taz_1", ["edge_a"], (0.0, 0.0)),
                    2: SumoZoneMapping(2, "taz_2", ["edge_b"], (1.0, 0.0)),
                },
                begin=0,
                end=3600,
                demand_entries=[
                    {"origin": 1, "dest": 2, "rounded_count": 3},
                    {"origin": 2, "dest": 1, "rounded_count": 0},
                ],
            )
            root = ET.parse(path).getroot()
            relations = root.findall("./interval/tazRelation")

            self.assertEqual(len(relations), 1)
            self.assertEqual(relations[0].attrib["from"], "taz_1")
            self.assertEqual(relations[0].attrib["to"], "taz_2")
            self.assertEqual(relations[0].attrib["count"], "3")

    def test_build_h_from_routes_duplicates_same_from_edge_and_keeps_unmapped_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            trips_file = tmp / "trips.xml"
            route_file = tmp / "routes.xml"
            trips_file.write_text(
                textwrap.dedent(
                    """\
                    <routes>
                      <trip id="v1" fromTaz="taz_1" toTaz="taz_2"/>
                      <trip id="v2" fromTaz="taz_1" toTaz="taz_2"/>
                      <trip id="v3" fromTaz="taz_2" toTaz="taz_1"/>
                    </routes>
                    """
                ),
                encoding="utf-8",
            )
            route_file.write_text(
                textwrap.dedent(
                    """\
                    <routes>
                      <vehicle id="v1"><route edges="edge_a edge_c"/></vehicle>
                      <vehicle id="v2"><route edges="edge_c edge_d"/></vehicle>
                      <vehicle id="v3"><route edges="edge_a edge_e"/></vehicle>
                    </routes>
                    """
                ),
                encoding="utf-8",
            )
            gate_mappings = {
                0: SumoGateMapping(0, "cam", "gate_1", from_edge_id="edge_a"),
                1: SumoGateMapping(1, "cam", "gate_2", from_edge_id="edge_a"),
                2: SumoGateMapping(2, "cam", "gate_3"),
            }

            H = _build_h_from_sumo_routes(
                trips_file=trips_file,
                route_file=route_file,
                gate_mappings=gate_mappings,
                od_pairs=[(1, 2), (2, 1)],
                n_gates=3,
            ).toarray()

            np.testing.assert_allclose(H[0], np.array([0.5, 1.0]))
            np.testing.assert_allclose(H[1], np.array([0.5, 1.0]))
            np.testing.assert_allclose(H[2], np.array([0.0, 0.0]))

    def test_build_turning_output_includes_camera_proportions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            route_file = Path(tmpdir) / "routes.xml"
            route_file.write_text(
                textwrap.dedent(
                    """\
                    <routes>
                      <vehicle id="v1"><route edges="edge_a edge_b edge_c"/></vehicle>
                      <vehicle id="v2"><route edges="edge_a edge_b"/></vehicle>
                      <vehicle id="v3"><route edges="edge_x edge_y"/></vehicle>
                    </routes>
                    """
                ),
                encoding="utf-8",
            )
            payload = _build_turning_output(
                route_file=route_file,
                gate_mappings={
                    0: SumoGateMapping(0, "cam_1", "gate_1", from_edge_id="edge_a", to_edge_id="edge_b"),
                    1: SumoGateMapping(1, "cam_1", "gate_2", from_edge_id="edge_b", to_edge_id="edge_c"),
                    2: SumoGateMapping(2, "cam_2", "gate_3", from_edge_id="edge_x", to_edge_id="edge_y"),
                },
                vehicle_type="car",
                period_name="08_00_09_00",
            )

            self.assertEqual(payload["matched_movements"], 3)
            camera_1 = next(camera for camera in payload["cameras"] if camera["camera_id"] == "cam_1")
            self.assertEqual(camera_1["total_count"], 3)
            self.assertEqual(camera_1["n_movements"], 2)
            proportions = {item["gate_id"]: item["proportion"] for item in camera_1["movements"]}
            self.assertAlmostEqual(proportions["gate_1"], 2.0 / 3.0)
            self.assertAlmostEqual(proportions["gate_2"], 1.0 / 3.0)

    def test_analytic_build_assignment_default_backend_still_works(self) -> None:
        G = nx.MultiDiGraph()
        G.add_node("1", x=0.0, y=0.0)
        G.add_node("2", x=1.0, y=0.0)
        G.add_node("3", x=2.0, y=0.0)
        G.add_edge("1", "2", key="a", travel_time=1.0)
        G.add_edge("2", "3", key="b", travel_time=1.0)
        gates = [
            GateInfo("cam", "gate_1", 0, 0.0, 0.0, "south", "east", edge=("1", "2", "a")),
            GateInfo("cam", "gate_2", 1, 0.0, 0.0, "east", "east", edge=("2", "3", "b")),
        ]

        H, od_pairs = build_assignment(
            G,
            zone_node_map={1: "1", 2: "3"},
            gates=gates,
        )

        self.assertEqual(od_pairs, [(1, 2), (2, 1)])
        self.assertEqual(H.shape, (2, 2))
        np.testing.assert_allclose(H.toarray()[:, 0], np.array([1.0, 1.0]))

    def test_run_all_saves_od_then_simulates_posterior_sumo(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = ProjectPaths.from_root(Path(tmpdir))
            events: list[tuple[str, str]] = []
            fake_results = [
                SimpleNamespace(info={"posterior_total_flow": 11.0}, vehicle_type="car"),
                SimpleNamespace(info={"posterior_total_flow": 7.0}, vehicle_type="bus"),
            ]

            def fake_save_od_matrix(result, zone_ids, output_dir):
                events.append(("save_od", result.vehicle_type))
                return {"mean": output_dir / f"od_{result.vehicle_type}.csv"}

            def fake_save_turning_summary(_, vehicle_type, __):
                events.append(("save_turning", vehicle_type))
                return Path(tmpdir) / f"turning_{vehicle_type}.json"

            def fake_simulate_posterior_sumo(*, vehicle_type, od_csv_path, **kwargs):
                events.append(("simulate", vehicle_type))
                self.assertEqual(od_csv_path, paths.output_dir / f"od_{vehicle_type}.csv")
                self.assertEqual(kwargs["zone_ids"], [1, 2])
                return SimpleNamespace(
                    vehicle_type=vehicle_type,
                    mappings=SimpleNamespace(gate_mappings={}),
                )

            def fake_save_observed_turning_sumo(*, gate_mappings, raw_obs, vehicle_type, output_dir):
                events.append(("save_observed_turning", vehicle_type))
                return output_dir / f"turning_{vehicle_type}__observed.json"

            with (
                patch("sumo_part.pipeline.prepare_network", return_value=("network", {1: "n1", 2: "n2"})),
                patch("sumo_part.pipeline.prepare_gates", return_value=(["gate"], {})),
                patch(
                    "sumo_part.pipeline.build_assignment",
                    return_value=(sp.csr_matrix(np.array([[1.0, 0.0]])), [(1, 2), (2, 1)]),
                ),
                patch(
                    "sumo_part.pipeline.prepare_observations",
                    return_value=SimpleNamespace(
                        edge_observations="edge_obs",
                        turning_observations="turning_obs",
                        raw_observations={},
                    ),
                ),
                patch("sumo_part.pipeline.estimate_posterior_od", side_effect=fake_results),
                patch("sumo_part.pipeline.save_od_matrix", side_effect=fake_save_od_matrix),
                patch("sumo_part.pipeline.save_turning_summary", side_effect=fake_save_turning_summary),
                patch("sumo_part.pipeline.simulate_posterior_sumo", side_effect=fake_simulate_posterior_sumo),
                patch("sumo_part.pipeline.save_observed_turning_sumo", side_effect=fake_save_observed_turning_sumo),
            ):
                results = run_all(
                    vehicle_types=["car", "bus"],
                    paths=paths,
                    sumo_config=SumoAssignmentConfig(),
                )

            self.assertEqual(list(results.keys()), ["car", "bus"])
            self.assertEqual(
                events,
                [
                    ("save_od", "car"),
                    ("save_turning", "car"),
                    ("simulate", "car"),
                    ("save_observed_turning", "car"),
                    ("save_od", "bus"),
                    ("save_turning", "bus"),
                    ("simulate", "bus"),
                    ("save_observed_turning", "bus"),
                ],
            )


@unittest.skipIf(_DEPENDENCY_ERROR is not None, f"Optional test dependency is unavailable: {_DEPENDENCY_ERROR}")
class BuildObservedTurningSumoTest(unittest.TestCase):
    def test_basic_counts_and_proportions(self) -> None:
        gate_mappings = {
            0: SumoGateMapping(0, "cam_1", "gate_1", from_edge_id="edge_a", to_edge_id="edge_b"),
            1: SumoGateMapping(1, "cam_1", "gate_2", from_edge_id="edge_c", to_edge_id="edge_d"),
            2: SumoGateMapping(2, "cam_2", "gate_1", from_edge_id="edge_x", to_edge_id="edge_y"),
        }
        raw_obs = {
            "car": {
                0: [100.0, 200.0, 300.0],  # mean = 200
                1: [50.0, 50.0],            # mean = 50
                2: [400.0],                 # mean = 400
            },
        }
        payload = build_observed_turning_sumo(gate_mappings, raw_obs, "car")

        self.assertEqual(payload["vehicle_type"], "car")
        self.assertEqual(payload["matched_movements"], 3)
        self.assertEqual(len(payload["movements"]), 3)
        self.assertEqual(len(payload["cameras"]), 2)

        # Check cam_1: gate_1=200, gate_2=50, total=250
        cam_1 = next(c for c in payload["cameras"] if c["camera_id"] == "cam_1")
        self.assertEqual(cam_1["total_count"], 250.0)
        self.assertEqual(cam_1["n_movements"], 2)
        proportions = {m["gate_id"]: m["proportion"] for m in cam_1["movements"]}
        self.assertAlmostEqual(proportions["gate_1"], 200.0 / 250.0)
        self.assertAlmostEqual(proportions["gate_2"], 50.0 / 250.0)

        # Check cam_2: gate_1=400, proportion=1.0
        cam_2 = next(c for c in payload["cameras"] if c["camera_id"] == "cam_2")
        self.assertEqual(cam_2["total_count"], 400.0)
        self.assertAlmostEqual(cam_2["movements"][0]["proportion"], 1.0)

    def test_skips_unmapped_and_same_edge_gates(self) -> None:
        gate_mappings = {
            0: SumoGateMapping(0, "cam_1", "gate_1", from_edge_id="edge_a", to_edge_id="edge_b"),
            1: SumoGateMapping(1, "cam_1", "gate_2", from_edge_id=None, to_edge_id="edge_d"),       # unmapped from
            2: SumoGateMapping(2, "cam_1", "gate_3", from_edge_id="edge_e", to_edge_id=None),       # unmapped to
            3: SumoGateMapping(3, "cam_1", "gate_4", from_edge_id="edge_f", to_edge_id="edge_f"),   # same edge
        }
        raw_obs = {"car": {0: [100.0], 1: [200.0], 2: [300.0], 3: [400.0]}}
        payload = build_observed_turning_sumo(gate_mappings, raw_obs, "car")

        self.assertEqual(payload["matched_movements"], 1)
        self.assertEqual(payload["movements"][0]["gate_index"], 0)

    def test_gate_with_no_observation_data(self) -> None:
        gate_mappings = {
            0: SumoGateMapping(0, "cam_1", "gate_1", from_edge_id="edge_a", to_edge_id="edge_b"),
        }
        raw_obs = {"car": {}}  # no data for gate 0
        payload = build_observed_turning_sumo(gate_mappings, raw_obs, "car")

        self.assertEqual(payload["matched_movements"], 1)
        self.assertEqual(payload["movements"][0]["count"], 0.0)

    def test_empty_gate_mappings(self) -> None:
        payload = build_observed_turning_sumo({}, {"car": {0: [100.0]}}, "car")
        self.assertEqual(payload["matched_movements"], 0)
        self.assertEqual(payload["movements"], [])
        self.assertEqual(payload["cameras"], [])


@unittest.skipIf(_DEPENDENCY_ERROR is not None, f"Optional test dependency is unavailable: {_DEPENDENCY_ERROR}")
@unittest.skipUnless(os.environ.get("SUMO_HOME"), "SUMO_HOME is not configured in this environment")
class PosteriorSumoIntegrationSmokeTest(unittest.TestCase):
    def test_discover_sumo_tools_for_posterior_pipeline(self) -> None:
        tools, version = discover_sumo_tools()
        self.assertIn("sumo", tools)
        self.assertIn("od2trips", tools)
        self.assertIn("duaIterate.py", tools)
        self.assertTrue(version)


if __name__ == "__main__":
    unittest.main()
