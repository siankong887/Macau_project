from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    output_dir: Path
    sumo_output_dir: Path
    network_graphml: Path
    network_osm: Path
    network_net_xml: Path
    zones_geojson: Path
    zones_esri: Path
    zones_taz_xml: Path
    zone_nodes_json: Path
    observation_json: Path
    camera_config_json: Path
    speed_mapping_csv: Path
    gate_edge_mapping_json: Path

    @classmethod
    def from_root(cls, root: Path | str) -> "ProjectPaths":
        #.resolve()方法用于解析路径，不管是相对路径还是带"..."这种省略号的路径都解析成绝对路径
        root = Path(root).resolve()
        data_dir = root / "data"
        output_dir = root / "output"
        return cls(
            root=root,
            data_dir=data_dir,
            output_dir=output_dir,
            sumo_output_dir=output_dir / "sumo",
            network_graphml=data_dir / "network" / "macau_drive.graphml",
            network_osm=data_dir / "network" / "macau-260220.osm",
            network_net_xml=data_dir / "network" / "macau_drive.net.xml",
            zones_geojson=data_dir / "zones" / "macau_zones_23.geojson",
            zones_esri=data_dir / "zones" / "macau_23_districts_esri.json",
            zones_taz_xml=data_dir / "zones" / "macau_zones_23.taz.xml",
            zone_nodes_json=data_dir / "zones" / "zone_centroids.json",
            observation_json=data_dir / "observations" / "time_limit.json",
            camera_config_json=data_dir / "config" / "a1_copy_2.json",
            speed_mapping_csv=data_dir / "speeds" / "speed_mapping.csv",
            gate_edge_mapping_json=output_dir / "gate_edge_mapping.json",
        )

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sumo_output_dir.mkdir(parents=True, exist_ok=True)

    def sumo_period_dir(self, period_name: str) -> Path:
        return self.sumo_output_dir / period_name

    def sumo_vehicle_dir(self, period_name: str, vehicle_type: str) -> Path:
        return self.sumo_period_dir(period_name) / vehicle_type


def default_paths() -> ProjectPaths:
    return ProjectPaths.from_root(Path(__file__).resolve().parent.parent)
