#!/usr/bin/env python3
"""Post-process CV tracking/count outputs into speed proxy CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
from collections import defaultdict
from pathlib import Path
from typing import Any

from cv_paths import CVPaths
from VechilCountCPU import (
    extract_cam_name_from_csv,
    judge_slide,
    load_manifest_lookup,
    on_segment,
    parse_segment_name,
)


PATHS = CVPaths.from_file(__file__)
CLS_MAP = {
    0: "car",
    1: "bus",
    2: "truck",
    3: "motorcycle",
}
FRAME_DIFF_LIMIT = 12 * 60 * 30
DEFAULT_VEHICLE_LENGTH_DM_JSON = "[45,108,65,20]"

MAIN_HEADER = [
    "video_name",
    "cam_name",
    "segment_name",
    "start_time",
    "end_time",
    "start_frame",
    "end_frame",
    "duration_sec",
    "gate_index",
    "cls_id",
    "cls_name",
    "track_id",
    "count_frame_id",
    "frame_first",
    "frame_second",
    "frame_diff",
    "speed",
]
ANOMALY_HEADER = [
    "video_name",
    "cam_name",
    "segment_name",
    "gate_index",
    "cls_id",
    "cls_name",
    "track_id",
    "count_frame_id",
    "reason",
    "detail",
]
SUMMARY_HEADER = [
    "video_name",
    "cam_name",
    "segment_name",
    "start_time",
    "end_time",
    "start_frame",
    "end_frame",
    "duration_sec",
    "gate_index",
    "cls_id",
    "cls_name",
    "sample_count",
    "mean_frame_diff",
    "speed",
    "frame_diff_list",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate speed proxy CSVs from tracking/count outputs."
    )
    parser.add_argument(
        "--tracking-root",
        default=str(PATHS.tracking_root),
        help="Root directory that contains per-camera tracking CSV folders.",
    )
    parser.add_argument(
        "--count-root",
        default=str(PATHS.count_root),
        help="Root directory that contains per-camera count CSV folders.",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional segment_manifest.csv path. Defaults to <tracking-root>/segment_manifest.csv.",
    )
    parser.add_argument(
        "--source-json",
        default=str(PATHS.source_json_path),
        help="Path to a1_copy_2_copy.json.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help='Output root. Defaults to Path(count_root).parent / "speed_proxy".',
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=2,
        help="Worker process count for segment-level processing.",
    )
    parser.add_argument(
        "--vehicle-length-dm",
        default=DEFAULT_VEHICLE_LENGTH_DM_JSON,
        help=(
            "Vehicle length JSON list string. "
            f"Defaults to {DEFAULT_VEHICLE_LENGTH_DM_JSON}."
        ),
    )
    return parser.parse_args()


def parse_vehicle_lengths(raw_value: str | None) -> list[float] | None:
    if raw_value is None:
        return None
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --vehicle-length-dm JSON: {exc}") from exc
    if not isinstance(value, list) or len(value) != 4:
        raise SystemExit("--vehicle-length-dm must be a JSON list of length 4.")
    parsed: list[float] = []
    for idx, item in enumerate(value):
        if not isinstance(item, (int, float)):
            raise SystemExit(
                f"--vehicle-length-dm item at index {idx} is not numeric: {item!r}"
            )
        parsed.append(float(item))
    return parsed


def parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_decimal(value: float) -> str:
    return f"{value:.3f}"


def cls_name(cls_id: int) -> str:
    return CLS_MAP.get(int(cls_id), f"unknown_{cls_id}")


def parse_line_segment(raw_line: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(raw_line, list) or len(raw_line) != 2:
        return None
    pt1, pt2 = raw_line
    if (
        not isinstance(pt1, list)
        or not isinstance(pt2, list)
        or len(pt1) != 2
        or len(pt2) != 2
    ):
        return None
    try:
        return (
            float(pt1[0]),
            float(pt1[1]),
            float(pt2[0]),
            float(pt2[1]),
        )
    except (TypeError, ValueError):
        return None


def load_speed_gate_config(source_json_path: Path) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(source_json_path.read_text(encoding="utf-8"))
    result: dict[str, list[dict[str, Any]]] = {}
    for cam in data.get("list", []):
        cam_name = str(cam.get("camera", "")).strip()
        if not cam_name:
            continue
        orientations = cam.get("gate_orientation", [])
        gates = []
        for gate_pos, gate in enumerate(cam.get("gate", []), start=1):
            raw_lines = gate.get("line", [])
            config: dict[str, Any] = {
                "gate_index": gate_pos,
                "valid": False,
                "detail": "",
                "orientation": None,
                "out_line": None,
            }
            if gate_pos - 1 >= len(orientations):
                config["detail"] = "missing gate_orientation"
                gates.append(config)
                continue

            orientation = orientations[gate_pos - 1]
            if orientation not in (-1, 1):
                config["detail"] = f"unsupported gate_orientation={orientation!r}"
                gates.append(config)
                continue

            if not isinstance(raw_lines, list) or len(raw_lines) not in (1, 2):
                config["detail"] = f"unsupported line length={len(raw_lines) if isinstance(raw_lines, list) else 'invalid'}"
                gates.append(config)
                continue

            line_index = 1 if len(raw_lines) == 2 else 0
            out_line = parse_line_segment(raw_lines[line_index])
            if out_line is None:
                config["detail"] = "invalid out line shape"
                gates.append(config)
                continue

            config["valid"] = True
            config["orientation"] = int(orientation)
            config["out_line"] = out_line
            gates.append(config)
        result[cam_name] = gates
    return result


def build_segment_meta(
    video_name: str,
    cam_name: str,
    segment_name: str,
    tracking_csv_path: Path,
    count_csv_path: Path,
    manifest_lookup: dict[str, dict[str, str]],
) -> dict[str, Any]:
    meta = manifest_lookup.get(segment_name, {})
    parsed_name = parse_segment_name(segment_name) or {}
    return {
        "video_name": str(meta.get("video_name", video_name)).strip() or video_name,
        "cam_name": cam_name,
        "segment_name": segment_name,
        "tracking_csv_path": str(tracking_csv_path),
        "count_csv_path": str(count_csv_path),
        "start_time": str(meta.get("start_time", parsed_name.get("start_time", ""))),
        "end_time": str(meta.get("end_time", parsed_name.get("end_time", ""))),
        "start_frame": parse_int(meta.get("start_frame", "")),
        "end_frame": parse_int(meta.get("end_frame", "")),
        "duration_sec": parse_float(meta.get("duration_sec", 0.0)),
    }


def build_tasks(
    tracking_root: Path,
    count_root: Path,
    manifest_lookup: dict[str, dict[str, str]],
    gate_config: dict[str, list[dict[str, Any]]],
    output_root: Path,
) -> list[dict[str, Any]]:
    tasks = []
    for count_csv_path in sorted(count_root.rglob("*_Count.csv")):
        if count_csv_path.name.endswith("_gate_summary.csv"):
            continue
        video_name = count_csv_path.parent.name
        base_name = count_csv_path.stem
        segment_name = base_name[:-6] if base_name.endswith("_Count") else base_name
        tracking_csv_path = tracking_root / video_name / f"{segment_name}.csv"
        cam_name = extract_cam_name_from_csv(tracking_csv_path.name) or f"cam_{video_name.replace('_', '')}"
        tasks.append(
            {
                "meta": build_segment_meta(
                    video_name,
                    cam_name,
                    segment_name,
                    tracking_csv_path,
                    count_csv_path,
                    manifest_lookup,
                ),
                "gate_configs": gate_config.get(cam_name, []),
                "main_output_path": str(output_root / video_name / f"{segment_name}_SpeedProxy.csv"),
                "anomaly_output_path": str(
                    output_root / video_name / f"{segment_name}_SpeedProxy_Anomalies.csv"
                ),
            }
        )
    return tasks


def read_count_events(count_csv_path: Path) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    events: list[dict[str, int]] = []
    duplicates: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    with count_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                event = {
                    "frame_id": int(float(row["frame_id"])),
                    "track_id": int(float(row["track_id"])),
                    "cls_id": int(float(row["cls"])),
                    "gate_index": int(float(row["gate_index"])),
                }
            except (KeyError, TypeError, ValueError):
                continue
            key = (event["track_id"], event["gate_index"])
            if key in seen:
                duplicates.append(event)
                continue
            seen.add(key)
            events.append(event)
    return events, duplicates


def read_tracking_rows(tracking_csv_path: Path, track_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    rows_by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not tracking_csv_path.exists():
        return rows_by_track
    with tracking_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                track_id = int(float(row["track_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            if track_id not in track_ids:
                continue
            try:
                parsed = {
                    "frame_id": int(float(row["frame_id"])),
                    "track_id": track_id,
                    "cls_id": int(float(row["cls"])),
                    "center_x": float(row["center_x"]),
                    "center_y": float(row["center_y"]),
                    "width": float(row["width"]),
                    "height": float(row["height"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            rows_by_track[track_id].append(parsed)
    for track_rows in rows_by_track.values():
        track_rows.sort(key=lambda item: item["frame_id"])
    return rows_by_track


def make_anomaly_row(
    meta: dict[str, Any],
    event: dict[str, int] | None,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    cls_id = event["cls_id"] if event is not None else ""
    return {
        "video_name": meta["video_name"],
        "cam_name": meta["cam_name"],
        "segment_name": meta["segment_name"],
        "gate_index": event["gate_index"] if event is not None else "",
        "cls_id": cls_id,
        "cls_name": cls_name(cls_id) if cls_id != "" else "",
        "track_id": event["track_id"] if event is not None else "",
        "count_frame_id": event["frame_id"] if event is not None else "",
        "reason": reason,
        "detail": detail,
    }


def build_main_row(
    meta: dict[str, Any],
    event: dict[str, int],
    frame_first: int,
    frame_second: int,
    speed_value: str,
) -> dict[str, Any]:
    return {
        "video_name": meta["video_name"],
        "cam_name": meta["cam_name"],
        "segment_name": meta["segment_name"],
        "start_time": meta["start_time"],
        "end_time": meta["end_time"],
        "start_frame": meta["start_frame"],
        "end_frame": meta["end_frame"],
        "duration_sec": meta["duration_sec"],
        "gate_index": event["gate_index"],
        "cls_id": event["cls_id"],
        "cls_name": cls_name(event["cls_id"]),
        "track_id": event["track_id"],
        "count_frame_id": event["frame_id"],
        "frame_first": frame_first,
        "frame_second": frame_second,
        "frame_diff": frame_second - frame_first,
        "speed": speed_value,
    }


def compute_speed_kmh(cls_id: int, frame_diff: int, vehicle_lengths_dm: list[float] | None) -> str:
    if vehicle_lengths_dm is None:
        return "N/A"
    if frame_diff <= 0:
        return "N/A"
    vehicle_length_m = vehicle_lengths_dm[int(cls_id)] / 10.0
    speed_kmh = vehicle_length_m / (frame_diff / 30.0) * 3.6
    return format_decimal(speed_kmh)


def proxy_points_for_row(
    row: dict[str, Any],
    out_line: tuple[float, float, float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    cx = row["center_x"]
    cy = row["center_y"]
    width = row["width"]
    height = row["height"]
    x1, y1, x2, y2 = out_line
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) >= abs(dy):
        return ((cx, cy - height / 2.0), (cx, cy + height / 2.0))
    return ((cx - width / 2.0, cy), (cx + width / 2.0, cy))


def point_crosses_out_line(
    last_point: tuple[float, float],
    current_point: tuple[float, float],
    out_line: tuple[float, float, float, float],
    orientation: int,
) -> bool:
    current_products = judge_slide(current_point, [out_line])
    last_products = judge_slide(last_point, [out_line])
    if not current_products or not last_products:
        return False

    p_product = current_products[0]
    l_product = last_products[0]
    x1, y1, x2, y2 = out_line

    if p_product * l_product < 0:
        a1, a2 = current_point
        b1, b2 = last_point
        ab = (b1 - a1, b2 - a2)
        ap_1 = (x1 - a1, y1 - a2)
        ap_2 = (x2 - a1, y2 - a2)
        cross_product1 = ab[0] * ap_1[1] - ab[1] * ap_1[0]
        cross_product2 = ab[0] * ap_2[1] - ab[1] * ap_2[0]
        if cross_product1 * cross_product2 < 0:
            if l_product < 0 and p_product > 0:
                return orientation == -1
            if l_product > 0 and p_product < 0:
                return orientation == 1
        return False

    if p_product == 0.0:
        if on_segment((x1, y1), (x2, y2), current_point) and l_product != 0.0:
            if l_product < 0:
                return orientation == -1
            if l_product > 0:
                return orientation == 1
        return False

    return False


def collect_crossing_frames(
    track_rows: list[dict[str, Any]],
    out_line: tuple[float, float, float, float],
    orientation: int,
) -> tuple[list[int], list[int]]:
    side_a: list[int] = []
    side_b: list[int] = []
    prev_a = None
    prev_b = None
    for row in track_rows:
        point_a, point_b = proxy_points_for_row(row, out_line)
        if prev_a is not None and prev_b is not None:
            if point_crosses_out_line(prev_a, point_a, out_line, orientation):
                side_a.append(row["frame_id"])
            if point_crosses_out_line(prev_b, point_b, out_line, orientation):
                side_b.append(row["frame_id"])
        prev_a = point_a
        prev_b = point_b
    return sorted(set(side_a)), sorted(set(side_b))


def pick_crossing_pair(
    side_a: list[int],
    side_b: list[int],
    count_frame_id: int,
) -> tuple[tuple[int, int] | None, str | None, str]:
    best_valid: tuple[int, float, int, int] | None = None
    best_long: tuple[int, float, int, int] | None = None
    saw_same_frame = False

    for frame_a in side_a:
        for frame_b in side_b:
            frame_first = min(frame_a, frame_b)
            frame_second = max(frame_a, frame_b)
            if not (frame_first <= count_frame_id <= frame_second):
                continue
            if frame_a == frame_b:
                saw_same_frame = True
                continue

            frame_diff = frame_second - frame_first
            center_dist = abs(((frame_first + frame_second) / 2.0) - count_frame_id)
            candidate = (frame_diff, center_dist, frame_first, frame_second)
            if frame_diff > FRAME_DIFF_LIMIT:
                if best_long is None or candidate < best_long:
                    best_long = candidate
                continue
            if best_valid is None or candidate < best_valid:
                best_valid = candidate

    if best_valid is not None:
        return (best_valid[2], best_valid[3]), None, ""
    if saw_same_frame:
        return None, "same_frame_pair", "both proxy points crossed on the same frame"
    if best_long is not None:
        detail = (
            f"frame_first={best_long[2]}, frame_second={best_long[3]}, "
            f"frame_diff={best_long[0]} exceeds limit {FRAME_DIFF_LIMIT}"
        )
        return None, "frame_diff_exceeds_limit", detail
    if side_a and side_b:
        return None, "count_anchor_mismatch", "no proxy pair brackets count_frame_id"
    if side_a or side_b:
        return None, "single_side_only", "only one proxy side crossed the out line"
    return None, "count_anchor_mismatch", "no proxy crossings found for count event"


def write_csv(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def build_summary_rows(
    meta: dict[str, Any],
    main_rows: list[dict[str, Any]],
    vehicle_lengths_dm: list[float] | None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in main_rows:
        grouped[(int(row["gate_index"]), int(row["cls_id"]))].append(row)

    summary_rows = []
    for (gate_index, cls_id), rows in sorted(grouped.items()):
        frame_diffs = [int(row["frame_diff"]) for row in rows]
        mean_frame_diff = sum(frame_diffs) / len(frame_diffs)
        if vehicle_lengths_dm is None:
            speed_value = "N/A"
        else:
            speeds = [float(row["speed"]) for row in rows if row["speed"] != "N/A"]
            speed_value = format_decimal(sum(speeds) / len(speeds)) if speeds else "N/A"
        summary_rows.append(
            {
                "video_name": meta["video_name"],
                "cam_name": meta["cam_name"],
                "segment_name": meta["segment_name"],
                "start_time": meta["start_time"],
                "end_time": meta["end_time"],
                "start_frame": meta["start_frame"],
                "end_frame": meta["end_frame"],
                "duration_sec": meta["duration_sec"],
                "gate_index": gate_index,
                "cls_id": cls_id,
                "cls_name": cls_name(cls_id),
                "sample_count": len(rows),
                "mean_frame_diff": format_decimal(mean_frame_diff),
                "speed": speed_value,
                "frame_diff_list": json.dumps(frame_diffs, ensure_ascii=False),
            }
        )
    return summary_rows


def process_segment_task(task: dict[str, Any], vehicle_lengths_dm: list[float] | None) -> dict[str, Any]:
    meta = task["meta"]
    count_csv_path = Path(meta["count_csv_path"])
    tracking_csv_path = Path(meta["tracking_csv_path"])
    main_output_path = Path(task["main_output_path"])
    anomaly_output_path = Path(task["anomaly_output_path"])

    count_events, duplicate_events = read_count_events(count_csv_path)
    track_rows_by_id = read_tracking_rows(
        tracking_csv_path,
        {event["track_id"] for event in count_events},
    )

    main_rows: list[dict[str, Any]] = []
    anomaly_rows: list[dict[str, Any]] = []

    for duplicate_event in duplicate_events:
        anomaly_rows.append(
            make_anomaly_row(meta, duplicate_event, "duplicate_count_event", "duplicate (track_id, gate_index) in count CSV")
        )

    gate_configs: list[dict[str, Any]] = task["gate_configs"]
    for event in count_events:
        gate_index = int(event["gate_index"])
        if gate_index <= 0 or gate_index > len(gate_configs):
            anomaly_rows.append(
                make_anomaly_row(
                    meta,
                    event,
                    "invalid_gate_config",
                    f"gate_index {gate_index} not found for camera {meta['cam_name']}",
                )
            )
            continue

        gate_config = gate_configs[gate_index - 1]
        if not gate_config.get("valid", False):
            anomaly_rows.append(
                make_anomaly_row(
                    meta,
                    event,
                    "invalid_gate_config",
                    str(gate_config.get("detail", "unknown gate config error")),
                )
            )
            continue

        track_rows = track_rows_by_id.get(event["track_id"], [])
        if len(track_rows) < 2:
            anomaly_rows.append(
                make_anomaly_row(meta, event, "missing_tracking_rows", "track rows missing or insufficient")
            )
            continue

        invalid_dim_row = next(
            (row for row in track_rows if row["width"] <= 0 or row["height"] <= 0),
            None,
        )
        if invalid_dim_row is not None:
            detail = (
                f"frame_id={invalid_dim_row['frame_id']}, width={invalid_dim_row['width']}, "
                f"height={invalid_dim_row['height']}"
            )
            anomaly_rows.append(
                make_anomaly_row(meta, event, "non_positive_bbox_dim", detail)
            )
            continue

        out_line = gate_config["out_line"]
        orientation = int(gate_config["orientation"])
        side_a, side_b = collect_crossing_frames(track_rows, out_line, orientation)
        pair, reason, detail = pick_crossing_pair(side_a, side_b, event["frame_id"])
        if pair is None:
            anomaly_rows.append(make_anomaly_row(meta, event, str(reason), detail))
            continue

        frame_first, frame_second = pair
        speed_value = compute_speed_kmh(event["cls_id"], frame_second - frame_first, vehicle_lengths_dm)
        main_rows.append(build_main_row(meta, event, frame_first, frame_second, speed_value))

    summary_rows = build_summary_rows(meta, main_rows, vehicle_lengths_dm)
    write_csv(main_output_path, MAIN_HEADER, main_rows)
    write_csv(anomaly_output_path, ANOMALY_HEADER, anomaly_rows)
    return {
        "video_name": meta["video_name"],
        "summary_rows": summary_rows,
    }


def write_video_summaries(output_root: Path, per_video_rows: dict[str, list[dict[str, Any]]]) -> None:
    for video_name, rows in sorted(per_video_rows.items()):
        summary_path = output_root / video_name / f"{video_name}_speed_proxy_summary.csv"
        sorted_rows = sorted(
            rows,
            key=lambda row: (
                parse_int(row["start_frame"]),
                parse_int(row["gate_index"]),
                parse_int(row["cls_id"]),
                row["segment_name"],
            ),
        )
        write_csv(summary_path, SUMMARY_HEADER, sorted_rows)


def main() -> None:
    args = parse_args()
    tracking_root = Path(args.tracking_root).expanduser().resolve()
    count_root = Path(args.count_root).expanduser().resolve()
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path
        else tracking_root / "segment_manifest.csv"
    )
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else count_root.parent / "speed_proxy"
    )
    vehicle_lengths_dm = parse_vehicle_lengths(args.vehicle_length_dm)

    if not tracking_root.is_dir():
        raise FileNotFoundError(f"Tracking root path does not exist: {tracking_root}")
    if not count_root.is_dir():
        raise FileNotFoundError(f"Count root path does not exist: {count_root}")

    manifest_lookup = load_manifest_lookup(str(manifest_path))
    gate_config = load_speed_gate_config(Path(args.source_json).expanduser().resolve())
    tasks = build_tasks(tracking_root, count_root, manifest_lookup, gate_config, output_root)
    if not tasks:
        print("No *_Count.csv files found. Nothing to process.")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    per_video_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        per_video_rows.setdefault(task["meta"]["video_name"], [])

    worker_count = max(1, int(args.processes))
    ctx = multiprocessing.get_context("spawn")
    job_args = [(task, vehicle_lengths_dm) for task in tasks]
    if worker_count == 1:
        results = [process_segment_task(task, vehicle_lengths_dm) for task in tasks]
    else:
        with ctx.Pool(worker_count) as pool:
            results = pool.starmap(process_segment_task, job_args)

    for result in results:
        per_video_rows[result["video_name"]].extend(result["summary_rows"])

    write_video_summaries(output_root, per_video_rows)
    print(f"Speed proxy output root: {output_root}")
    print(f"Processed segments      : {len(tasks)}")


if __name__ == "__main__":
    main()
