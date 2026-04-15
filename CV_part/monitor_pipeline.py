#!/usr/bin/env python3
"""Periodic monitor for a long-running CV pipeline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import signal
import time


FATAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"traceback",
        r"linalgerror",
        r"file not found",
        r"filenotfounderror",
        r"returned non-zero exit status",
        r"cuda out of memory",
        r"segmentation fault",
        r"\bkilled\b",
        r"runtimeerror",
        r"track.*failed",
        r"跟踪失败",
        r"解码器初始化失败",
    )
]


@dataclass
class Snapshot:
    utc_time: str
    pipeline_pid: int
    pipeline_alive: bool
    tracking_csvs: int
    expected_tracking_csvs: int
    tracking_progress_pct: float
    count_csvs: int
    count_ok_files: int
    gate_summaries: int
    expected_videos: int
    fatal_hits_total: int
    new_fatal_hits: int
    merged_manifest_present: bool
    full_log_size: int
    gpu0_log_size: int
    gpu1_log_size: int


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor a long-running CV pipeline.")
    parser.add_argument("--pipeline-pid", type=int, required=True, help="Top-level pipeline PID.")
    parser.add_argument("--full-root", required=True, help="Full run root.")
    parser.add_argument("--tracking-root", required=True, help="Tracking output root.")
    parser.add_argument("--count-root", required=True, help="Count output root.")
    parser.add_argument("--full-log", required=True, help="Top-level pipeline log path.")
    parser.add_argument("--gpu0-log", required=True, help="GPU0 worker log path.")
    parser.add_argument("--gpu1-log", required=True, help="GPU1 worker log path.")
    parser.add_argument("--video-list", required=True, help="Video list used by the run.")
    parser.add_argument("--poll-seconds", type=int, default=600, help="Polling interval in seconds.")
    return parser.parse_args()


def utc_now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def count_video_list_items(video_list_path: Path) -> int:
    count = 0
    with video_list_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line and not line.startswith("#"):
                count += 1
    return count


def count_manifest_rows(manifest_path: Path) -> int:
    if not manifest_path.exists():
        return 0
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def count_tracking_csvs(tracking_root: Path) -> int:
    total = 0
    for path in tracking_root.rglob("*.csv"):
        if "runs" in path.parts:
            continue
        if path.name.startswith("segment_manifest"):
            continue
        total += 1
    return total


def count_count_outputs(count_root: Path):
    count_csvs = 0
    ok_files = 0
    gate_summaries = 0
    for path in count_root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name.endswith("_Count.csv"):
            count_csvs += 1
        elif name.endswith("_Count.csv.ok"):
            ok_files += 1
        elif name.endswith("_gate_summary.csv"):
            gate_summaries += 1
    return count_csvs, ok_files, gate_summaries


def read_new_fatal_hits(log_path: Path, offset: int):
    if not log_path.exists():
        return offset, []

    size = log_path.stat().st_size
    if offset > size:
        offset = 0

    hits = []
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(offset)
        for line in f:
            text = line.rstrip()
            if any(pattern.search(text) for pattern in FATAL_PATTERNS):
                hits.append(text)
        new_offset = f.tell()
    return new_offset, hits


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_monitor_log(path: Path, line: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_snapshot(args, expected_tracking_csvs: int, expected_videos: int, fatal_hits_total: int, new_fatal_hits: int):
    tracking_root = Path(args.tracking_root)
    count_root = Path(args.count_root)
    full_log = Path(args.full_log)
    gpu0_log = Path(args.gpu0_log)
    gpu1_log = Path(args.gpu1_log)

    tracking_csvs = count_tracking_csvs(tracking_root)
    count_csvs, count_ok_files, gate_summaries = count_count_outputs(count_root)
    tracking_progress_pct = (
        round(tracking_csvs / expected_tracking_csvs * 100.0, 2)
        if expected_tracking_csvs > 0
        else 0.0
    )

    return Snapshot(
        utc_time=utc_now_str(),
        pipeline_pid=args.pipeline_pid,
        pipeline_alive=process_alive(args.pipeline_pid),
        tracking_csvs=tracking_csvs,
        expected_tracking_csvs=expected_tracking_csvs,
        tracking_progress_pct=tracking_progress_pct,
        count_csvs=count_csvs,
        count_ok_files=count_ok_files,
        gate_summaries=gate_summaries,
        expected_videos=expected_videos,
        fatal_hits_total=fatal_hits_total,
        new_fatal_hits=new_fatal_hits,
        merged_manifest_present=(tracking_root / "segment_manifest.csv").exists(),
        full_log_size=full_log.stat().st_size if full_log.exists() else 0,
        gpu0_log_size=gpu0_log.stat().st_size if gpu0_log.exists() else 0,
        gpu1_log_size=gpu1_log.stat().st_size if gpu1_log.exists() else 0,
    )


def write_summary(summary_path: Path, snapshot: Snapshot, args, recent_fatal_hits):
    status = "completed" if not snapshot.pipeline_alive else "running"
    if recent_fatal_hits:
        status += " with fatal-log-hits"

    lines = [
        "# Full Pipeline Monitor Summary",
        "",
        f"- Status: {status}",
        f"- Snapshot time: {snapshot.utc_time}",
        f"- Pipeline PID: {snapshot.pipeline_pid}",
        f"- Pipeline alive: {snapshot.pipeline_alive}",
        f"- Tracking CSVs: {snapshot.tracking_csvs}/{snapshot.expected_tracking_csvs} ({snapshot.tracking_progress_pct}%)",
        f"- Count CSVs: {snapshot.count_csvs}/{snapshot.expected_tracking_csvs}",
        f"- Count OK files: {snapshot.count_ok_files}/{snapshot.expected_tracking_csvs}",
        f"- Gate summaries: {snapshot.gate_summaries}/{snapshot.expected_videos}",
        f"- Merged manifest present: {snapshot.merged_manifest_present}",
        f"- Fatal hits seen: {snapshot.fatal_hits_total}",
        "",
        "## Paths",
        "",
        f"- Full root: `{args.full_root}`",
        f"- Tracking root: `{args.tracking_root}`",
        f"- Count root: `{args.count_root}`",
        f"- Full log: `{args.full_log}`",
        f"- GPU0 log: `{args.gpu0_log}`",
        f"- GPU1 log: `{args.gpu1_log}`",
    ]

    if recent_fatal_hits:
        lines.extend([
            "",
            "## Recent Fatal Hits",
            "",
        ])
        for hit in recent_fatal_hits[-20:]:
            lines.append(f"- {hit}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    full_root = Path(args.full_root)
    full_root.mkdir(parents=True, exist_ok=True)
    monitor_dir = full_root / "monitor"
    monitor_log_path = monitor_dir / "monitor.log"
    status_json_path = monitor_dir / "latest_status.json"
    summary_path = monitor_dir / "final_summary.md"
    state_path = monitor_dir / "monitor_state.json"

    gpu0_manifest = Path(args.tracking_root) / "runs" / "dual_gpu_main" / "segment_manifest_gpu0.csv"
    gpu1_manifest = Path(args.tracking_root) / "runs" / "dual_gpu_main" / "segment_manifest_gpu1.csv"
    expected_tracking_csvs = count_manifest_rows(gpu0_manifest) + count_manifest_rows(gpu1_manifest)
    expected_videos = count_video_list_items(Path(args.video_list))

    offsets = {
        args.full_log: 0,
        args.gpu0_log: 0,
        args.gpu1_log: 0,
    }
    fatal_hits_total = 0
    recent_fatal_hits = []

    append_monitor_log(monitor_log_path, f"[{utc_now_str()}] monitor started for pid={args.pipeline_pid}")

    while True:
        new_fatal_hits = []
        for log_path_str in (args.full_log, args.gpu0_log, args.gpu1_log):
            new_offset, hits = read_new_fatal_hits(Path(log_path_str), offsets[log_path_str])
            offsets[log_path_str] = new_offset
            if hits:
                for hit in hits:
                    tagged = f"{Path(log_path_str).name}: {hit}"
                    new_fatal_hits.append(tagged)
                    recent_fatal_hits.append(tagged)

        fatal_hits_total += len(new_fatal_hits)
        snapshot = build_snapshot(
            args,
            expected_tracking_csvs=expected_tracking_csvs,
            expected_videos=expected_videos,
            fatal_hits_total=fatal_hits_total,
            new_fatal_hits=len(new_fatal_hits),
        )

        append_monitor_log(
            monitor_log_path,
            (
                f"[{snapshot.utc_time}] alive={snapshot.pipeline_alive} "
                f"tracking={snapshot.tracking_csvs}/{snapshot.expected_tracking_csvs} "
                f"count={snapshot.count_csvs}/{snapshot.expected_tracking_csvs} "
                f"ok={snapshot.count_ok_files}/{snapshot.expected_tracking_csvs} "
                f"summaries={snapshot.gate_summaries}/{snapshot.expected_videos} "
                f"new_fatal_hits={snapshot.new_fatal_hits}"
            ),
        )
        for hit in new_fatal_hits:
            append_monitor_log(monitor_log_path, f"  FATAL-HIT {hit}")

        write_json(status_json_path, asdict(snapshot))
        write_json(
            state_path,
            {
                "snapshot": asdict(snapshot),
                "recent_fatal_hits": recent_fatal_hits[-50:],
                "offsets": offsets,
            },
        )
        write_summary(summary_path, snapshot, args, recent_fatal_hits)

        if not snapshot.pipeline_alive:
            append_monitor_log(monitor_log_path, f"[{utc_now_str()}] pipeline no longer alive, monitor exiting")
            return

        time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    main()
