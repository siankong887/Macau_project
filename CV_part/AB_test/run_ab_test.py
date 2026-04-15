#!/usr/bin/env python3
"""Run reproducible A/B tests on fixed CV segments.

This runner detects each segment once, saves the detection cache, then replays
the same detections through multiple tracker backends before running the count
pipeline on each backend's tracking CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
AB_TEST_DIR = SCRIPT_PATH.parent
CV_PART_DIR = AB_TEST_DIR.parent
REPO_ROOT = CV_PART_DIR.parent
if str(CV_PART_DIR) not in sys.path:
    sys.path.insert(0, str(CV_PART_DIR))

from cv_paths import CVPaths
from DetectionTrackingWithGPU import track_segment
import run_peak_hours as rph


PATHS = CVPaths.from_file(__file__)
DEFAULT_SEGMENTS_JSON = AB_TEST_DIR / "segments_2x15m.json"
DEFAULT_BACKENDS = ("bytetrack", "oasort")
MANIFEST_HEADER = [
    "video_name",
    "cam_key",
    "segment_name",
    "video_path",
    "start_time",
    "end_time",
    "start_frame",
    "end_frame",
    "duration_sec",
    "is_tail",
    "status",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run A/B tests on selected CV segments.")
    parser.add_argument(
        "--segments-json",
        default=str(DEFAULT_SEGMENTS_JSON),
        help="JSON file that lists the selected segments.",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Optional output root. Defaults to CV_part/AB_test/runs/<timestamp>.",
    )
    parser.add_argument(
        "--model-path",
        default=str(PATHS.model_pt_path),
        help="Model .pt path used for detection.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_BACKENDS),
        help="Tracker backends to compare.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.getenv("BATCH_SIZE", "512")),
        help="Detection batch size.",
    )
    parser.add_argument(
        "--count-processes",
        type=int,
        default=4,
        help="Worker process count for VechilCountCPU.py.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite cached detections and tracking/count outputs if they already exist.",
    )
    return parser.parse_args()


def load_segments(segments_json_path: Path):
    data = json.loads(segments_json_path.read_text(encoding="utf-8"))
    segments = []
    for idx, raw in enumerate(data, start=1):
        video_path = Path(raw["video_path"]).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Segment #{idx} video not found: {video_path}")

        video_name = video_path.stem
        start_time = str(raw["start_time"]).strip()
        end_time = str(raw["end_time"]).strip()
        start_sec = rph.time_str_to_seconds(start_time)
        end_sec = rph.time_str_to_seconds(end_time)
        if end_sec <= start_sec:
            raise ValueError(
                f"Segment #{idx} has non-positive duration: {video_name} {start_time} -> {end_time}"
            )

        segment_name = f"{video_name}_{rph.time_str_to_label(start_time)}__{rph.time_str_to_label(end_time)}"
        segments.append({
            "index": idx,
            "video_path": str(video_path),
            "video_name": video_name,
            "cam_key": rph.video_name_to_cam_key(video_name),
            "start_time": start_time,
            "end_time": end_time,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_frame": int(round(start_sec * rph.FPS)),
            "end_frame": int(round(end_sec * rph.FPS)),
            "duration_sec": round(end_sec - start_sec, 3),
            "segment_name": segment_name,
            "label": str(raw.get("label", "")).strip(),
            "note": str(raw.get("note", "")).strip(),
        })
    return segments


def default_output_root():
    run_tag = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return AB_TEST_DIR / "runs" / run_tag


def ensure_output_layout(output_root: Path, backends):
    output_root.mkdir(parents=True, exist_ok=True)
    layout = {
        "root": output_root,
        "detections": output_root / "detections",
        "meta": output_root / "meta",
        "backends": {},
    }
    layout["detections"].mkdir(parents=True, exist_ok=True)
    layout["meta"].mkdir(parents=True, exist_ok=True)

    for backend in backends:
        backend_root = output_root / backend
        tracking_root = backend_root / "tracking"
        count_root = backend_root / "count"
        backend_layout = {
            "root": backend_root,
            "tracking_root": tracking_root,
            "count_root": count_root,
            "manifest_path": backend_root / "segment_manifest.csv",
        }
        tracking_root.mkdir(parents=True, exist_ok=True)
        count_root.mkdir(parents=True, exist_ok=True)
        layout["backends"][backend] = backend_layout

    return layout


def write_manifest(manifest_path: Path, rows):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in MANIFEST_HEADER})


def build_manifest_rows(segments):
    rows = []
    for segment in segments:
        rows.append({
            "video_name": segment["video_name"],
            "cam_key": segment["cam_key"],
            "segment_name": segment["segment_name"],
            "video_path": segment["video_path"],
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "start_frame": segment["start_frame"],
            "end_frame": segment["end_frame"],
            "duration_sec": segment["duration_sec"],
            "is_tail": False,
            "status": "planned",
        })
    return rows


def _decode_segment(decoder, runtime, model, total_frames, batch_size):
    torch = runtime["torch"]
    decode_and_stack = runtime["_decode_and_stack"]
    non_max_suppression = runtime["non_max_suppression"]

    decode_q = queue.Queue(maxsize=2)

    def safe_decode():
        try:
            decode_and_stack(decoder, batch_size, total_frames, decode_q)
        except Exception as exc:
            print(f"    解码线程异常: {exc}", flush=True)
            decode_q.put(None)

    decode_thread = threading.Thread(target=safe_decode, daemon=True)
    decode_thread.start()

    pending_tensors = []
    frame_ids = []
    counts = []
    process_count = 0
    segment_start = time.time()

    while True:
        item = decode_q.get()
        if item is None:
            break

        batch, got = item
        input_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        input_batch = batch.to(dtype=input_dtype).div_(255.0)

        with torch.inference_mode():
            preds = model(input_batch)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            dets_list = non_max_suppression(preds, conf_thres=0.1, iou_thres=0.7)

        for batch_index, det in enumerate(dets_list):
            frame_ids.append(process_count + batch_index)
            if det is not None and len(det) > 0:
                pending_tensors.append(det)
                counts.append(int(len(det)))
            else:
                counts.append(0)

        process_count += got
        if process_count and process_count % (batch_size * 20) == 0:
            elapsed = time.time() - segment_start
            fps = process_count / elapsed if elapsed > 0 else 0.0
            print(f"    检测进度 {process_count}/{total_frames} 帧, 平均 {fps:.0f} FPS", flush=True)

        del batch, input_batch, preds, dets_list

    decode_thread.join()

    if pending_tensors:
        dets = np.ascontiguousarray(torch.cat(pending_tensors, dim=0).cpu().numpy(), dtype=np.float32)
    else:
        dets = np.empty((0, 6), dtype=np.float32)

    return {
        "frame_ids": np.asarray(frame_ids, dtype=np.int32),
        "counts": np.asarray(counts, dtype=np.int32),
        "dets": dets,
        "processed_frames": int(process_count),
    }


def detect_segment(segment, runtime, model, batch_size, detection_cache_path: Path, overwrite: bool):
    if detection_cache_path.exists() and not overwrite:
        cached = np.load(detection_cache_path)
        return {
            "frame_ids": cached["frame_ids"],
            "counts": cached["counts"],
            "dets": cached["dets"],
            "processed_frames": int(len(cached["frame_ids"])),
            "cache_hit": True,
        }

    nvc = runtime["nvc"]
    video_path = segment["video_path"]
    total_frames = int(segment["end_frame"]) - int(segment["start_frame"])
    can_seek = "encoded_1_fixed" in video_path

    if total_frames <= 0:
        raise ValueError(f"Invalid frame range for segment: {segment['segment_name']}")

    decoder = nvc.SimpleDecoder(
        enc_file_path=video_path,
        gpu_id=rph.GPU_ID,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.RGBP,
    )
    current_frame, ready = rph.move_decoder_to_frame(
        decoder,
        0,
        int(segment["start_frame"]),
        can_seek,
    )
    if not ready or current_frame != int(segment["start_frame"]):
        raise RuntimeError(f"Failed to reach segment start frame for {segment['segment_name']}")

    result = _decode_segment(decoder, runtime, model, total_frames, batch_size)
    del decoder

    if result["processed_frames"] != total_frames:
        raise RuntimeError(
            f"Processed frame count mismatch for {segment['segment_name']}: "
            f"expected {total_frames}, got {result['processed_frames']}"
        )

    detection_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        detection_cache_path,
        frame_ids=result["frame_ids"],
        counts=result["counts"],
        dets=result["dets"],
    )
    result["cache_hit"] = False
    return result


def run_tracking_for_backend(backend, segment, detection_result, csv_path: Path, overwrite: bool):
    if csv_path.exists() and not overwrite:
        return "skipped_existing"

    os.environ["TRACKER_BACKEND"] = backend
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    track_segment(
        [(
            detection_result["frame_ids"].tolist(),
            detection_result["dets"],
            detection_result["counts"].tolist(),
        )],
        str(csv_path),
    )
    return "written"


def run_count_for_backend(backend_layout, count_processes):
    cmd = [
        sys.executable,
        str(CV_PART_DIR / "VechilCountCPU.py"),
        "--csv-root",
        str(backend_layout["tracking_root"]),
        "--count-root",
        str(backend_layout["count_root"]),
        "--manifest-path",
        str(backend_layout["manifest_path"]),
        "--processes",
        str(count_processes),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def write_run_summary(output_root: Path, segments, backends, layout, detection_meta):
    summary_path = output_root / "README.md"
    lines = [
        "# CV A/B Test",
        "",
        f"- Run UTC time: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}",
        f"- Backends: {', '.join(backends)}",
        "",
        "## Segments",
        "",
    ]

    for segment in segments:
        lines.append(
            f"- `{segment['segment_name']}` | video `{segment['video_name']}` | "
            f"{segment['start_time']} -> {segment['end_time']} | "
            f"{segment['duration_sec']} s"
        )
        if segment["note"]:
            lines.append(f"  note: {segment['note']}")

    lines.extend([
        "",
        "## Output Layout",
        "",
        f"- detection caches: `{layout['detections']}`",
        f"- run metadata: `{layout['meta']}`",
    ])
    for backend in backends:
        backend_layout = layout["backends"][backend]
        lines.append(f"- {backend} tracking: `{backend_layout['tracking_root']}`")
        lines.append(f"- {backend} count: `{backend_layout['count_root']}`")
        lines.append(f"- {backend} manifest: `{backend_layout['manifest_path']}`")

    lines.extend([
        "",
        "## Detection Cache Stats",
        "",
    ])
    for item in detection_meta:
        lines.append(
            f"- `{item['segment_name']}`: processed {item['processed_frames']} frames, "
            f"{item['num_boxes']} boxes, cache_hit={item['cache_hit']}"
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    segments_json_path = Path(args.segments_json).expanduser().resolve()
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else default_output_root()
    )

    rph.BATCH_SIZE = int(args.batch_size)
    segments = load_segments(segments_json_path)
    layout = ensure_output_layout(output_root, args.backends)
    manifest_rows = build_manifest_rows(segments)

    (layout["meta"] / "segments.json").write_text(
        json.dumps(segments, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for backend in args.backends:
        write_manifest(layout["backends"][backend]["manifest_path"], manifest_rows)

    print(f"A/B output root: {output_root}")
    print(f"Segments json  : {segments_json_path}")
    print(f"Backends       : {', '.join(args.backends)}")
    for segment in segments:
        print(
            f"  - {segment['segment_name']} | {segment['video_path']} | "
            f"{segment['start_time']} -> {segment['end_time']}"
        )

    runtime = rph.load_runtime_dependencies()
    model = rph.load_model(runtime, args.model_path)
    print(f"Detection model loaded: {args.model_path}")

    detection_results = {}
    detection_meta = []
    for segment in segments:
        cache_path = layout["detections"] / segment["video_name"] / f"{segment['segment_name']}.npz"
        print(f"\n[Detect] {segment['segment_name']}")
        result = detect_segment(segment, runtime, model, args.batch_size, cache_path, args.overwrite)
        detection_results[segment["segment_name"]] = result
        detection_meta.append({
            "segment_name": segment["segment_name"],
            "processed_frames": int(result["processed_frames"]),
            "num_boxes": int(result["dets"].shape[0]),
            "cache_hit": bool(result["cache_hit"]),
            "cache_path": str(cache_path),
        })
        print(
            f"  detection ready | frames={result['processed_frames']} | "
            f"boxes={result['dets'].shape[0]} | cache_hit={result['cache_hit']}"
        )

    for backend in args.backends:
        backend_layout = layout["backends"][backend]
        print(f"\n=== Tracking backend: {backend} ===")
        for segment in segments:
            csv_path = backend_layout["tracking_root"] / segment["video_name"] / f"{segment['segment_name']}.csv"
            print(f"[Track:{backend}] {segment['segment_name']}")
            status = run_tracking_for_backend(
                backend,
                segment,
                detection_results[segment["segment_name"]],
                csv_path,
                args.overwrite,
            )
            print(f"  tracking status: {status} -> {csv_path}")

        print(f"[Count:{backend}] start")
        run_count_for_backend(backend_layout, args.count_processes)
        print(f"[Count:{backend}] done")

    (layout["meta"] / "detection_meta.json").write_text(
        json.dumps(detection_meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_run_summary(output_root, segments, args.backends, layout, detection_meta)
    print(f"\nA/B test completed. Summary: {output_root / 'README.md'}")


if __name__ == "__main__":
    main()
