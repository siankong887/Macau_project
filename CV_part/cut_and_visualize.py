#!/usr/bin/env python3
"""
Cut video segments by timestamp and draw gate detection lines from
a1_copy_2_copy.json onto the clipped segments.

Camera key is derived from the video filename automatically:
  m_1.mp4 → cam_m1,  r_1.mp4 → cam_r1,  a_2.mp4 → cam_a2

Usage:
    python cut_and_visualize.py \
        --videos /path/to/m_1.mp4,00:39:34,00:54:34 \
                 /path/to/r_1.mp4,00:36:15,00:51:15 \
        --config CV_part/a1_copy_2_copy.json \
        --output CV_part/AB_test \
        [--fps 30]
"""

import argparse
import json
import subprocess
from pathlib import Path

import cv2

# Distinct colours for each gate (BGR)
GATE_COLOURS = [
    (0, 255, 0),     # green
    (0, 165, 255),   # orange
    (255, 0, 0),     # blue
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
]


def video_name_to_cam_key(video_name: str) -> str:
    """Derive camera key from video filename: 'm_1' → 'cam_m1', 'a_2' → 'cam_a2'."""
    return "cam_" + video_name.replace("_", "")


def ts_to_frames(ts: str, fps: float) -> int:
    """Convert HH:MM:SS timestamp to frame number."""
    parts = ts.split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return int((h * 3600 + m * 60 + s) * fps)


def load_gate_lines(config_path: str) -> dict[str, list[dict]]:
    """Return {cam_key: [{gate_id, line: [(x1,y1),(x2,y2)]}]}."""
    with open(config_path) as f:
        data = json.load(f)

    cam_gates: dict[str, list[dict]] = {}
    for cam in data["list"]:
        key = cam["camera"]
        gates = []
        for g in cam.get("gate", []):
            line_raw = g.get("line")
            if not line_raw:
                continue
            seg = line_raw[0]
            gates.append({
                "gate_id": g["gate_id"],
                "line": [(seg[0][0], seg[0][1]), (seg[1][0], seg[1][1])],
            })
        cam_gates[key] = gates
    return cam_gates


def cut_and_overlay(src: str, dst_cut: str, dst_gates: str | None,
                    start_frame: int, end_frame: int, fps: float,
                    gates: list[dict]):
    """
    Read source video from start_frame to end_frame, write two outputs:
      1) dst_cut   — plain cut segment
      2) dst_gates — same segment with gate lines overlaid (if gates provided)

    Both outputs are written via ffmpeg pipe to avoid OpenCV VideoWriter
    timebase issues (these HLS-recorded videos have timebase=90000).
    """
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {src}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = end_frame - start_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    print(f"  seeked to frame {start_frame}, will read {total_frames} frames ({w}x{h} @ {fps}fps)")

    def make_ffmpeg_writer(dst):
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}", "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            dst,
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    proc_cut = make_ffmpeg_writer(dst_cut)
    proc_gates = make_ffmpeg_writer(dst_gates) if dst_gates and gates else None

    frame_idx = 0
    while frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"  [WARN] EOF at frame {start_frame + frame_idx} (expected {end_frame})")
            break

        raw = frame.tobytes()
        proc_cut.stdin.write(raw)

        if proc_gates is not None:
            overlay = frame.copy()
            for i, g in enumerate(gates):
                colour = GATE_COLOURS[i % len(GATE_COLOURS)]
                pt1, pt2 = g["line"]
                cv2.line(overlay, pt1, pt2, colour, 2, cv2.LINE_AA)
                label_pos = (min(pt1[0], pt2[0]), min(pt1[1], pt2[1]) - 5)
                cv2.putText(overlay, g["gate_id"], label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
            proc_gates.stdin.write(overlay.tobytes())

        frame_idx += 1
        if frame_idx % 5000 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  [progress] {frame_idx}/{total_frames} ({pct:.1f}%)")

    proc_cut.stdin.close()
    proc_cut.wait()
    stderr_cut = proc_cut.stderr.read().decode()
    if stderr_cut:
        print(f"  [ffmpeg cut stderr] {stderr_cut[:300]}")

    if proc_gates is not None:
        proc_gates.stdin.close()
        proc_gates.wait()
        stderr_gates = proc_gates.stderr.read().decode()
        if stderr_gates:
            print(f"  [ffmpeg gates stderr] {stderr_gates[:300]}")

    cap.release()
    print(f"  done — {frame_idx} frames written")


def parse_video_arg(arg: str) -> tuple[str, str, str]:
    """Parse 'path,start,end' → (path, start_ts, end_ts)."""
    parts = arg.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Expected 'video_path,HH:MM:SS,HH:MM:SS', got: {arg}")
    return parts[0].strip(), parts[1].strip(), parts[2].strip()


def main():
    parser = argparse.ArgumentParser(
        description="Cut video segments by timestamp and visualize gate lines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python cut_and_visualize.py \\
      --videos /path/to/m_1.mp4,00:39:34,00:54:34 \\
               /path/to/r_1.mp4,00:36:15,00:51:15

  python cut_and_visualize.py \\
      --videos /path/to/a_2.mp4,01:00:00,01:15:00 \\
      --fps 25 --output ./clips""")
    parser.add_argument("--videos", nargs="+", required=True,
                        help="Each entry: video_path,HH:MM:SS,HH:MM:SS  (path,start,end)")
    parser.add_argument("--config", type=str,
                        default="CV_part/a1_copy_2_copy.json",
                        help="Camera gate config JSON (default: CV_part/a1_copy_2_copy.json)")
    parser.add_argument("--output", type=str, default="CV_part/AB_test",
                        help="Output directory (default: CV_part/AB_test)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Actual fps of source videos (default: 30)")
    args = parser.parse_args()

    cam_gates = load_gate_lines(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for video_arg in args.videos:
        video_path, start_ts, end_ts = parse_video_arg(video_arg)

        if not Path(video_path).exists():
            print(f"[SKIP] source not found: {video_path}")
            continue

        video_name = Path(video_path).stem  # e.g. "m_1"
        start_frame = ts_to_frames(start_ts, args.fps)
        end_frame = ts_to_frames(end_ts, args.fps)
        segment_name = f"{video_name}_{start_ts.replace(':','_')}__{end_ts.replace(':','_')}"

        print(f"\n=== {segment_name} ===")
        print(f"  source: {video_path}")
        print(f"  time:   {start_ts} → {end_ts}  (frame {start_frame} → {end_frame})")

        cut_path = str(out_dir / f"{segment_name}_cut.mp4")

        cam_key = video_name_to_cam_key(video_name)
        gates = cam_gates.get(cam_key, [])
        gates_path = str(out_dir / f"{segment_name}_gates.mp4") if gates else None

        if gates:
            print(f"  will overlay {len(gates)} gate lines (from {cam_key})")
        else:
            print(f"  [WARN] no gates for cam_key={cam_key} in config, cut only")

        cut_and_overlay(video_path, cut_path, gates_path,
                        start_frame, end_frame, args.fps, gates)

    print("\nAll done.")


if __name__ == "__main__":
    main()
