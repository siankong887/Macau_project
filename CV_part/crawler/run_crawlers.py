from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


CURRENT_DIR = Path(__file__).resolve().parent
CV_PART_DIR = CURRENT_DIR.parent
REPO_ROOT = CV_PART_DIR.parent

DEFAULT_DURATION = "19h"
MAIN_OUTPUT_DIRNAME = "main_ts"
BORDER_OUTPUT_DIRNAME = "border_cam_images"
BORDER_IMAGES = ("image1", "image5")


def default_output_root() -> Path:
    if REPO_ROOT.parent.name == REPO_ROOT.name:
        return REPO_ROOT.parent / "temp"
    return REPO_ROOT / "temp"


def parse_duration_seconds(raw_value: str) -> int:
    value = str(raw_value).strip().lower()
    if not value:
        raise argparse.ArgumentTypeError("duration cannot be empty")
    if value == "0":
        return 0

    if ":" in value:
        parts = value.split(":")
        if len(parts) not in (2, 3) or not all(part.isdigit() for part in parts):
            raise argparse.ArgumentTypeError("duration must look like MM:SS or HH:MM:SS")
        numbers = [int(part) for part in parts]
        if len(numbers) == 2:
            minutes, seconds = numbers
            return minutes * 60 + seconds
        hours, minutes, seconds = numbers
        return hours * 3600 + minutes * 60 + seconds

    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", value)
    if not match:
        raise argparse.ArgumentTypeError("duration examples: 68400, 19h, 30m, 01:30:00")

    amount = float(match.group(1))
    unit = match.group(2) or "s"
    multiplier = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
    }[unit]
    seconds = int(round(amount * multiplier))
    if seconds < 0:
        raise argparse.ArgumentTypeError("duration cannot be negative")
    return seconds


def make_run_dir(output_root: Path, run_name: str | None) -> Path:
    output_root = output_root.expanduser().resolve()
    if run_name:
        return output_root / run_name

    base_name = datetime.now().strftime("crawler_run_%Y%m%d_%H%M%S")
    run_dir = output_root / base_name
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{base_name}_{suffix}"
        suffix += 1
    return run_dir


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def start_process(name: str, command: list[str], env: dict[str, str] | None = None) -> subprocess.Popen:
    print(f"\n[{name}] {format_command(command)}", flush=True)
    kwargs = {
        "cwd": str(CURRENT_DIR),
        "env": env,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)


def terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + 10
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.2)

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def wait_for_processes(processes: list[tuple[str, subprocess.Popen]]) -> int:
    return_codes: dict[str, int] = {}
    running = list(processes)
    try:
        while running:
            for name, process in list(running):
                return_code = process.poll()
                if return_code is None:
                    continue

                print(f"\n[{name}] exited with code {return_code}", flush=True)
                return_codes[name] = return_code
                running.remove((name, process))
                if return_code != 0:
                    for _, other in running:
                        terminate_process_tree(other)
                    return return_code
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping crawler processes...", flush=True)
        for _, process in running:
            terminate_process_tree(process)
        return 130

    return max(return_codes.values(), default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start both Macau crawler scripts in one command.")
    parser.add_argument(
        "--duration",
        type=parse_duration_seconds,
        default=parse_duration_seconds(DEFAULT_DURATION),
        help="Run duration. Examples: 68400, 19h, 30m, 01:30:00. Use 0 for endless. Default: 19h.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help=f"Parent output directory. Default: {default_output_root()}",
    )
    parser.add_argument(
        "--run-name",
        help="Optional run folder name under --output-root. Defaults to crawler_run_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print directories and commands without starting the crawlers.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    run_dir = make_run_dir(args.output_root, args.run_name)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Run directory already exists and is not empty: {run_dir}")

    main_output_dir = run_dir / MAIN_OUTPUT_DIRNAME
    border_output_dir = run_dir / BORDER_OUTPUT_DIRNAME
    main_temp_dir = run_dir / "_main_ts_temp"

    main_command = [
        sys.executable,
        str(CURRENT_DIR / "main.py"),
        "--duration",
        str(args.duration),
        "--per-video-dirs",
    ]
    border_command = [
        sys.executable,
        str(CURRENT_DIR / "scrape_border_cam.py"),
        "--images",
        *BORDER_IMAGES,
        "--output",
        str(border_output_dir),
        "--duration",
        str(args.duration),
        "--cache-buster",
        "--skip-duplicates",
        "--duplicate-warn-rounds",
        "20",
    ]

    main_env = os.environ.copy()
    main_env.update(
        {
            "CV_CRAWLER_CAMERA_CSV": str(CURRENT_DIR / "camera_location.csv"),
            "CV_CRAWLER_WORKSPACE_DIR": str(main_output_dir),
            "CV_CRAWLER_VIDEOS_DIR": str(main_output_dir),
            "CV_CRAWLER_TEMP_DIR": str(main_temp_dir),
            "CV_CRAWLER_LOG_PATH": str(main_output_dir / "video_log.log"),
        }
    )

    print(f"Run directory: {run_dir}", flush=True)
    print(f"Duration: {args.duration}s", flush=True)
    print(f"Traffic TS output: {main_output_dir}", flush=True)
    print(f"Border image output: {border_output_dir}", flush=True)
    print(f"Border images: {' '.join(BORDER_IMAGES)}", flush=True)

    if args.dry_run:
        print(f"\n[main_ts] {format_command(main_command)}", flush=True)
        print(f"[border_cam_images] {format_command(border_command)}", flush=True)
        return 0

    main_output_dir.mkdir(parents=True, exist_ok=True)
    border_output_dir.mkdir(parents=True, exist_ok=True)

    processes = [
        ("main_ts", start_process("main_ts", main_command, main_env)),
        ("border_cam_images", start_process("border_cam_images", border_command)),
    ]
    return wait_for_processes(processes)


if __name__ == "__main__":
    raise SystemExit(main())
