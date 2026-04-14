"""Centralized path management for the CV pipeline.

The canonical runtime/artifact root remains the legacy-named
`ProjectTextDocument` directory inside `CV_part`, but all scripts should reach
it through this module instead of hard-coded strings. Static assets such as
`bach2.pt`, `time_limit.json`, and `a1_copy_2_copy.json` stay in `CV_part/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import os


_LEGACY_ARTIFACT_DIRNAME = "ProjectTextDocument"
_DEFAULT_VIDEO_DIR_CANDIDATES = (
    "/root/autodl-tmp/port_bridge",
    "/data/encoded_1_fixed",
    "/data/encoded_2",
    "E:/encoded_1_fixed",
    "E:/encoded_2",
)
_CRAWLER_DIRNAME = "crawler"


def _resolve_path(raw_value: str | None, default: Path, base_dir: Path) -> Path:
    if not raw_value:
        return default

    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_path_list(raw_value: str | None) -> tuple[Path, ...]:
    if not raw_value:
        return ()

    items = [item.strip() for item in raw_value.split(os.pathsep) if item.strip()]
    return tuple(Path(item).expanduser() for item in items)


def _existing_video_dirs() -> tuple[Path, ...]:
    paths = tuple(Path(item).expanduser() for item in _DEFAULT_VIDEO_DIR_CANDIDATES)
    return tuple(path for path in paths if path.is_dir())


def _locate_cv_part_dir(file_path: str | Path) -> Path:
    current = Path(file_path).expanduser().resolve()
    if current.is_file():
        current = current.parent

    for candidate in (current, *current.parents):
        if candidate.name == "CV_part":
            return candidate

    raise ValueError(f"Cannot locate CV_part root from path: {file_path}")


@dataclass(frozen=True)
class DualGPURunPaths:
    run_dir: Path
    log_dir: Path
    list_dir: Path
    gpu0_list: Path
    gpu1_list: Path
    gpu0_log: Path
    gpu1_log: Path
    gpu0_manifest: Path
    gpu1_manifest: Path

    @classmethod
    def from_tracking_root(cls, tracking_root: str | Path, run_tag: str) -> "DualGPURunPaths":
        tracking_root = Path(tracking_root).expanduser().resolve()
        run_dir = tracking_root / "runs" / run_tag
        log_dir = run_dir / "logs"
        list_dir = run_dir / "lists"
        return cls(
            run_dir=run_dir,
            log_dir=log_dir,
            list_dir=list_dir,
            gpu0_list=list_dir / "gpu0_videos.txt",
            gpu1_list=list_dir / "gpu1_videos.txt",
            gpu0_log=log_dir / "gpu0.log",
            gpu1_log=log_dir / "gpu1.log",
            gpu0_manifest=run_dir / "segment_manifest_gpu0.csv",
            gpu1_manifest=run_dir / "segment_manifest_gpu1.csv",
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "run_dir": str(self.run_dir),
            "log_dir": str(self.log_dir),
            "list_dir": str(self.list_dir),
            "gpu0_list": str(self.gpu0_list),
            "gpu1_list": str(self.gpu1_list),
            "gpu0_log": str(self.gpu0_log),
            "gpu1_log": str(self.gpu1_log),
            "gpu0_manifest": str(self.gpu0_manifest),
            "gpu1_manifest": str(self.gpu1_manifest),
        }


@dataclass(frozen=True)
class CVPaths:
    repo_root: Path
    cv_part_dir: Path
    artifacts_dir: Path
    crawler_dir: Path
    crawler_main_path: Path
    crawler_log_config_path: Path
    crawler_camera_csv_path: Path
    crawler_workspace_dir: Path
    crawler_videos_dir: Path
    crawler_temp_dir: Path
    crawler_log_path: Path
    model_stem: str
    model_pt_path: Path
    time_limit_json_path: Path
    tracking_root: Path
    count_root: Path
    gate_line_json_path: Path
    source_json_path: Path
    temp_video_dir: Path
    default_video_dirs: tuple[Path, ...]

    @classmethod
    def from_cv_part_dir(cls, cv_part_dir: str | Path) -> "CVPaths":
        cv_part_dir = Path(cv_part_dir).expanduser().resolve()
        repo_root = cv_part_dir.parent
        model_stem = os.getenv("CV_MODEL_STEM", "bach2").strip() or "bach2"

        artifacts_dir = _resolve_path(
            os.getenv("CV_ARTIFACTS_DIR"),
            cv_part_dir / _LEGACY_ARTIFACT_DIRNAME,
            cv_part_dir,
        )
        default_video_dirs = _resolve_path_list(os.getenv("CV_VIDEO_DIRS")) or _existing_video_dirs()

        return cls(
            repo_root=repo_root,
            cv_part_dir=cv_part_dir,
            artifacts_dir=artifacts_dir,
            crawler_dir=cv_part_dir / _CRAWLER_DIRNAME,
            crawler_main_path=cv_part_dir / _CRAWLER_DIRNAME / "main.py",
            crawler_log_config_path=cv_part_dir / _CRAWLER_DIRNAME / "log_config.py",
            crawler_camera_csv_path=_resolve_path(
                os.getenv("CV_CRAWLER_CAMERA_CSV"),
                cv_part_dir / _CRAWLER_DIRNAME / "camera_location.csv",
                cv_part_dir,
            ),
            crawler_workspace_dir=_resolve_path(
                os.getenv("CV_CRAWLER_WORKSPACE_DIR"),
                artifacts_dir / _CRAWLER_DIRNAME,
                cv_part_dir,
            ),
            crawler_videos_dir=_resolve_path(
                os.getenv("CV_CRAWLER_VIDEOS_DIR"),
                artifacts_dir / _CRAWLER_DIRNAME / "videos",
                cv_part_dir,
            ),
            crawler_temp_dir=_resolve_path(
                os.getenv("CV_CRAWLER_TEMP_DIR"),
                artifacts_dir / _CRAWLER_DIRNAME / "ts_temp_storage",
                cv_part_dir,
            ),
            crawler_log_path=_resolve_path(
                os.getenv("CV_CRAWLER_LOG_PATH"),
                artifacts_dir / _CRAWLER_DIRNAME / "video_log.log",
                cv_part_dir,
            ),
            model_stem=model_stem,
            model_pt_path=_resolve_path(
                os.getenv("CV_MODEL_PT_PATH"),
                cv_part_dir / f"{model_stem}.pt",
                cv_part_dir,
            ),
            time_limit_json_path=_resolve_path(
                os.getenv("CV_TIME_LIMIT_JSON"),
                cv_part_dir / "time_limit.json",
                cv_part_dir,
            ),
            tracking_root=_resolve_path(
                os.getenv("CV_TRACKING_ROOT"),
                artifacts_dir / "DetactionTrackingCsv",
                cv_part_dir,
            ),
            count_root=_resolve_path(
                os.getenv("CV_COUNT_ROOT"),
                artifacts_dir / "GateCountCsv",
                cv_part_dir,
            ),
            gate_line_json_path=_resolve_path(
                os.getenv("CV_GATE_LINE_JSON"),
                artifacts_dir / "GateLineJson.json",
                cv_part_dir,
            ),
            source_json_path=_resolve_path(
                os.getenv("CV_SOURCE_JSON"),
                cv_part_dir / "a1_copy_2_copy.json",
                cv_part_dir,
            ),
            temp_video_dir=_resolve_path(
                os.getenv("CV_TEMP_VIDEO_DIR"),
                artifacts_dir / "temp_videos",
                cv_part_dir,
            ),
            default_video_dirs=default_video_dirs,
        )

    @classmethod
    def from_file(cls, file_path: str | Path) -> "CVPaths":
        return cls.from_cv_part_dir(_locate_cv_part_dir(file_path))

    @classmethod
    def current(cls) -> "CVPaths":
        return cls.from_file(__file__)

    def dual_gpu_run(self, run_tag: str) -> DualGPURunPaths:
        return DualGPURunPaths.from_tracking_root(self.tracking_root, run_tag)

    def as_dict(self) -> dict[str, str]:
        return {
            "repo_root": str(self.repo_root),
            "cv_part_dir": str(self.cv_part_dir),
            "artifacts_dir": str(self.artifacts_dir),
            "crawler_dir": str(self.crawler_dir),
            "crawler_main_path": str(self.crawler_main_path),
            "crawler_log_config_path": str(self.crawler_log_config_path),
            "crawler_camera_csv_path": str(self.crawler_camera_csv_path),
            "crawler_workspace_dir": str(self.crawler_workspace_dir),
            "crawler_videos_dir": str(self.crawler_videos_dir),
            "crawler_temp_dir": str(self.crawler_temp_dir),
            "crawler_log_path": str(self.crawler_log_path),
            "model_stem": self.model_stem,
            "model_pt_path": str(self.model_pt_path),
            "time_limit_json_path": str(self.time_limit_json_path),
            "tracking_root": str(self.tracking_root),
            "count_root": str(self.count_root),
            "gate_line_json_path": str(self.gate_line_json_path),
            "source_json_path": str(self.source_json_path),
            "temp_video_dir": str(self.temp_video_dir),
            "default_video_dirs": os.pathsep.join(str(path) for path in self.default_video_dirs),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Print centralized CV_part paths.")
    parser.add_argument(
        "key",
        nargs="?",
        help="Optional single key to print. Without it, prints all known paths as JSON.",
    )
    parser.add_argument(
        "--run-tag",
        help="Include per-run paths for the provided dual-GPU run tag.",
    )
    args = parser.parse_args()

    paths = CVPaths.current()
    values = paths.as_dict()
    if args.run_tag:
        values.update(paths.dual_gpu_run(args.run_tag).as_dict())

    if args.key:
        try:
            print(values[args.key])
        except KeyError as exc:
            raise SystemExit(f"Unknown key: {exc.args[0]}")
        return

    print(json.dumps(values, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
