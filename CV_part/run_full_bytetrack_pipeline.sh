#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

VIDEO_LIST="${VIDEO_LIST:-${SCRIPT_DIR}/videos_all_port_bridge.txt}"
MODEL_PATH="${MODEL_PATH:-$(${PYTHON_BIN} "${SCRIPT_DIR}/cv_paths.py" model_pt_path)}"
TIME_LIMIT_JSON="${TIME_LIMIT_JSON:-$(${PYTHON_BIN} "${SCRIPT_DIR}/cv_paths.py" time_limit_json_path)}"

FULL_ROOT="${FULL_ROOT:-${SCRIPT_DIR}/full_runs/20260415_bytetrack_full_32videos}"
TRACKING_ROOT="${TRACKING_ROOT:-${FULL_ROOT}/tracking}"
COUNT_ROOT="${COUNT_ROOT:-${FULL_ROOT}/count}"
RUN_TAG="${RUN_TAG:-dual_gpu_main}"

BATCH_SIZE="${BATCH_SIZE:-512}"
TRACK_WORKERS="${TRACK_WORKERS:-20}"
COUNT_PROCESSES="${COUNT_PROCESSES:-8}"
TRACKER_BACKEND="${TRACKER_BACKEND:-bytetrack}"

RUN_ROOT="${TRACKING_ROOT}/runs/${RUN_TAG}"
GPU0_MANIFEST="${RUN_ROOT}/segment_manifest_gpu0.csv"
GPU1_MANIFEST="${RUN_ROOT}/segment_manifest_gpu1.csv"
MERGED_MANIFEST="${TRACKING_ROOT}/segment_manifest.csv"
PIPELINE_LOG_DIR="${FULL_ROOT}/logs"
mkdir -p "${TRACKING_ROOT}" "${COUNT_ROOT}" "${PIPELINE_LOG_DIR}"

echo "FULL_ROOT       : ${FULL_ROOT}"
echo "TRACKING_ROOT   : ${TRACKING_ROOT}"
echo "COUNT_ROOT      : ${COUNT_ROOT}"
echo "RUN_TAG         : ${RUN_TAG}"
echo "VIDEO_LIST      : ${VIDEO_LIST}"
echo "TRACKER_BACKEND : ${TRACKER_BACKEND}"
echo "BATCH_SIZE      : ${BATCH_SIZE}"
echo "TRACK_WORKERS   : ${TRACK_WORKERS}"
echo "COUNT_PROCESSES : ${COUNT_PROCESSES}"

cd "${REPO_ROOT}"

TRACKER_BACKEND="${TRACKER_BACKEND}" \
BATCH_SIZE="${BATCH_SIZE}" \
TRACK_WORKERS="${TRACK_WORKERS}" \
TRACKING_ROOT="${TRACKING_ROOT}" \
RUN_TAG="${RUN_TAG}" \
TIME_LIMIT_JSON="${TIME_LIMIT_JSON}" \
MODEL_PATH="${MODEL_PATH}" \
bash "${SCRIPT_DIR}/run_dual_gpu_cv.sh" "${VIDEO_LIST}"

${PYTHON_BIN} - "${MERGED_MANIFEST}" "${GPU0_MANIFEST}" "${GPU1_MANIFEST}" <<'PY'
import csv
import sys
from pathlib import Path

dst = Path(sys.argv[1])
srcs = [Path(item) for item in sys.argv[2:]]
header = [
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

rows = []
seen = set()
for src in srcs:
    if not src.exists():
        continue
    with src.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (
                row.get("segment_name", ""),
                row.get("video_path", ""),
                row.get("start_frame", ""),
                row.get("end_frame", ""),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append({key: row.get(key, "") for key in header})

rows.sort(
    key=lambda row: (
        row.get("video_name", ""),
        int(row.get("start_frame", "0") or 0),
        row.get("segment_name", ""),
    )
)

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)

print(f"Merged manifest written: {dst} ({len(rows)} rows)")
PY

${PYTHON_BIN} "${SCRIPT_DIR}/VechilCountCPU.py" \
  --csv-root "${TRACKING_ROOT}" \
  --count-root "${COUNT_ROOT}" \
  --manifest-path "${MERGED_MANIFEST}" \
  --processes "${COUNT_PROCESSES}"

echo "Full pipeline finished."
echo "Tracking root : ${TRACKING_ROOT}"
echo "Count root    : ${COUNT_ROOT}"
echo "Manifest path : ${MERGED_MANIFEST}"
