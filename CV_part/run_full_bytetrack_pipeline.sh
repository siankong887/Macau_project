#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PATHS_PY="${SCRIPT_DIR}/cv_paths.py"

if [[ "${OS:-}" == "Windows_NT" && "${ALLOW_WINDOWS_SH:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
Windows detected. This .sh entrypoint keeps Linux/server-oriented defaults.
Use the Windows launcher instead:
  powershell -ExecutionPolicy Bypass -File CV_part/run_full_bytetrack_pipeline.ps1

Set ALLOW_WINDOWS_SH=1 only if you intentionally want to run this Bash script under Git Bash.
EOF
  exit 1
fi

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
FULL_ROOT="${FULL_ROOT:-${SCRIPT_DIR}/full_runs/${RUN_TAG}}"
TRACKING_ROOT="${TRACKING_ROOT:-${FULL_ROOT}/tracking}"
COUNT_ROOT="${COUNT_ROOT:-${FULL_ROOT}/count}"
IMAGE_COUNT_ROOT="${IMAGE_COUNT_ROOT:-${FULL_ROOT}/image_count}"
PIPELINE_LOG_DIR="${FULL_ROOT}/logs"

MODEL_PATH="${MODEL_PATH:-$(${PYTHON_BIN} "${PATHS_PY}" model_pt_path)}"
TIME_LIMIT_JSON="${TIME_LIMIT_JSON:-$(${PYTHON_BIN} "${PATHS_PY}" time_limit_json_path)}"
SOURCE_JSON="${SOURCE_JSON:-$(${PYTHON_BIN} "${PATHS_PY}" source_json_path)}"
GATE_LINE_JSON="${GATE_LINE_JSON:-$(${PYTHON_BIN} "${PATHS_PY}" gate_line_json_path)}"
REID_MODEL_PATH="${REID_MODEL_PATH:-${SCRIPT_DIR}/models/openvino/public/vehicle-reid-0001/FP32/vehicle-reid-0001.xml}"

# Linux defaults are single-GPU. Override GPU_IDS=0,1 only on a multi-GPU server.
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-512}"
TRACK_WORKERS="${TRACK_WORKERS:-14}"
COUNT_PROCESSES="${COUNT_PROCESSES:-18}"
TRACKER_BACKEND="${TRACKER_BACKEND:-bytetrack}"
NVDEC_DECODER_MODE="${NVDEC_DECODER_MODE:-demux}"
SEGMENT_MODE="${SEGMENT_MODE:-hourly-from-start}"
SEGMENT_SECONDS="${SEGMENT_SECONDS:-3600}"
EXPECTED_VIDEO_SECONDS="${EXPECTED_VIDEO_SECONDS:-68100}"

# Future crawler output is TS by default. VIDEO_LIST wins; otherwise VIDEO_DIRS is scanned recursively.
VIDEO_LIST="${VIDEO_LIST:-}"
VIDEO_DIRS="${VIDEO_DIRS:-$(${PYTHON_BIN} "${PATHS_PY}" crawler_videos_dir)}"
VIDEO_EXTENSIONS="${VIDEO_EXTENSIONS:-ts}"
VIDEO_LIST_PATH=""

RUN_DETECTION="${RUN_DETECTION:-1}"
RUN_GATE_COUNT="${RUN_GATE_COUNT:-1}"
RUN_IMAGE_COUNT="${RUN_IMAGE_COUNT:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
COUNT_EXTRA_ARGS="${COUNT_EXTRA_ARGS:-}"

# Image counting. Use IMAGE_DIRS=/path/image1:/path/image5 or BORDER_IMAGE_ROOT=/path/border_cam_images.
IMAGE_DIRS="${IMAGE_DIRS:-}"
IMAGE_DIR_1="${IMAGE_DIR_1:-}"
IMAGE_DIR_2="${IMAGE_DIR_2:-}"
BORDER_IMAGE_ROOT="${BORDER_IMAGE_ROOT:-}"
IMAGE_CONF="${IMAGE_CONF:-0.4}"
IMAGE_OPENVINO_DEVICE="${IMAGE_OPENVINO_DEVICE:-AUTO}"
IMAGE_REID_THRESH="${IMAGE_REID_THRESH:-0.50}"
IMAGE_MAX_DIST="${IMAGE_MAX_DIST:-300}"
IMAGE_LOOKBACK="${IMAGE_LOOKBACK:-3}"
IMAGE_ZONE_Y_MIN="${IMAGE_ZONE_Y_MIN:-}"
IMAGE_ZONE_Y_MAX="${IMAGE_ZONE_Y_MAX:-}"
IMAGE_DEFAULT_ZONE_Y_MIN="${IMAGE_DEFAULT_ZONE_Y_MIN:-0.10}"
IMAGE_DEFAULT_ZONE_Y_MAX="${IMAGE_DEFAULT_ZONE_Y_MAX:-0.80}"
IMAGE1_ZONE_Y_MIN="${IMAGE1_ZONE_Y_MIN:-0.25}"
IMAGE1_ZONE_Y_MAX="${IMAGE1_ZONE_Y_MAX:-0.99}"
IMAGE5_ZONE_Y_MIN="${IMAGE5_ZONE_Y_MIN:-0.01}"
IMAGE5_ZONE_Y_MAX="${IMAGE5_ZONE_Y_MAX:-0.45}"
IMAGE_EXTRA_ARGS="${IMAGE_EXTRA_ARGS:-}"

RUN_ROOT="${TRACKING_ROOT}/runs/${RUN_TAG}"
LIST_DIR="${RUN_ROOT}/lists"
WORKER_LOG_DIR="${RUN_ROOT}/logs"
MERGED_MANIFEST="${TRACKING_ROOT}/segment_manifest.csv"

is_enabled() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

split_list() {
  local raw="$1"
  tr ',;' '::' <<<"${raw}" | tr ':' '\n' | sed '/^[[:space:]]*$/d'
}

print_config() {
  cat <<EOF
Run tag             : ${RUN_TAG}
Repo root           : ${REPO_ROOT}
Full root           : ${FULL_ROOT}
Tracking root       : ${TRACKING_ROOT}
Count root          : ${COUNT_ROOT}
Image count root    : ${IMAGE_COUNT_ROOT}
Model path          : ${MODEL_PATH}
Time limit json     : ${TIME_LIMIT_JSON}
Source json         : ${SOURCE_JSON}
Gate line json      : ${GATE_LINE_JSON}
GPU ids             : ${GPU_IDS}
Video extensions    : ${VIDEO_EXTENSIONS}
Tracker backend     : ${TRACKER_BACKEND}
NVDEC decoder mode  : ${NVDEC_DECODER_MODE}
Segment mode        : ${SEGMENT_MODE}
Segment seconds     : ${SEGMENT_SECONDS}
Expected video sec  : ${EXPECTED_VIDEO_SECONDS}
BATCH_SIZE          : ${BATCH_SIZE}
TRACK_WORKERS       : ${TRACK_WORKERS}
COUNT_PROCESSES     : ${COUNT_PROCESSES}
RUN_DETECTION       : ${RUN_DETECTION}
RUN_GATE_COUNT      : ${RUN_GATE_COUNT}
RUN_IMAGE_COUNT     : ${RUN_IMAGE_COUNT}
Image zone defaults : image1=${IMAGE1_ZONE_Y_MIN}..${IMAGE1_ZONE_Y_MAX}, image5=${IMAGE5_ZONE_Y_MIN}..${IMAGE5_ZONE_Y_MAX}, other=${IMAGE_DEFAULT_ZONE_Y_MIN}..${IMAGE_DEFAULT_ZONE_Y_MAX}
EOF
}

ensure_required_files() {
  local missing=0
  if is_enabled "${RUN_DETECTION}" || (is_enabled "${RUN_IMAGE_COUNT}" && [[ -n "${IMAGE_DIRS}${IMAGE_DIR_1}${IMAGE_DIR_2}${BORDER_IMAGE_ROOT}" ]]); then
    if [[ ! -f "${MODEL_PATH}" ]]; then
      echo "Required model file not found: ${MODEL_PATH}" >&2
      missing=1
    fi
  fi
  if is_enabled "${RUN_DETECTION}" && [[ "${SEGMENT_MODE}" == "time-limit" ]] && [[ ! -f "${TIME_LIMIT_JSON}" ]]; then
    echo "Required time limit json not found: ${TIME_LIMIT_JSON}" >&2
    missing=1
  fi
  if is_enabled "${RUN_GATE_COUNT}" && [[ ! -f "${SOURCE_JSON}" ]]; then
    echo "Required source json not found: ${SOURCE_JSON}" >&2
    missing=1
  fi
  if is_enabled "${RUN_IMAGE_COUNT}" && [[ -n "${IMAGE_DIRS}${IMAGE_DIR_1}${IMAGE_DIR_2}${BORDER_IMAGE_ROOT}" ]] && [[ ! -f "${REID_MODEL_PATH}" ]]; then
    echo "OpenVINO ReID model not found: ${REID_MODEL_PATH}" >&2
    missing=1
  fi
  if [[ "${missing}" -ne 0 ]]; then
    exit 1
  fi
}

build_video_list() {
  mkdir -p "${LIST_DIR}"

  if [[ -n "${VIDEO_LIST}" ]]; then
    if [[ ! -f "${VIDEO_LIST}" ]]; then
      echo "Video list not found: ${VIDEO_LIST}" >&2
      exit 1
    fi
    VIDEO_LIST_PATH="${VIDEO_LIST}"
    return
  fi

  local generated="${LIST_DIR}/all_videos.txt"
  : >"${generated}"

  mapfile -t video_dirs < <(split_list "${VIDEO_DIRS}")
  mapfile -t video_exts < <(split_list "${VIDEO_EXTENSIONS}")

  if [[ "${#video_dirs[@]}" -eq 0 ]]; then
    echo "No VIDEO_LIST provided and VIDEO_DIRS is empty." >&2
    exit 1
  fi
  if [[ "${#video_exts[@]}" -eq 0 ]]; then
    echo "VIDEO_EXTENSIONS is empty." >&2
    exit 1
  fi

  local find_args=()
  local first=1
  local ext
  for ext in "${video_exts[@]}"; do
    ext="${ext#.}"
    if [[ "${first}" -eq 0 ]]; then
      find_args+=("-o")
    fi
    find_args+=("-iname" "*.${ext}")
    first=0
  done

  local dir
  for dir in "${video_dirs[@]}"; do
    if [[ ! -d "${dir}" ]]; then
      echo "Video directory not found, skip: ${dir}" >&2
      continue
    fi
    find "${dir}" -type f \( "${find_args[@]}" \) -print
  done | sort >"${generated}"

  if [[ ! -s "${generated}" ]]; then
    echo "No videos found. VIDEO_DIRS=${VIDEO_DIRS}, VIDEO_EXTENSIONS=${VIDEO_EXTENSIONS}" >&2
    exit 1
  fi

  VIDEO_LIST_PATH="${generated}"
}

split_video_list_by_gpu() {
  mapfile -t gpu_ids < <(split_list "${GPU_IDS}")
  if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
    echo "GPU_IDS is empty." >&2
    exit 1
  fi

  GPU_LISTS=()
  GPU_MANIFESTS=()
  GPU_LOGS=()

  local idx
  for idx in "${!gpu_ids[@]}"; do
    GPU_LISTS[idx]="${LIST_DIR}/gpu${idx}_videos.txt"
    GPU_MANIFESTS[idx]="${RUN_ROOT}/segment_manifest_gpu${idx}.csv"
    GPU_LOGS[idx]="${WORKER_LOG_DIR}/gpu${idx}.log"
    : >"${GPU_LISTS[idx]}"
  done

  local line
  local assign_idx=0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    printf '%s\n' "${line}" >>"${GPU_LISTS[$((assign_idx % ${#gpu_ids[@]}))]}"
    assign_idx=$((assign_idx + 1))
  done <"${VIDEO_LIST_PATH}"

  if [[ "${assign_idx}" -eq 0 ]]; then
    echo "Video list has no runnable entries: ${VIDEO_LIST_PATH}" >&2
    exit 1
  fi

  echo "Video list          : ${VIDEO_LIST_PATH}"
  echo "Video count         : ${assign_idx}"
  for idx in "${!gpu_ids[@]}"; do
    local count
    count=$(grep -cve '^[[:space:]]*$' "${GPU_LISTS[idx]}" 2>/dev/null || true)
    echo "GPU ${gpu_ids[idx]} videos      : ${count}"
    echo "GPU ${gpu_ids[idx]} log         : ${GPU_LOGS[idx]}"
  done
}

run_detection_tracking() {
  mkdir -p "${TRACKING_ROOT}" "${COUNT_ROOT}" "${PIPELINE_LOG_DIR}" "${WORKER_LOG_DIR}" "${LIST_DIR}"
  build_video_list
  split_video_list_by_gpu

  local pids=()
  local idx
  for idx in "${!gpu_ids[@]}"; do
    if [[ ! -s "${GPU_LISTS[idx]}" ]]; then
      echo "GPU ${gpu_ids[idx]} list is empty, skip."
      continue
    fi

    echo "Starting worker ${idx} on CUDA_VISIBLE_DEVICES=${gpu_ids[idx]}..."
    (
      export CUDA_VISIBLE_DEVICES="${gpu_ids[idx]}"
      export OMP_NUM_THREADS=1
      export MKL_NUM_THREADS=1
      export OPENBLAS_NUM_THREADS=1
      export NUMEXPR_NUM_THREADS=1
      export BATCH_SIZE
      export TRACK_WORKERS
      export TRACKER_BACKEND
      export NVDEC_DECODER_MODE
      cd "${REPO_ROOT}"
      set -x
      "${PYTHON_BIN}" "${SCRIPT_DIR}/run_peak_hours.py" \
        --video-list "${GPU_LISTS[idx]}" \
        --time-limit-json "${TIME_LIMIT_JSON}" \
        --segment-mode "${SEGMENT_MODE}" \
        --segment-seconds "${SEGMENT_SECONDS}" \
        --expected-video-seconds "${EXPECTED_VIDEO_SECONDS}" \
        --model-path "${MODEL_PATH}" \
        --tracking-root "${TRACKING_ROOT}" \
        --manifest-path "${GPU_MANIFESTS[idx]}" \
        ${EXTRA_ARGS}
    ) >"${GPU_LOGS[idx]}" 2>&1 &
    pids+=("$!")
  done

  if [[ "${#pids[@]}" -eq 0 ]]; then
    echo "No GPU workers were started." >&2
    exit 1
  fi

  local exit_code=0
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}" || exit_code=$?
  done

  if [[ "${exit_code}" -ne 0 ]]; then
    echo "Detection/tracking failed with exit code ${exit_code}." >&2
    echo "Check worker logs under: ${WORKER_LOG_DIR}" >&2
    exit "${exit_code}"
  fi

  echo "All detection/tracking workers finished."
}

merge_manifests() {
  mkdir -p "${TRACKING_ROOT}"
  "${PYTHON_BIN}" - "${MERGED_MANIFEST}" "${GPU_MANIFESTS[@]}" <<'PY'
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
        print(f"Manifest source missing, skip: {src}")
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
            rows.append({name: row.get(name, "") for name in header})

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
}

run_gate_count() {
  mkdir -p "${COUNT_ROOT}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/VechilCountCPU.py" \
    --csv-root "${TRACKING_ROOT}" \
    --count-root "${COUNT_ROOT}" \
    --manifest-path "${MERGED_MANIFEST}" \
    --source-json "${SOURCE_JSON}" \
    --gate-line-json "${GATE_LINE_JSON}" \
    --processes "${COUNT_PROCESSES}" \
    ${COUNT_EXTRA_ARGS}
}

resolve_image_dirs() {
  if [[ -z "${IMAGE_DIRS}" && ( -n "${IMAGE_DIR_1}" || -n "${IMAGE_DIR_2}" ) ]]; then
    IMAGE_DIRS="${IMAGE_DIR_1}:${IMAGE_DIR_2}"
  fi
  if [[ -z "${IMAGE_DIRS}" && -n "${BORDER_IMAGE_ROOT}" ]]; then
    IMAGE_DIRS="${BORDER_IMAGE_ROOT}/image1:${BORDER_IMAGE_ROOT}/image5"
  fi
}

safe_name_for_path() {
  local path="$1"
  local base
  base="$(basename "${path}")"
  if [[ -z "${base}" || "${base}" == "." || "${base}" == "/" ]]; then
    base="images"
  fi
  printf '%s' "${base}" | tr -c 'A-Za-z0-9._-' '_'
}

image_zone_for_dir() {
  local image_dir="$1"
  local name
  local zone_min="${IMAGE_DEFAULT_ZONE_Y_MIN}"
  local zone_max="${IMAGE_DEFAULT_ZONE_Y_MAX}"
  name="$(basename "${image_dir}")"

  case "${name,,}" in
    image1)
      zone_min="${IMAGE1_ZONE_Y_MIN}"
      zone_max="${IMAGE1_ZONE_Y_MAX}"
      ;;
    image5)
      zone_min="${IMAGE5_ZONE_Y_MIN}"
      zone_max="${IMAGE5_ZONE_Y_MAX}"
      ;;
  esac

  if [[ -n "${IMAGE_ZONE_Y_MIN}" ]]; then zone_min="${IMAGE_ZONE_Y_MIN}"; fi
  if [[ -n "${IMAGE_ZONE_Y_MAX}" ]]; then zone_max="${IMAGE_ZONE_Y_MAX}"; fi

  printf '%s %s\n' "${zone_min}" "${zone_max}"
}

run_image_count() {
  resolve_image_dirs
  if [[ -z "${IMAGE_DIRS}" ]]; then
    echo "RUN_IMAGE_COUNT is enabled, but IMAGE_DIRS/BORDER_IMAGE_ROOT is not set. Skip image counting."
    return
  fi

  mkdir -p "${IMAGE_COUNT_ROOT}"
  mapfile -t image_dirs < <(split_list "${IMAGE_DIRS}")
  if [[ "${#image_dirs[@]}" -eq 0 ]]; then
    echo "No image directories configured. Skip image counting."
    return
  fi

  local idx=0
  local image_dir
  for image_dir in "${image_dirs[@]}"; do
    if [[ ! -d "${image_dir}" ]]; then
      echo "Image directory not found: ${image_dir}" >&2
      exit 1
    fi

    idx=$((idx + 1))
    local label
    local output_csv
    local output_log
    local zone_y_min
    local zone_y_max
    label="$(safe_name_for_path "${image_dir}")"
    read -r zone_y_min zone_y_max < <(image_zone_for_dir "${image_dir}")
    output_csv="${IMAGE_COUNT_ROOT}/${idx}_${label}_vehicle_count_results_openvino.csv"
    output_log="${IMAGE_COUNT_ROOT}/${idx}_${label}.log"

    echo "Running image count ${idx}: ${image_dir}"
    echo "Image count zone   : y=${zone_y_min}..${zone_y_max}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/count_vehicles_in_images_openvino.py" \
      --image-dir "${image_dir}" \
      --model "${MODEL_PATH}" \
      --conf "${IMAGE_CONF}" \
      --output-csv "${output_csv}" \
      --reid-model-path "${REID_MODEL_PATH}" \
      --openvino-device "${IMAGE_OPENVINO_DEVICE}" \
      --zone-y-min "${zone_y_min}" \
      --zone-y-max "${zone_y_max}" \
      --reid-thresh "${IMAGE_REID_THRESH}" \
      --max-dist "${IMAGE_MAX_DIST}" \
      --lookback "${IMAGE_LOOKBACK}" \
      ${IMAGE_EXTRA_ARGS} 2>&1 | tee "${output_log}"
    echo "Image count CSV: ${output_csv}"
    echo "Image count log: ${output_log}"
  done
}

main() {
  mkdir -p "${FULL_ROOT}" "${PIPELINE_LOG_DIR}" "${RUN_ROOT}"
  print_config
  ensure_required_files
  cd "${REPO_ROOT}"

  if is_enabled "${RUN_DETECTION}"; then
    run_detection_tracking
    merge_manifests
  else
    echo "RUN_DETECTION is disabled. Reusing existing tracking root: ${TRACKING_ROOT}"
    if [[ ! -f "${MERGED_MANIFEST}" ]]; then
      echo "Merged manifest not found: ${MERGED_MANIFEST}" >&2
      exit 1
    fi
  fi

  if is_enabled "${RUN_GATE_COUNT}"; then
    run_gate_count
  else
    echo "RUN_GATE_COUNT is disabled."
  fi

  if is_enabled "${RUN_IMAGE_COUNT}"; then
    run_image_count
  else
    echo "RUN_IMAGE_COUNT is disabled."
  fi

  cat <<EOF
Full pipeline finished.
Tracking root    : ${TRACKING_ROOT}
Count root       : ${COUNT_ROOT}
Manifest path    : ${MERGED_MANIFEST}
Image count root : ${IMAGE_COUNT_ROOT}
Worker logs      : ${WORKER_LOG_DIR}
EOF
}

main "$@"
