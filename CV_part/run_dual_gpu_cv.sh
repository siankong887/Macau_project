#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PATHS_PY="${SCRIPT_DIR}/cv_paths.py"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage:
  bash CV_part/run_dual_gpu_cv.sh /path/to/videos_all.txt

Environment variables you can override:
  BATCH_SIZE=1024
  TRACK_WORKERS=20
  TRACKER_BACKEND=bytetrack|oasort
  TIME_LIMIT_JSON=/abs/path/to/time_limit.json
  MODEL_PATH=/abs/path/to/bach2.pt
  TRACKING_ROOT=/abs/path/to/DetactionTrackingCsv
  RUN_TAG=run_001
  EXTRA_ARGS="--plan-only"

The input list file should contain one video path per line.
Blank lines and lines starting with # are ignored.
EOF
  exit 1
fi

VIDEO_LIST="$1"
if [[ ! -f "${VIDEO_LIST}" ]]; then
  echo "Video list not found: ${VIDEO_LIST}" >&2
  exit 1
fi

BATCH_SIZE="${BATCH_SIZE:-1024}"
TRACK_WORKERS="${TRACK_WORKERS:-20}"
TRACKER_BACKEND="${TRACKER_BACKEND:-bytetrack}"
DEFAULT_TIME_LIMIT_JSON="$(${PYTHON_BIN} "${PATHS_PY}" time_limit_json_path)"
DEFAULT_MODEL_PATH="$(${PYTHON_BIN} "${PATHS_PY}" model_pt_path)"
DEFAULT_TRACKING_ROOT="$(${PYTHON_BIN} "${PATHS_PY}" tracking_root)"
TIME_LIMIT_JSON="${TIME_LIMIT_JSON:-${DEFAULT_TIME_LIMIT_JSON}}"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
TRACKING_ROOT="${TRACKING_ROOT:-${DEFAULT_TRACKING_ROOT}}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

WORK_DIR="${TRACKING_ROOT}/runs/${RUN_TAG}"
LOG_DIR="${WORK_DIR}/logs"
LIST_DIR="${WORK_DIR}/lists"
mkdir -p "${LOG_DIR}" "${LIST_DIR}" "${TRACKING_ROOT}"

GPU0_LIST="${LIST_DIR}/gpu0_videos.txt"
GPU1_LIST="${LIST_DIR}/gpu1_videos.txt"
GPU0_LOG="${LOG_DIR}/gpu0.log"
GPU1_LOG="${LOG_DIR}/gpu1.log"
GPU0_MANIFEST="${WORK_DIR}/segment_manifest_gpu0.csv"
GPU1_MANIFEST="${WORK_DIR}/segment_manifest_gpu1.csv"

: > "${GPU0_LIST}"
: > "${GPU1_LIST}"

awk '
  BEGIN { idx = 0 }
  /^[[:space:]]*#/ { next }
  /^[[:space:]]*$/ { next }
  {
    if (idx % 2 == 0) {
      print $0 >> gpu0
    } else {
      print $0 >> gpu1
    }
    idx++
  }
' gpu0="${GPU0_LIST}" gpu1="${GPU1_LIST}" "${VIDEO_LIST}"

GPU0_COUNT=$(grep -cve '^[[:space:]]*$' "${GPU0_LIST}" 2>/dev/null || true)
GPU1_COUNT=$(grep -cve '^[[:space:]]*$' "${GPU1_LIST}" 2>/dev/null || true)

echo "Run tag        : ${RUN_TAG}"
echo "Repo root      : ${REPO_ROOT}"
echo "Tracking root  : ${TRACKING_ROOT}"
echo "Model path     : ${MODEL_PATH}"
echo "Time limit json: ${TIME_LIMIT_JSON}"
echo "BATCH_SIZE     : ${BATCH_SIZE}"
echo "TRACK_WORKERS  : ${TRACK_WORKERS}"
echo "TRACKER_BACKEND: ${TRACKER_BACKEND}"
echo "GPU0 videos    : ${GPU0_COUNT}"
echo "GPU1 videos    : ${GPU1_COUNT}"
echo "GPU0 log       : ${GPU0_LOG}"
echo "GPU1 log       : ${GPU1_LOG}"

if [[ -s "${GPU0_LIST}" ]]; then
  echo "Starting GPU0 worker..."
  (
    export CUDA_VISIBLE_DEVICES=0
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export BATCH_SIZE
    export TRACK_WORKERS
    export TRACKER_BACKEND
    cd "${REPO_ROOT}"
    set -x
    ${PYTHON_BIN} "${SCRIPT_DIR}/run_peak_hours.py" \
      --video-list "${GPU0_LIST}" \
      --time-limit-json "${TIME_LIMIT_JSON}" \
      --model-path "${MODEL_PATH}" \
      --tracking-root "${TRACKING_ROOT}" \
      --manifest-path "${GPU0_MANIFEST}" \
      ${EXTRA_ARGS}
  ) >"${GPU0_LOG}" 2>&1 &
  PID0=$!
else
  PID0=""
  echo "GPU0 list is empty, skip GPU0 worker."
fi

if [[ -s "${GPU1_LIST}" ]]; then
  echo "Starting GPU1 worker..."
  (
    export CUDA_VISIBLE_DEVICES=1
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export BATCH_SIZE
    export TRACK_WORKERS
    export TRACKER_BACKEND
    cd "${REPO_ROOT}"
    set -x
    ${PYTHON_BIN} "${SCRIPT_DIR}/run_peak_hours.py" \
      --video-list "${GPU1_LIST}" \
      --time-limit-json "${TIME_LIMIT_JSON}" \
      --model-path "${MODEL_PATH}" \
      --tracking-root "${TRACKING_ROOT}" \
      --manifest-path "${GPU1_MANIFEST}" \
      ${EXTRA_ARGS}
  ) >"${GPU1_LOG}" 2>&1 &
  PID1=$!
else
  PID1=""
  echo "GPU1 list is empty, skip GPU1 worker."
fi

EXIT_CODE=0

if [[ -n "${PID0}" ]]; then
  wait "${PID0}" || EXIT_CODE=$?
fi

if [[ -n "${PID1}" ]]; then
  wait "${PID1}" || EXIT_CODE=$?
fi

echo "All GPU workers finished."
echo "GPU0 manifest: ${GPU0_MANIFEST}"
echo "GPU1 manifest: ${GPU1_MANIFEST}"
echo "GPU0 log     : ${GPU0_LOG}"
echo "GPU1 log     : ${GPU1_LOG}"

exit "${EXIT_CODE}"
