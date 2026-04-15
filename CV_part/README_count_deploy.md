# CV Part Full Pipeline Deployment Notes

This document describes how to deploy the full `CV_part` runtime on a new
instance, including the complete chain:

- video decoding
- YOLO detection
- CPU tracking
- gate counting

## 1. Current Baseline Environment

This document was written against the currently running instance:

- OS: Ubuntu 22.04 style userspace (`glibc 2.35`)
- Python: `3.12.3`
- NVIDIA driver: `570.124.04`
- CUDA reported by `nvidia-smi`: `12.8`
- ffmpeg / ffprobe: `4.4.2`
- GPU: RTX 4090 x 2 on the current machine

The matching Python package set is recorded in
[`requirements.txt`](./requirements.txt).

## 2. Required Runtime Scope

To run the full pipeline, you need the Python environment from
`requirements.txt`, plus:

- working NVIDIA driver / CUDA runtime
- `ffmpeg` and `ffprobe`
- `PyNvVideoCodec` compatibility on the new machine
- the model file `CV_part/bach2.pt`
- `CV_part/time_limit.json`
- `CV_part/a1_copy_2_copy.json`

## 3. Files Needed on the New Machine

At minimum, prepare these repo files and runtime assets on the new machine:

1. The repository code, especially:
   - `CV_part/DetectionTrackingWithGPU.py`
   - `CV_part/run_peak_hours.py`
   - `CV_part/VechilCountCPU.py`
   - `CV_part/cv_paths.py`
   - `CV_part/run_dual_gpu_cv.sh`
   - `CV_part/run_full_bytetrack_pipeline.sh`
   - `CV_part/tracker_backends.py`
2. Model and metadata files:
   - `CV_part/bach2.pt`
   - `CV_part/time_limit.json`
   - `CV_part/a1_copy_2_copy.json`
3. Writable output directories for:
   - tracking CSVs
   - count CSVs
   - logs / manifests
4. Existing `GateLineJson.json` is optional
   - if it does not exist, `VechilCountCPU.py` can regenerate it from
     `a1_copy_2_copy.json`

## 4. Recommended Layout on a New Machine

One practical layout is:

```text
Macau_project/
  CV_part/
    DetectionTrackingWithGPU.py
    run_peak_hours.py
    VechilCountCPU.py
    run_dual_gpu_cv.sh
    run_full_bytetrack_pipeline.sh
    tracker_backends.py
    cv_paths.py
    bach2.pt
    time_limit.json
    a1_copy_2_copy.json
    requirements.txt
    README_count_deploy.md
  data/
    tracking_out/
    count_out/
```

You do not have to preserve the original artifact directory names if you pass
explicit CLI arguments.

## 5. Create the Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r CV_part/requirements.txt
```

Also make sure these system tools are installed:

```bash
ffmpeg
ffprobe
```

And verify GPU runtime separately:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Also verify `PyNvVideoCodec` before running a large job:

```bash
python -c "import PyNvVideoCodec; print(PyNvVideoCodec.__file__)"
```

## 6. Run the Full Pipeline

The simplest entrypoint for the full chain is:

```bash
bash CV_part/run_full_bytetrack_pipeline.sh
```

Useful environment overrides:

```bash
VIDEO_LIST=/abs/path/to/videos_all.txt
FULL_ROOT=/abs/path/to/full_run_dir
TRACKING_ROOT=/abs/path/to/full_run_dir/tracking
COUNT_ROOT=/abs/path/to/full_run_dir/count
BATCH_SIZE=512
TRACK_WORKERS=20
COUNT_PROCESSES=24
TRACKER_BACKEND=bytetrack
bash CV_part/run_full_bytetrack_pipeline.sh
```

If you want to run only the dual-GPU detection + tracking stage first:

```bash
TRACKER_BACKEND=bytetrack \
BATCH_SIZE=1024 \
TRACK_WORKERS=20 \
TRACKING_ROOT=/abs/path/to/tracking_out \
RUN_TAG=run_001 \
bash CV_part/run_dual_gpu_cv.sh /abs/path/to/videos_all.txt
```

## 7. Parameters That Matter for Speed

For the full chain, the main throughput knobs are:

- `BATCH_SIZE`
  - affects detection throughput and GPU memory usage
- `TRACK_WORKERS`
  - number of CPU worker processes used for tracking tasks submitted by
    `run_peak_hours.py`
- `COUNT_PROCESSES`
  - number of worker processes used by `VechilCountCPU.py`
- `TRACKER_BACKEND`
  - currently `bytetrack` or `oasort`

The counting-stage throughput knob remains:

- `--processes`
  - used internally by `VechilCountCPU.py`
  - controlled in the full pipeline by `COUNT_PROCESSES`

Resumed runs are incremental:

- if a tracking CSV already exists, the corresponding segment is skipped
- if both `*_Count.csv` and `*_Count.csv.ok` exist, that count segment is skipped

## 8. Expected Outputs

After a successful full run, you should see:

- tracking segment CSVs:
  - `<tracking_root>/<video_name>/<segment_name>.csv`
- merged tracking manifest:
  - `<tracking_root>/segment_manifest.csv`
- per-segment count files:
  - `<count_root>/<video_name>/<segment_name>_Count.csv`
- count completion markers:
  - `<segment_name>_Count.csv.ok`
- per-camera summary files:
  - `<count_root>/<video_name>/<video_name>_gate_summary.csv`
- worker logs:
  - `.../logs/gpu0.log`
  - `.../logs/gpu1.log`

## 9. What `requirements.txt` Does Not Cover

Even with `requirements.txt`, a new full-pipeline machine can still fail if any
of these are mismatched:

- NVIDIA driver version
- CUDA / PyTorch wheel compatibility
- `PyNvVideoCodec` wheel compatibility
- missing `ffmpeg` / `ffprobe`
- missing model / JSON resource files

That is why the deployment recipe should always be:

1. restore repo files
2. restore resource files
3. create Python environment
4. verify system tools
5. run a small smoke test before a full job

## 10. Minimal Smoke Test for the Full Chain

Before launching all videos, run a small smoke test on one video list:

```bash
printf '%s\n' /abs/path/to/sample_video.mp4 > /tmp/cv_one_video.txt

VIDEO_LIST=/tmp/cv_one_video.txt \
FULL_ROOT=/abs/path/to/test_full_run \
TRACKER_BACKEND=bytetrack \
BATCH_SIZE=256 \
TRACK_WORKERS=4 \
COUNT_PROCESSES=4 \
bash CV_part/run_full_bytetrack_pipeline.sh
```

If this succeeds and writes tracking CSVs, `_Count.csv`, and
`_gate_summary.csv`, the new machine is usually in good shape.
