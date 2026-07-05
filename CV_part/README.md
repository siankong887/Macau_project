# CV_part 启动说明

这份目录主要负责三类任务：

1. 视频检测追踪：读取 `.ts` 视频，跑 YOLO 检测和 ByteTrack/OA-SORT 追踪，输出每个片段的轨迹 CSV。
2. 过线计数：读取轨迹 CSV，根据门线配置统计车辆过线数量。
3. 图片计数：读取边境摄像头抓拍图片，用 YOLO + OpenVINO ReID 估算图片序列中的唯一车辆数。

推荐新接手的人优先使用一站式启动脚本：

- Windows：`run_full_bytetrack_pipeline.ps1`
- Linux：`run_full_bytetrack_pipeline.sh`

`run_dual_gpu_cv.sh` 已删除，不再维护旧的双卡入口。Linux 默认也是单卡；多卡需要显式设置 `GPU_IDS`。

## 目录和模型

关键文件：

- `run_full_bytetrack_pipeline.ps1`：Windows 一站式启动脚本。
- `run_full_bytetrack_pipeline.sh`：Linux 一站式启动脚本。
- `run_peak_hours.py`：检测追踪阶段主脚本。
- `VechilCountCPU.py`：过线计数阶段脚本。
- `count_vehicles_in_images_openvino.py`：图片计数脚本。
- `cv_paths.py`：集中管理默认路径。

关键资源：

- YOLO 模型：`CV_part/models/yolo/bach2.pt`
- 时间片配置：`CV_part/time_limit.json`
- 道路/门线源配置：`CV_part/a1_copy_2_copy.json`
- OpenVINO ReID 模型：`CV_part/models/openvino/public/vehicle-reid-0001/FP32/vehicle-reid-0001.xml`

## 一站式启动

一站式脚本会依次执行：

1. 检测追踪：调用 `run_peak_hours.py`
2. 合并 `segment_manifest_gpu*.csv` 为 `segment_manifest.csv`
3. 过线计数：调用 `VechilCountCPU.py`
4. 图片计数：调用 `count_vehicles_in_images_openvino.py`

### Windows

本地 Windows 默认按单张 RTX 4080 Laptop GPU / 32GB RAM 保守设置：

- `GPU_IDS=0`
- `BATCH_SIZE=128`
- `TRACK_WORKERS=14`
- `COUNT_PROCESSES=18`
- `VIDEO_EXTENSIONS=ts`

示例：

```powershell
cd C:\Users\gaoke\Documents\Project\Macau_project\Macau_project

powershell -ExecutionPolicy Bypass -File CV_part\run_full_bytetrack_pipeline.ps1 `
  -VideoDirs "C:\Users\gaoke\Documents\Project\Macau_project\temp\full_run_511\main_ts" `
  -BorderImageRoot "C:\Users\gaoke\Documents\Project\Macau_project\temp\full_run_511\border_cam_images" `
  -FullRoot "C:\Users\gaoke\Documents\Project\Macau_project\temp\full_run_511\pipeline_output"

```

如果图片目录是 crawler 输出的结构，例如：

```text
D:\data\border_cam_images\
  image1\
  image5\
```

可以用：

```powershell
powershell -ExecutionPolicy Bypass -File CV_part\run_full_bytetrack_pipeline.ps1 `
  -VideoDirs "D:\data\macau_ts" `
  -BorderImageRoot "D:\data\border_cam_images" `
  -FullRoot "D:\runs\macau_local_001"
```

如果已经有视频清单：

```powershell
powershell -ExecutionPolicy Bypass -File CV_part\run_full_bytetrack_pipeline.ps1 `
  -VideoList "D:\data\videos.txt" `
  -ImageDirs "D:\data\border_images\image1;D:\data\border_images\image5" `
  -FullRoot "D:\runs\macau_local_001"
```

### Linux

Linux 脚本默认也是单卡：

- `GPU_IDS=0`
- `BATCH_SIZE=512`
- `TRACK_WORKERS=14`
- `COUNT_PROCESSES=18`
- `VIDEO_EXTENSIONS=ts`

示例：

```bash
VIDEO_DIRS=/data/macau_ts \
IMAGE_DIRS=/data/border_images/image1:/data/border_images/image5 \
FULL_ROOT=/data/runs/macau_001 \
bash CV_part/run_full_bytetrack_pipeline.sh
```

多卡服务器需要显式设置：

```bash
GPU_IDS=0,1 \
VIDEO_DIRS=/data/macau_ts \
IMAGE_DIRS=/data/border_images/image1:/data/border_images/image5 \
FULL_ROOT=/data/runs/macau_001 \
bash CV_part/run_full_bytetrack_pipeline.sh
```

## 输出目录

一站式脚本默认输出到：

```text
<FULL_ROOT>/
  tracking/
    segment_manifest.csv
    <video_name>/<segment_name>.csv
    runs/<RUN_TAG>/
      lists/
      logs/
      segment_manifest_gpu0.csv
  count/
    <video_name>/<segment_name>_Count.csv
    <video_name>/<segment_name>_Count.csv.ok
    <video_name>/<video_name>_gate_summary.csv
  image_count/
    *_vehicle_count_results_openvino.csv
    *.log
  logs/
```

## 常用环境变量

这些变量 Windows PowerShell 和 Linux Bash 都支持。Windows 也可以用脚本参数传入 `-VideoList`、`-VideoDirs`、`-ImageDirs`、`-BorderImageRoot`、`-FullRoot`、`-RunTag`。

### 输入输出

- `RUN_TAG`：本次运行标签。默认是当前时间。
- `FULL_ROOT`：本次完整输出根目录。
- `TRACKING_ROOT`：检测追踪 CSV 输出目录，默认 `<FULL_ROOT>/tracking`。
- `COUNT_ROOT`：过线计数输出目录，默认 `<FULL_ROOT>/count`。
- `IMAGE_COUNT_ROOT`：图片计数输出目录，默认 `<FULL_ROOT>/image_count`。
- `VIDEO_LIST`：视频清单文件，一行一个视频路径。设置后优先使用它。
- `VIDEO_DIRS`：视频目录列表。Linux 用 `:`、`,` 或 `;` 分隔；Windows 用 `;` 或 `,` 分隔。
- `VIDEO_EXTENSIONS`：扫描视频后缀，默认 `ts`。可设为 `ts,mp4,avi`。
- `IMAGE_DIRS`：图片目录列表。通常放两个目录：`image1` 和 `image5`。
- `BORDER_IMAGE_ROOT`：图片根目录。脚本会自动使用 `<root>/image1` 和 `<root>/image5`。

### 模型和配置

- `MODEL_PATH`：YOLO `.pt` 模型路径，默认来自 `cv_paths.py model_pt_path`。
- `TIME_LIMIT_JSON`：视频时间片配置，默认 `CV_part/time_limit.json`。
- `SOURCE_JSON`：道路/门线源 JSON，默认 `CV_part/a1_copy_2_copy.json`。
- `GATE_LINE_JSON`：生成/读取的门线 JSON。
- `REID_MODEL_PATH`：OpenVINO ReID `.xml` 模型路径。

### 阶段开关

这些变量取 `1/true/yes/on` 表示开启，其它值表示关闭。

- `RUN_DETECTION`：是否跑检测追踪，默认 `1`。
- `RUN_GATE_COUNT`：是否跑过线计数，默认 `1`。
- `RUN_IMAGE_COUNT`：是否跑图片计数，默认 `1`。

只跑检测追踪：

```bash
RUN_GATE_COUNT=0 RUN_IMAGE_COUNT=0 bash CV_part/run_full_bytetrack_pipeline.sh
```

Windows：

```powershell
$env:RUN_GATE_COUNT = "0"
$env:RUN_IMAGE_COUNT = "0"
powershell -ExecutionPolicy Bypass -File CV_part\run_full_bytetrack_pipeline.ps1 -VideoDirs "D:\data\macau_ts"
```

只复用已有 tracking 结果跑计数：

```bash
RUN_DETECTION=0 TRACKING_ROOT=/data/runs/macau_001/tracking COUNT_ROOT=/data/runs/macau_001/count bash CV_part/run_full_bytetrack_pipeline.sh
```

### 性能参数

- `GPU_IDS`：使用哪些 GPU。默认 `0`。多卡例子：`0,1`。
- `BATCH_SIZE`：YOLO 推理 batch。显存不够先降它。
- `TRACK_WORKERS`：CPU tracking worker 数。
- `COUNT_PROCESSES`：过线计数进程数。
- `TRACKER_BACKEND`：追踪器，默认 `bytetrack`，可选 `oasort`。

推荐起点：

```text
Windows 本地：BATCH_SIZE=128, TRACK_WORKERS=14, COUNT_PROCESSES=18
Linux 单卡服务器：BATCH_SIZE=512, TRACK_WORKERS=14, COUNT_PROCESSES=18
显存吃紧：先降 BATCH_SIZE 到 64 或 32
内存吃紧：先降 TRACK_WORKERS 和 COUNT_PROCESSES
```

### 图片计数参数

- `IMAGE_CONF`：图片 YOLO 置信度，默认 `0.4`。
- `IMAGE_OPENVINO_DEVICE`：OpenVINO 设备，默认 `AUTO`。
- `IMAGE_REID_THRESH`：ReID 匹配阈值，默认 `0.50`。
- `IMAGE_MAX_DIST`：跨帧匹配最大中心距离，默认 `300`。
- `IMAGE_LOOKBACK`：向前匹配帧数，默认 `3`。
- `IMAGE_ZONE_Y_MIN`：图片统计区域顶部比例，默认 `0.10`。
- `IMAGE_ZONE_Y_MAX`：图片统计区域底部比例，默认 `0.80`。
- `IMAGE_EXTRA_ARGS`：透传给 `count_vehicles_in_images_openvino.py` 的额外参数。

例如只统计 car：

```bash
IMAGE_EXTRA_ARGS=--car-only bash CV_part/run_full_bytetrack_pipeline.sh
```

## 分阶段手动启动

一般不需要手动拆开跑，但排错时可以用。

### 只跑检测追踪

```bash
BATCH_SIZE=512 \
TRACK_WORKERS=14 \
TRACKER_BACKEND=bytetrack \
python CV_part/run_peak_hours.py \
  --video-list /path/to/videos.txt \
  --time-limit-json CV_part/time_limit.json \
  --model-path CV_part/models/yolo/bach2.pt \
  --tracking-root /data/runs/manual/tracking \
  --manifest-path /data/runs/manual/tracking/segment_manifest.csv
```

也可以先 dry-run 看分段计划：

```bash
python CV_part/run_peak_hours.py \
  --plan-only \
  --video-list /path/to/videos.txt \
  --time-limit-json CV_part/time_limit.json \
  --tracking-root /data/runs/manual/tracking \
  --manifest-path /data/runs/manual/tracking/segment_manifest.csv
```

### 只跑过线计数

```bash
python CV_part/VechilCountCPU.py \
  --csv-root /data/runs/manual/tracking \
  --count-root /data/runs/manual/count \
  --manifest-path /data/runs/manual/tracking/segment_manifest.csv \
  --source-json CV_part/a1_copy_2_copy.json \
  --gate-line-json /data/runs/manual/GateLineJson.json \
  --processes 18
```

`VechilCountCPU.py` 会跳过已经存在且带 `.ok` 标记的计数文件。

### 只跑图片计数

```bash
python CV_part/count_vehicles_in_images_openvino.py \
  --image-dir /data/border_images/image1 \
  --model CV_part/models/yolo/bach2.pt \
  --conf 0.4 \
  --output-csv /data/runs/manual/image_count/image1.csv \
  --reid-model-path CV_part/models/openvino/public/vehicle-reid-0001/FP32/vehicle-reid-0001.xml \
  --openvino-device AUTO \
  --zone-y-min 0.10 \
  --zone-y-max 0.80 \
  --reid-thresh 0.50 \
  --max-dist 300 \
  --lookback 3
```

预览统计区域：

```bash
python CV_part/count_vehicles_in_images_openvino.py \
  --image-dir /data/border_images/image1 \
  --preview
```

## 排错提示

- 找不到模型：确认 `CV_part/models/yolo/bach2.pt` 存在，或设置 `MODEL_PATH`。
- 找不到 `.ts`：确认 `VIDEO_DIRS` 指向的是视频目录，或直接设置 `VIDEO_LIST`。
- CUDA OOM：降低 `BATCH_SIZE`。
- 内存占用太高：降低 `TRACK_WORKERS` 和 `COUNT_PROCESSES`。
- 图片计数找不到 ReID：确认 `REID_MODEL_PATH` 指向 `vehicle-reid-0001.xml`。
- Windows 下不要跑 `.sh`；用 `.ps1`。
