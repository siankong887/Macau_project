"""按整点锚点等分的分段批处理脚本。

核心能力：
1. 读取 `time_limit.json` 中每个摄像头的整点锚点，按相邻锚点等分为 4 个片段。
2. 同一视频按时间顺序连续处理；每完成一个片段就把检测结果提交给 CPU 多进程追踪。
3. 支持 `--plan-only` dry-run，在没有 GPU 依赖的机器上也能生成分段清单与 manifest。

用法示例：
    python CV_part/run_peak_hours.py --plan-only --video-list videos.txt
    python CV_part/run_peak_hours.py E:/encoded_1_fixed/a_1.mp4
"""

import argparse
import csv
import gc
import json
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import threading
import time

from cv_paths import CVPaths

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
NUM_TRACK_WORKERS = int(os.getenv("TRACK_WORKERS", "12"))
FLUSH_INTERVAL = 5
FPS = 30.0
GPU_ID = 0
PATHS = CVPaths.from_file(__file__)
DEFAULT_VIDEO_DIRS = [str(path) for path in PATHS.default_video_dirs]

_pending_counter = None


def _init_worker(counter):
    global _pending_counter
    _pending_counter = counter


def time_str_to_seconds(time_str):
    h, m, s = map(int, time_str.split(":"))
    return h * 3600 + m * 60 + s


def seconds_to_time_str(total_seconds):
    whole = int(round(total_seconds)) % 86400
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def time_str_to_label(time_str):
    return time_str.replace(":", "_")


def get_video_duration(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


def read_video_list(list_path):
    videos = []
    with open(list_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            videos.append(os.path.abspath(line))
    return videos


def parse_args():
    default_model_path = str(PATHS.model_pt_path)
    default_time_limit_json = str(PATHS.time_limit_json_path)
    default_tracking_root = str(PATHS.tracking_root)
    default_manifest_path = os.path.join(default_tracking_root, "segment_manifest.csv")

    parser = argparse.ArgumentParser(description="按整点锚点连续处理视频片段。")
    parser.add_argument("videos", nargs="*", help="直接指定要处理的视频路径。")
    parser.add_argument(
        "--video-list",
        help="文本文件，逐行列出要处理的视频路径。",
    )
    parser.add_argument(
        "--video-dir",
        action="append",
        dest="video_dirs",
        help="追加一个自动扫描视频的目录。",
    )
    parser.add_argument(
        "--time-limit-json",
        default=default_time_limit_json,
        help="time_limit.json 路径。",
    )
    parser.add_argument(
        "--model-path",
        default=default_model_path,
        help="PyTorch .pt 模型路径。",
    )
    parser.add_argument(
        "--tracking-root",
        default=default_tracking_root,
        help="追踪明细 CSV 根目录。",
    )
    parser.add_argument(
        "--manifest-path",
        default=default_manifest_path,
        help="segment manifest 输出路径。",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="只生成并打印分段计划，不执行检测追踪。",
    )
    return parser.parse_args()


def collect_videos(args):
    all_videos = []

    if args.videos:
        all_videos.extend(os.path.abspath(p) for p in args.videos)

    if args.video_list:
        all_videos.extend(read_video_list(args.video_list))

    if not all_videos:
        scan_dirs = args.video_dirs if args.video_dirs else DEFAULT_VIDEO_DIRS
        for folder in scan_dirs:
            if not os.path.isdir(folder):
                print(f"警告: 视频目录不存在，跳过读取: {folder}")
                continue
            for fn in sorted(os.listdir(folder)):
                if fn.endswith((".mp4", ".avi")) and not fn.startswith("."):
                    all_videos.append(os.path.join(folder, fn))

    unique_videos = []
    seen = set()
    for path in all_videos:
        norm = os.path.abspath(path)
        if norm in seen:
            continue
        seen.add(norm)
        unique_videos.append(norm)
    return unique_videos


def video_name_to_cam_key(video_name):
    return "cam_" + video_name.replace("_", "")


def append_segment(rows, video_name, cam_key, video_path, tracking_root,
                   start_sec, end_sec, start_frame, end_frame, is_tail, status):
    if end_frame <= start_frame:
        return

    start_time = seconds_to_time_str(start_sec)
    end_time = seconds_to_time_str(end_sec)
    segment_name = f"{video_name}_{time_str_to_label(start_time)}__{time_str_to_label(end_time)}"
    tracking_dir = os.path.join(tracking_root, video_name)
    rows.append({
        "video_name": video_name,
        "cam_key": cam_key,
        "video_path": video_path,
        "segment_name": segment_name,
        "start_time": start_time,
        "end_time": end_time,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "duration_sec": round(max(0.0, end_sec - start_sec), 3),
        "is_tail": bool(is_tail),
        "status": status,
        "tracking_csv_path": os.path.join(tracking_dir, f"{segment_name}.csv"),
    })


def build_segments_for_video(video_name, cam_key, video_path, entries, tracking_root, video_duration=None):
    rows = []
    anchor_secs = []

    for entry in entries:
        tl = str(entry.get("time_limit", "")).strip()
        if not tl:
            continue
        anchor_secs.append(time_str_to_seconds(tl))

    if len(anchor_secs) < 2:
        return rows

    last_step = None
    for idx in range(len(anchor_secs) - 1):
        start_anchor = anchor_secs[idx]
        end_anchor = anchor_secs[idx + 1]
        if end_anchor <= start_anchor:
            print(f"警告: {video_name} 的锚点顺序异常: {seconds_to_time_str(start_anchor)} -> {seconds_to_time_str(end_anchor)}")
            continue

        step = (end_anchor - start_anchor) / 4.0
        last_step = step
        sec_bounds = [start_anchor + step * i for i in range(5)]
        frame_bounds = [int(round(boundary * FPS)) for boundary in sec_bounds]

        for part_idx in range(4):
            append_segment(
                rows,
                video_name,
                cam_key,
                video_path,
                tracking_root,
                sec_bounds[part_idx],
                sec_bounds[part_idx + 1],
                frame_bounds[part_idx],
                frame_bounds[part_idx + 1],
                is_tail=False,
                status="planned",
            )

    if last_step is None:
        return rows

    last_anchor = anchor_secs[-1]
    if video_duration is None:
        rows.append({
            "video_name": video_name,
            "cam_key": cam_key,
            "video_path": video_path,
            "segment_name": "",
            "start_time": seconds_to_time_str(last_anchor),
            "end_time": "",
            "start_frame": int(round(last_anchor * FPS)),
            "end_frame": "",
            "duration_sec": "",
            "is_tail": True,
            "status": "tail_pending_duration_check",
            "tracking_csv_path": "",
        })
        return rows

    current = last_anchor
    while current < video_duration - 1e-6:
        next_sec = min(current + last_step, video_duration)
        start_frame = int(round(current * FPS))
        end_frame = int(round(next_sec * FPS))
        append_segment(
            rows,
            video_name,
            cam_key,
            video_path,
            tracking_root,
            current,
            next_sec,
            start_frame,
            end_frame,
            is_tail=True,
            status="planned",
        )
        current = next_sec

    return rows


def write_manifest(manifest_path, rows):
    output_dir = os.path.dirname(manifest_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def print_plan_summary(video_segments):
    print(f"共生成 {sum(len(rows) for rows in video_segments.values())} 条 manifest 记录")
    for video_name, rows in video_segments.items():
        planned = [row for row in rows if row["status"] == "planned"]
        tail_pending = [row for row in rows if row["status"] == "tail_pending_duration_check"]
        print(f"  {video_name}: planned={len(planned)}, tail_pending={len(tail_pending)}")
        for preview in planned[:3]:
            print(
                f"    {preview['segment_name']}: "
                f"{preview['start_time']} -> {preview['end_time']} | "
                f"frames {preview['start_frame']}:{preview['end_frame']}"
            )


def load_runtime_dependencies():
    import torch
    import PyNvVideoCodec as nvc

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from DetectionTrackingWithGPU import (
        _decode_and_stack,
        non_max_suppression,
    )

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    return {
        "torch": torch,
        "nvc": nvc,
        "_decode_and_stack": _decode_and_stack,
        "non_max_suppression": non_max_suppression,
    }


def load_model(runtime, model_path):
    torch = runtime["torch"]
    from ultralytics import YOLO

    yolo = YOLO(model_path)
    yolo.to(f"cuda:{GPU_ID}")
    model = yolo.model
    model.eval()
    if torch.cuda.is_available():
        model.half()
    return model


def wait_for_backpressure(pending_counter):
    max_pending = NUM_TRACK_WORKERS + 2
    while pending_counter.value >= max_pending:
        print(
            f"  !! 【内存背压预警】: 当前待处理追踪任务 {pending_counter.value} 个，"
            f"超过安全阈值 {max_pending}，暂停投递...",
            flush=True,
        )
        time.sleep(10)
        gc.collect()


def _track_and_count(segment_dets, csv_path):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from DetectionTrackingWithGPU import track_segment

        track_segment(segment_dets, csv_path)
        print(f"    跟踪完成, 保存结果至: {os.path.basename(csv_path)}", flush=True)
    except Exception as exc:
        print(f"    跟踪失败, 文件 {os.path.basename(csv_path)} 发生错误: {exc}", flush=True)
    finally:
        with _pending_counter.get_lock():
            _pending_counter.value -= 1


def move_decoder_to_frame(decoder, current_frame, target_frame, can_seek):
    if current_frame == target_frame:
        return current_frame, True

    if current_frame > target_frame:
        print(f"  !! 时序异常: 当前解码位置 {current_frame} 已超过目标起点 {target_frame}")
        return current_frame, False

    if can_seek:
        decoder.seek_to_index(target_frame)
        return target_frame, True

    print(f"  >> 目标片段起点 {target_frame} 需要顺序跳帧，当前位于 {current_frame}")
    while current_frame < target_frame:
        need = min(BATCH_SIZE, target_frame - current_frame)
        skipped = decoder.get_batch_frames(need)
        if not skipped:
            return current_frame, False
        current_frame += len(skipped)
    return current_frame, True


def process_segment(decoder, runtime, model, pool, pending_counter,
                    current_frame, segment_row, can_seek):
    torch = runtime["torch"]
    decode_and_stack = runtime["_decode_and_stack"]
    non_max_suppression = runtime["non_max_suppression"]

    total_frames = int(segment_row["end_frame"]) - int(segment_row["start_frame"])
    if total_frames <= 0:
        return current_frame, 0

    current_frame, ready = move_decoder_to_frame(
        decoder,
        current_frame,
        int(segment_row["start_frame"]),
        can_seek,
    )
    if not ready:
        print(f"  !! 无法抵达片段起点，跳过 {segment_row['segment_name']}")
        return current_frame, 0

    decode_q = queue.Queue(maxsize=2)

    def safe_decode():
        try:
            decode_and_stack(decoder, BATCH_SIZE, total_frames, decode_q)
        except Exception as exc:
            print(f"    解码线程运行异常: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            decode_q.put(None)

    decode_thread = threading.Thread(target=safe_decode, daemon=True)
    decode_thread.start()

    segment_dets = []
    pending_tensors = []
    pending_ids = []
    pending_counts = []
    process_count = 0
    batch_count = 0
    seg_start = time.time()
    progress_interval = 100

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

        for bi, det in enumerate(dets_list):
            pending_ids.append(process_count + bi)
            det_count = 0
            if det is not None and len(det) > 0:
                pending_tensors.append(det)
                det_count = len(det)
            pending_counts.append(det_count)

        process_count += got
        batch_count += 1

        if batch_count % progress_interval == 0:
            elapsed = time.time() - seg_start
            fps_now = process_count / elapsed if elapsed > 0 else 0
            pct = process_count / total_frames * 100 if total_frames > 0 else 0
            print(
                f"    进度: {process_count}/{total_frames} ({pct:.0f}%), "
                f"平均速度: {fps_now:.0f} FPS, 用时: {elapsed:.0f}s",
                flush=True,
            )

        if pending_ids and batch_count % FLUSH_INTERVAL == 0:
            if pending_tensors:
                big_gpu = torch.cat(pending_tensors, dim=0)
                big_np = big_gpu.cpu().numpy()
            else:
                big_np = np.empty((0, 6), dtype=np.float32)
            segment_dets.append((list(pending_ids), big_np, list(pending_counts)))
            pending_tensors.clear()
            pending_ids = []
            pending_counts = []

    if pending_ids:
        if pending_tensors:
            big_gpu = torch.cat(pending_tensors, dim=0)
            big_np = big_gpu.cpu().numpy()
        else:
            big_np = np.empty((0, 6), dtype=np.float32)
        segment_dets.append((list(pending_ids), big_np, list(pending_counts)))

    decode_thread.join()
    current_frame = int(segment_row["start_frame"]) + process_count

    if process_count <= 0:
        print(f"  !! {segment_row['segment_name']} 没有读到任何帧，可能已到视频末尾。")
        return current_frame, 0

    os.makedirs(os.path.dirname(segment_row["tracking_csv_path"]), exist_ok=True)
    wait_for_backpressure(pending_counter)
    with pending_counter.get_lock():
        pending_counter.value += 1
    pool.apply_async(_track_and_count, (segment_dets, segment_row["tracking_csv_path"]))

    elapsed = time.time() - seg_start
    fps_val = process_count / elapsed if elapsed > 0 else 0
    print(
        f"  {segment_row['segment_name']} 推理完成: 共处理 {process_count} 帧, "
        f"耗时 {elapsed:.1f}s, 平均推断速度 {fps_val:.0f} FPS → 已提交 CPU 追踪",
        flush=True,
    )

    del segment_dets
    gc.collect()
    return current_frame, process_count


def process_video(video_path, video_rows, runtime, model, pool, pending_counter):
    nvc = runtime["nvc"]
    can_seek = "encoded_1_fixed" in video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    try:
        decoder = nvc.SimpleDecoder(
            enc_file_path=video_path,
            gpu_id=GPU_ID,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGBP,
        )
    except Exception as exc:
        print(f"[{video_name}] 解码器初始化失败: {exc}")
        return False

    current_frame = 0
    planned_rows = [row for row in video_rows if row["status"] == "planned"]
    for row in planned_rows:
        csv_path = row["tracking_csv_path"]
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            print(f"  跳过已完成片段: {row['segment_name']}")
            if can_seek:
                current_frame = int(row["end_frame"])
            continue

        print(
            f"  [片段] {row['segment_name']} | {row['start_time']} -> {row['end_time']} | "
            f"frames {row['start_frame']}:{row['end_frame']}"
        )
        current_frame, processed = process_segment(
            decoder,
            runtime,
            model,
            pool,
            pending_counter,
            current_frame,
            row,
            can_seek,
        )
        if processed <= 0:
            break

    del decoder
    return True


def main():
    args = parse_args()

    with open(args.time_limit_json, "r", encoding="utf-8") as f:
        time_limit_data = json.load(f)

    all_videos = collect_videos(args)
    if not all_videos:
        print("未发现任何符合要求的视频文件可处理，脚本退出。")
        return

    video_segments = {}
    manifest_rows = []
    skipped = 0
    for video_path in all_videos:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cam_key = video_name_to_cam_key(video_name)

        if cam_key not in time_limit_data:
            print(f"{video_name}: 缺少对应 time_limit 标定信息，跳过。")
            skipped += 1
            continue

        if os.path.exists(video_path):
            video_duration = get_video_duration(video_path)
        else:
            video_duration = None
            print(f"警告: 视频文件不存在，dry-run 仅基于文件名规划: {video_path}")

        rows = build_segments_for_video(
            video_name,
            cam_key,
            video_path,
            time_limit_data[cam_key],
            args.tracking_root,
            video_duration=video_duration,
        )
        if not rows:
            print(f"{video_name}: 未生成任何片段，跳过。")
            skipped += 1
            continue

        video_segments[video_name] = rows
        manifest_rows.extend(rows)

    write_manifest(args.manifest_path, manifest_rows)
    print(f"已生成 manifest: {args.manifest_path}")
    print_plan_summary(video_segments)

    if args.plan_only:
        print("当前为 --plan-only 模式，不执行检测追踪。")
        return

    runtime = load_runtime_dependencies()
    model = load_model(runtime, args.model_path)
    print("检测模型加载完毕，使用体系: PyTorch")

    ctx = mp.get_context("spawn")
    pending_counter = ctx.Value("i", 0)
    pool = ctx.Pool(NUM_TRACK_WORKERS, initializer=_init_worker, initargs=(pending_counter,))

    total_start = time.time()
    processed = 0
    try:
        for index, video_path in enumerate(all_videos, start=1):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            if video_name not in video_segments:
                continue
            if not os.path.exists(video_path):
                print(f"[{index}/{len(all_videos)}] {video_name}: 视频文件不存在，跳过实际执行。")
                skipped += 1
                continue

            print(f"\n{'=' * 60}")
            print(f"[{index}/{len(all_videos)}] 正在处理视频文件: {video_name}")
            video_start = time.time()
            ok = process_video(
                video_path,
                video_segments[video_name],
                runtime,
                model,
                pool,
                pending_counter,
            )
            if ok:
                processed += 1
            else:
                skipped += 1
            elapsed = time.time() - video_start
            print(f"  [{video_name} 最终状态] 本视频处理结束, 用时 {elapsed:.1f} 秒")
    finally:
        print(f"\n{'=' * 60}")
        print(f"GPU 的分发作业已经全部达成，静待剩余 {pending_counter.value} 个追踪任务收尾...")
        pool.close()
        while pending_counter.value > 0:
            print(f"  CPU 跟踪分析倒数中: 尚未结算任务数 {pending_counter.value} 个...", flush=True)
            time.sleep(30)
        pool.join()

    total_elapsed = time.time() - total_start
    print(
        f"\n【全体任务竣工】: 共完成 {processed} 支视频, "
        f"跳过 {skipped} 支, 总耗时 {total_elapsed:.1f} 秒"
    )


if __name__ == "__main__":
    main()
