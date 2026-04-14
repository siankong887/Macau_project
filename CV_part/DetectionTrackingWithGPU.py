"""DetectionTrackingWithGPU.py

核心功能说明:
1. 此脚本提供了 GPU 目标检测与 CPU 多进程目标跟踪的基础架构模块。
2. 封装了基于 PyTorch 的 YOLO 模型加载流程，以适应视频目标智能检测需求。
3. 提供了解码 (PyNvVideoCodec)、检测推断、边界推断、SORT 追踪处理的流水线功能类和函数。
   它被其他主控脚本（如 run_peak_hours.py）调用来执行具体的分析任务。

性能与架构设计:
1. GIL 安全与并发：由 Python 多线程负责将 C++ 层面的 NV 解码操作释放 GIL 优势发挥出来，同时不阻塞主线程做 YOLO 推演。
2. 零拷贝思想：尝试使用 `torch.from_dlpack` 和直接指针操作在不引发 CPU 内存和 GPU 显存之间拷贝的情况下处理流式数据。
3. 异步 CSV 写入机制：把耗时的硬盘 IO 独立成专门的守护线程和队列，保护高吞吐。
"""

from ultralytics import YOLO
try:
    from ultralytics.utils.ops import non_max_suppression
except ImportError:
    from ultralytics.utils.nms import non_max_suppression
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import IterableSimpleNamespace
import multiprocessing as mp
import os
import csv
import json
import time
import torch
import numpy as np
import PyNvVideoCodec as nvc
import threading, queue
import shutil

from cv_paths import CVPaths

CV_PATHS = CVPaths.from_file(__file__)

# HDD→SSD 临时缓存目录（为了克服机械硬盘随机读取极慢劣势，视频处理前先自动拷到 SSD）
TEMP_VIDEO_DIR = str(CV_PATHS.temp_video_dir)

# 小幅通用加速: 启用 cuDNN 自动内核选择和利用 TF32 加速矩阵乘运算
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ByteTrack 跟踪器配置（与 ultralytics 的 bytetrack.yaml 默认值一致，为了不读盘而在代码内硬编码）
_TRACKER_ARGS = IterableSimpleNamespace(
    track_high_thresh=0.25, # 首次激活轨迹的最低置信度阈值
    track_low_thresh=0.1,   # 允许维持追踪的最低置信度阈值 (ByteTrack 核心原理：兼顾低分段框)
    new_track_thresh=0.25,  # 诞生新轨迹所需置信度
    track_buffer=30,        # 轨迹中断缓冲保留帧数 (1秒钟/FPS=30)
    match_thresh=0.8,       # 匈牙利算法等匹配框之间的最大IOU距离
    fuse_score=True,        # 融合目标框得分与位置距离权重
)


class _DetResult:
    """
    轻量化的检测结果包装器。
    用于绕过原 ultralytics 的厚重 Results 类，为其提供兼容 ByteTracker.update() 需要的最精简属性接口。
    """
    __slots__ = ("conf", "xywh", "cls") # 固定内存插槽防止动态字典开销

    def __init__(self, conf, xywh, cls):
        self.conf = np.atleast_1d(np.asarray(conf)) # 置信度概率
        self.xywh = np.asarray(xywh)                # 边界框位置坐标 (中心X, 中心Y, 宽度W, 高度H)
        self.cls = np.atleast_1d(np.asarray(cls))   # 判定出的分类 ID

        # ByteTrack.update() 会对 results 做布尔切片；单元素索引时必须保持 2D/1D 形状不坍缩
        if self.xywh.ndim == 1:
            self.xywh = self.xywh[None, :]

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        conf = self.conf[idx]
        xywh = self.xywh[idx]
        cls = self.cls[idx]

        if np.isscalar(conf):
            conf = np.asarray([conf])
            xywh = np.asarray([xywh])
            cls = np.asarray([cls])

        return _DetResult(conf=conf, xywh=xywh, cls=cls)


class AsyncCSVWriter:
    """
    高性能的非阻塞异步 CSV 表记录器：
    应对每秒万级别的逐帧框坐标输出。
    - 采用单进单出的 queue 架构，使主计算进程只须将列表推给内存堆即立刻返回。
    - 守护专属线程从队列消费内容，实施积累式的批量写入 `writerows`。
    - 指定冲刷阈值 (flush_every) 定期调用 fsync 保存硬盘以防断电丢失。
    """
    def __init__(self, path, header, flush_every=10000, buf_bytes=1024 * 1024):
        self.path = path        # 输出目标文件绝对路径
        self.header = header    # 表格头部标识
        self.flush_every = int(flush_every) # 冲刷底线阀值
        self.buf_bytes = buf_bytes # 定义 IO 二进制缓冲块体积（这里大约是 1MB）
        self.q = queue.Queue(maxsize=8192)
        self.t = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def start(self):
        """点火起跑后台写入进程"""
        if not self._started:
            self._started = True
            self.t.start()

    def write_rows(self, rows):
        """非阻塞性接收并提交一行行列表数据集"""
        # rows 必须为：list of list 嵌套结构
        if rows:
            self.q.put(rows)

    def close(self):
        """主动发送中止信号 None 给消费队列并等待后台写完数据干净断网"""
        self.q.put(None)
        self.t.join()

    def _run(self):
        """在专用守护线程中永久存活并不断拉取缓存池数据的具体死循环"""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # 激活带内存缓冲的二进制文本 IO 文件句柄
        with open(self.path, "w", newline="", buffering=self.buf_bytes, encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
            to_flush = 0

            while True:
                item = self.q.get()
                if item is None:
                    break   # 接到哨兵退出死循环
                writer.writerows(item)
                to_flush += len(item)

                # 到达一定缓冲则强制由操作系统 OS cache 下刷到真实扇区
                if to_flush >= self.flush_every:
                    f.flush()
                    to_flush = 0


def track_segment(segment_dets, csv_path, frame_rate=30):
    """
    多进程 Worker 池的封装主执行体（在一个 CPU 核心子进程上独立调用）：
    拿到一个片段由 GPU 计算好的全部检测结果，通过纯 CPU 跑 ByteTrack 跟踪器，以分析出前后关联轨迹，末了写进 CSV。

    Args:
        segment_dets: list of (frame_ids, big_np, counts) 嵌套汇总的大数组
                      frame_ids: 每张有检测物的帧 ID 的列表
                      big_np: 平铺式多框联接的大张量 ndarray [总检测框数, 6维] -> 即(x1, y1, x2, y2, 置信度conf, 类别分类cls)
                      counts: 因为上述是一维平摊堆叠的，这里记录着这幅画面原来有几个人(截断数)的列表
        csv_path: 目标 CSV 存放物理路径
        frame_rate: 被监控路段对应流的帧速率，用于维持轨迹生存长短预测
    """
    tracker = BYTETracker(args=_TRACKER_ARGS, frame_rate=frame_rate)
    header = ["frame_id", "track_id", "cls", "center_x", "center_y",
              "width", "height", "conf"]
    writer = AsyncCSVWriter(csv_path, header)
    writer.start()

    reset_count = 0 # ByteTrack 中的卡尔曼滤波器可能会矩阵异常导致抛错挂起，我们捕获重置它
    tracking_input_frames = 0 # 统计真正送入跟踪器的有检测帧数量
    written_rows = 0          # 统计最终成功写出的轨迹行数

    # 将汇总压缩好的数据序列重新拆解播放出来
    for frame_ids, big_np, counts in segment_dets:
        offset = 0 # 全局内存游标点
        for frame_id, count in zip(frame_ids, counts):
            # 获取对应的一帧里边的所有目标的列表框（这里是极快的不搬运数据的 numpy view 浅切片）
            det_np = big_np[offset:offset + count]
            offset += count

            # 边框数学形态转换：将原数组的对角模型 [左上x, 左上y, 右下x, 右下y]
            # 反算转换为 ByteTracker 和 Yolo 通用兼容期望的物理中心加体宽高的模型 (cx, cy, w, h)
            xywh = np.column_stack([
                (det_np[:, 0] + det_np[:, 2]) / 2,   # cx = (x1+x2)/2
                (det_np[:, 1] + det_np[:, 3]) / 2,   # cy = (y1+y2)/2
                det_np[:, 2] - det_np[:, 0],         # width = x2 - x1
                det_np[:, 3] - det_np[:, 1],         # height = y2 - y1
            ])

            # 使用轻量类对象包一层兼容 ultralytics 检测输出的形状模式，送去预测更新轨迹
            wrapped = _DetResult(conf=det_np[:, 4], xywh=xywh, cls=det_np[:, 5])
            tracking_input_frames += 1

            # 安全触发 tracker 推理与异常恢复重建
            try:
                tracks = tracker.update(wrapped, None) # None 为附加保留参数不用理会
            except Exception:
                # 卡尔曼滤波器 (Kalman Filter) 的协方差矩阵有时碰到奇异点崩溃溢出 -> 销毁实例完全重置 tracker 再次尝试
                tracker = BYTETracker(args=_TRACKER_ARGS, frame_rate=frame_rate)
                reset_count += 1
                try:
                    tracks = tracker.update(wrapped, None)
                except Exception:
                    continue  # 若二次重试仍旧挽救不了产生不可抗拒之暴毙错，则跳过放弃该帧不再纠结

            if len(tracks) == 0:
                continue # 本帧画面无人或者目标无法匹配忽略

            # 解析完成出的具有上下文连贯 track_id 身份识别的目标信息存入 CSV 数组队列缓冲中
            rows = []
            for t in tracks:
                # 记录：[x1, y1, x2, y2, track_id, conf, cls, info] // t 的具体内部形态索引
                cx = (t[0] + t[2]) / 2
                cy = (t[1] + t[3]) / 2
                w = t[2] - t[0]
                h = t[3] - t[1]
                rows.append([
                    frame_id, int(t[4]), int(t[6]),  # 帧编号, 身份标识追踪ID, 归属种类表
                    round(cx, 2), round(cy, 2),      # X轴坐标点(2小数点后精度压缩存储), Y轴同上
                    round(w, 2), round(h, 2),        # 体型长度数据
                    round(t[5], 3)                   # 当前融合确信度评分
                ])
            if rows:
                written_rows += len(rows)
                writer.write_rows(rows)

    if reset_count > 0:
        print(f"    预警：跟踪器在此环节曾发生 {reset_count} 次重大数学重置故障: {os.path.basename(csv_path)}", flush=True)

    if tracking_input_frames > 0 and written_rows == 0:
        print(
            f"    严重预警：本片段共有 {tracking_input_frames} 帧检测结果送入跟踪器，但最终没有写出任何轨迹行: "
            f"{os.path.basename(csv_path)}",
            flush=True,
        )

    writer.close()
    return csv_path


def _decode_and_stack(decoder, batch_size, total_frames, out_queue):
    """
    负责统筹 NvCodec 硬件解码循环工作的专属线程：连续解码图片帧组合为 batch (stack)，然后推进入待推断队列。
    利用到高级 GIL 安全特性：调用底层 C 扩展 `get_batch_frames()` 时刻它主动放弃占有了系统 CPU 执行许可 GIL。
    与此同时 TRT 主线路也在调用 C++ 硬件也放弃，因此解码准备和算图推算是可以处于 100% 同一层级物理并行执行无干涉抢占。
    """
    decoded = 0 # 记录成功产出的像素桢数

    while decoded < total_frames:
        need = min(batch_size, total_frames - decoded)  # 最后不够塞满整个 batch_size 时的防止超限裁剪

        # 通知驱动程序硬件抽出 need 数量画面的数组指针
        frames = decoder.get_batch_frames(need)
        if not frames:
            break

        got = len(frames)
        # 高级手法 DLPack 指针迁移：不涉及数据数值拷贝的，仅交涉指配所有权的方式转移到了 PyTorch 系统 Tensor 数据块。
        t_list = [torch.from_dlpack(f) for f in frames]
        # 一次内存空间分配复制重新排布为大块连续 Tensor：维度形态转换为 (B, 3色, H高, W宽) // uint8 类型停泊在 GPU 上。
        batch = torch.stack(t_list, dim=0)

        # 极重要！务必要主动打断废弃 t_list 和 frames 强引用，以使 C 底层知道 Python 不需要了这几片缓冲池可以拿去解下一波帧重用了。
        del t_list, frames
        out_queue.put((batch, got))
        decoded += got

    out_queue.put(None)  # 循环结束抛个终结哨兵标记：告知主线排期任务该杀青打板停止消费了


def video_process(VideoInfo):
    """
    旧版本的完整独立视频流处理器（该设计被抽象合并至 run_peak_hours）。
    用于独立完整运行全部时相队列，并且自动适配了有无索引错误视频的补全 seek 问题，并在主路径上进行了 HDD-SSD 文件迁移功能。
    """
    # 展开变量
    VideoPath, CsvFolderPath, ModelPath, Index, TimeLimitList = VideoInfo

    # flag标识: 判断是否可以使用 seek 视频帧命令跳转（已修复过的视频支持）
    can_seek = "encoded_1_fixed" in VideoPath

    # --- 性能优化：自动将机械硬盘中超大的影片移动倒高IO的SSD存储驱动器中临时缓冲 ---
    temp_video_path = None
    # 截取源盘符和终盘符判定是否处于同一分区物理硬件盘
    src_drive = os.path.splitdrive(os.path.abspath(VideoPath))[0].upper()
    dst_drive = os.path.splitdrive(os.path.abspath(TEMP_VIDEO_DIR))[0].upper()

    if src_drive != dst_drive:
        os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)
        temp_video_path = os.path.join(TEMP_VIDEO_DIR, os.path.basename(VideoPath))
        print(f"磁盘搬运中：将机械磁盘读取内容置换转移到高速SSD磁盘临时缓存加速域: {VideoPath} -> {temp_video_path}")
        copy_start = time.time()
        shutil.copy2(VideoPath, temp_video_path)
        print(f"搬运算区完毕，整体复制操作所消耗去 {time.time() - copy_start:.1f} 秒")
        VideoPath = temp_video_path

    # --- GPU 集群多路设备的调配配置 ---
    gpu_count = torch.cuda.device_count()
    gpu_id = Index % gpu_count if gpu_count > 0 else 0  # 负载均衡将多个视频流分配给多路并机系统 (如有多张 3090, 4090 并排调度)
    device_str = f"cuda:{gpu_id}"

    # 判断当前主板能不能承载原生的快速处理（针对半精度的兼容性校验测频）
    use_half = torch.cuda.is_available()

    # 初始化启动推理决策模型核心
    yolo = YOLO(ModelPath)
    yolo.to(device_str)
    internal_model = yolo.model
    internal_model.eval()
    if use_half:
        internal_model.half()

    print(f"启动新工位进程 | 处理的影视轨名称: {os.path.basename(VideoPath)} | 所配发核心硬件: {device_str} | 底层模型加速算法: PyTorch")

    # 便携辅助小函数
    def time_str_to_seconds(time_str):
        try:
            h, m, s = map(int, time_str.split(":"))
            return h * 3600 + m * 60 + s
        except ValueError:
            return float("inf")

    # 全局建立解码器实例，跨循环复用生命周期
    try:
        decoder = nvc.SimpleDecoder(
            enc_file_path=VideoPath,
            gpu_id=gpu_id,
            use_device_memory=True, # 直接吐出置留于显存的像素结果
            output_color_type=nvc.OutputColorType.RGBP  # Tensor网络需要 Planar 形态的三维矩阵（RGB 平铺图排列表）
        )
        meta = decoder.get_stream_metadata()

        # =================【特别防错】==================
        # 源生视频内部含带的元数据对于 FPS(帧率) 很不牢靠。有的视频因压制问题时间戳错乱(33秒报时但居然能播放28小时)。
        # 不受制于这数据而强行进行硬锁定成工业标准 fps = 30.0！
        fps = 30.0
        # ===============================================

        width, height = meta.width, meta.height
        meta_fps = float(meta.avg_frame_rate) if getattr(meta, "avg_frame_rate", 0) else 0
        print(f"NV解码硬件就绪备齐: 检测得长宽分辨率 {width}x{height} | 重置覆盖认定fps: {fps} (底层系统原有探测到虚假帧率为: {meta_fps:.3f})")
    except Exception as e:
        print(f"解码器引擎因硬件/路径权限/不支持格式出错彻底初始化停摆崩溃阻断: {VideoPath}, 具体回执抛出错误: {e}")
        return

    # 从系统 OS 环境量提取批处理定义大小和分核工蜂总数，默认留底策略
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "512"))
    NUM_TRACK_WORKERS = int(os.getenv("TRACK_WORKERS", "16"))
    FLUSH_INTERVAL = 5  # 定期清理 GPU Tensor 刷入 CPU 进程堆时间长度设置

    # 循环播放位置计数器 (为了当某些无法安全使用 seek 跳转命令操作的老录像破录像提供纯遍历解算的“人肉跳动法”依赖标志戳)
    current_frame = 0

    # 本视频分配建立的专属后台跟踪异步处理器工人系统
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(NUM_TRACK_WORKERS)
    async_results = []

    video_start_time = time.time()
    TimeLimitIndex = 0

    # 时序序列轮流播发
    while TimeLimitIndex + 1 < len(TimeLimitList):
        CurrentTimeLimit = TimeLimitList[TimeLimitIndex]
        NextTimeLimit = TimeLimitList[TimeLimitIndex + 1]

        CsvFilePath = os.path.join(
            CsvFolderPath,
            f"{os.path.basename(VideoPath).split('.')[0]}_{TimeLimitIndex}.csv"
        )

        CurrentTimeLimitSeconds = time_str_to_seconds(CurrentTimeLimit)
        NextTimeLimitSeconds = time_str_to_seconds(NextTimeLimit)
        TimeLimitIndex += 1

        try:
            # 决定这一片段的工作头尾截断数量
            TotalFrames = int((NextTimeLimitSeconds - CurrentTimeLimitSeconds) * fps)
            StartFrame = int(CurrentTimeLimitSeconds * fps)

            # --- 安全帧起止寻址器与跳转 ---
            if current_frame < StartFrame:
                if can_seek:
                    # 只有修复过头信息的能直接调底层 NvDecAPI 跳到对应宏块直接索引过去
                    decoder.seek_to_index(StartFrame)
                    current_frame = StartFrame
                    print(f"驱动控制光头直接跳转飞跃到预设起始侦测宏块 {StartFrame}")
                else:
                    # 对于损坏文件，进行漫长而苦闷的无效解码空投。全速抽走不要的东西直接抛散让系统忽略丢包！
                    skip_count = StartFrame - current_frame
                    print(f"检测到元错误坏帧流，执行慢速暴力丢包废除掉 {skip_count} 帧以前进到达业务首发预设目标点 (当前位流指点: {current_frame}, 拟定目的地: {StartFrame})...")
                    while current_frame < StartFrame:
                        need = min(BATCH_SIZE, StartFrame - current_frame)
                        skipped = decoder.get_batch_frames(need)
                        if not skipped:
                            print(f"惊悚警告: 在漫不经心的跳帧穿越过程中竟然发现视频的实际末尾长度已到尽头被腰斩了! 当前抛置: {current_frame}, 既定目所: {StartFrame}")
                            break
                        current_frame += len(skipped)
            elif current_frame > StartFrame:
                print(f"注意时序错乱隐患! 当前已经驶过了帧 {current_frame} 可是现在目标需要截获退后的帧 {StartFrame} 段！您的外部时间分配或者配置绝对有了严重重复逻辑的颠倒错误。")

            seg_start_time = time.time()
            print(
                f"\n=========\n即将执行下述片面核心批次渲染分工 {TimeLimitIndex}: 获取片段 {CurrentTimeLimit} 到期 -> {NextTimeLimit} \n"
                f"(其中发启编号对应: {StartFrame}, 需要消化总共: {TotalFrames}张图, 以单波队列吞吐量: {BATCH_SIZE}张处理)"
            )

            # ===== 步入GPU流核心：异步协程分发和归一排队计算 =====
            decode_q = queue.Queue(maxsize=2)  # 设计一个轻量的深2小推车缓冲系统。过大全炸显存，过小浪费等待空转
            decode_thread = threading.Thread(
                target=_decode_and_stack,
                args=(decoder, BATCH_SIZE, TotalFrames, decode_q),
                daemon=True,
            )
            decode_thread.start()

            segment_dets = []
            ProcessCount = 0
            batch_count = 0

            # 大缓存列表：承装无数GPU Tensor然后整团拼成车厢用 .cpu().numpy() 一次推入。解决细碎运输时PCIe带宽高延迟问题
            pending_tensors = []
            pending_ids = []
            pending_counts = []

            while True:
                item = decode_q.get()
                if item is None:
                    break
                batch, got = item

                # Pytorch张量类型分色洗点调整，将 [0-255] 大小的色度位映射缩放去至神经网络能看的懂读明白理解接受的 [0,1]。并调整其数值深度标准精度类型。
                if use_half:
                    input_batch = batch.to(dtype=torch.float16).div_(255.0)
                else:
                    input_batch = batch.to(dtype=torch.float32).div_(255.0)

                # 使用 with 模型抑制和关闭反向导数计算与记录图（内存将成倍大降、效能跃升！）
                with torch.inference_mode():
                    preds = internal_model(input_batch)
                    if isinstance(preds, (list, tuple)):
                        preds = preds[0]
                    # 超密集的检测中必定重叠出无数张复眼一样的重复检测预测矩阵目标，运用大浪淘沙过滤非极大数值。重合度限制在0.7剔除
                    dets_list = non_max_suppression(preds, conf_thres=0.5, iou_thres=0.7)

                # 从上述矩阵过滤中挑选并记入待处理区
                for bi, det in enumerate(dets_list):
                    if det is not None and len(det) > 0:
                        pending_tensors.append(det)
                        pending_ids.append(ProcessCount + bi)
                        pending_counts.append(len(det))

                ProcessCount += got
                current_frame += got
                batch_count += 1

                # 每运转满一个指定频率触发下行转移通道
                if pending_tensors and (
                    batch_count % FLUSH_INTERVAL == 0
                ):
                    big_gpu = torch.cat(pending_tensors, dim=0)
                    big_np = big_gpu.cpu().numpy()                 # PCIe 高速物理搬运发生在此
                    segment_dets.append((list(pending_ids), big_np, list(pending_counts)))
                    pending_tensors.clear()
                    pending_ids = []
                    pending_counts = []

            # 最后的倔将：零散的还没满整箱出厂要求的数据全部倾销出倒掉。免得残留无法结算缺失
            if pending_tensors:
                big_gpu = torch.cat(pending_tensors, dim=0)
                big_np = big_gpu.cpu().numpy()
                segment_dets.append((list(pending_ids), big_np, list(pending_counts)))
                pending_tensors.clear()
                pending_ids = []
                pending_counts = []

            decode_thread.join()

            seg_gpu_elapsed = time.time() - seg_start_time
            gpu_fps = ProcessCount / seg_gpu_elapsed if seg_gpu_elapsed > 0 else 0

            # ===== 脱出死海，将包好的大数据交予给CPU，主线程无休无止即可直接迈向进入下一个影片分析。真正重负压多线程无缝接替 ======
            ar = pool.apply_async(track_segment, (segment_dets, CsvFilePath))
            async_results.append((TimeLimitIndex, ar))
            del segment_dets  # 显式脱开内存标记指向，触发垃圾自动回收机制清理

            print(
                f"片段 {TimeLimitIndex} 号内圈所有繁重GPU任务计算与识别已处理干净利落完成: 总出片量 {ProcessCount} 张影帧, "
                f"所耗费耗时 {seg_gpu_elapsed:.1f} 秒等值，并稳定平均爆发其性能算数高达为: {gpu_fps:.0f} 帧率每秒 → 后续逻辑将移交分给其它各个散工做 CPU分析"
            )

        except Exception as e:
            print(f"当进行对该路片段 {CurrentTimeLimit} 层操作时出现罕见异常失误阻断运行: {e}")
            import traceback
            traceback.print_exc()

    # ===== 回归现实：等待挂在背景后的池子里边所散落出的工单最终处理报告完成 ======
    gpu_total_elapsed = time.time() - video_start_time
    print(f"\n=======================================================\n这所有的系列截段影片GPU相关测算步骤已经告绝终结束跑完了~ ，时间合计用量用了 {gpu_total_elapsed:.1f}秒，正耐心等着那批做细活苦力的CPU进度工们汇总...\n")

    pool.close()
    for idx, ar in async_results:
        try:
            ar.get() # 取值结果。挂死此处等待。
            print(f"片段区号 【{idx}】的追踪报告以及最后的数据盘上写保存均获无失通过完美竣工~")
        except Exception as e:
            print(f"报告长官! 在执行收尾那片号为 【{idx}】 时跟踪发生系统级深底故障无法完成被抛弃: {e}")
    pool.join()

    # 处理彻底结束撤掉驱动硬件绑定。删除占位的巨无霸硬盘SSD数据文件腾空地方。
    del decoder
    if temp_video_path and os.path.exists(temp_video_path):
        os.remove(temp_video_path)
        print(f"完成回收：临时用到的外借来高速SSD缓存文件已经被删除清扫恢复存储: {temp_video_path}")

    total_elapsed = time.time() - video_start_time
    print(f"最终判定当前这个 {os.path.basename(VideoPath).split('.')[0]} 影片工程大件彻底被吃抹完毕,从头到尾合计时间花销为 {total_elapsed:.1f} 秒")


if __name__ == "__main__":
    # 找寻挂载硬盘内所在的视频源目录
    VideoFolderPaths = [str(path) for path in CV_PATHS.default_video_dirs]
    # 调用训练出来的巴赫主模型用于识别引擎启动路径设置
    ModelPath = str(CV_PATHS.model_pt_path)
    # 设置所要落地出数据的追踪目录表输出地址
    CsvFolderPath = str(CV_PATHS.tracking_root)
    # 获取各个标定机器位置对应的物理参照与摄像机对标时间的校准参考数据标
    TimeLimitJsonPath = str(CV_PATHS.time_limit_json_path)

    # 读取散落各地大分区的多方面目标源合批记录
    AllVideos = []
    for folder in VideoFolderPaths:
        AllVideos.extend(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.endswith(('.mp4', '.avi')) and not f.startswith('.')
        )

    with open(TimeLimitJsonPath, "r", encoding="utf-8") as f:
        TimeLimitJson = json.load(f)

    os.makedirs(CsvFolderPath, exist_ok=True)

    # 制定分析作战策略和配装
    Task = []
    for i, VideoPath in enumerate(AllVideos):
        # 取本名字截断比如 a_1 -> JSON key内被记做统配识别标志 cam_a1（去下划线，拼凑标准形式检索）
        FileName = os.path.basename(VideoPath).split(".")[0]
        CamName = "cam_" + FileName.replace("_", "")
        try:
            TimeLimitList = TimeLimitJson[CamName]
        except KeyError:
            print(f"很遗憾，该系统未发现或缺少名为录影摄像头名 {CamName} (隶属文件载录体为: {FileName})的配置文件对应. 该文件已经被自动弃置和过载跳过。")
            continue

        TimeLimitList = [item["time_limit"] for item in TimeLimitList]
        Task.append((VideoPath, CsvFolderPath, ModelPath, i, TimeLimitList))

    # 一路执行所有收集好的下水管道测试任务。（本身任务内置并发系统能把一处全跑满不再支持更表层多并开发）
    total_start = time.time()
    for task in Task:
        video_process(task)
    total_elapsed = time.time() - total_start
    print(f"\n====================== 万众敬仰的全部分发运行视频分析作战顺利谢幕！====================\n这批总耗时一共经过消耗了 {total_elapsed:.1f}s ! \n谢谢使用。")
