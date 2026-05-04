"""
Count unique vehicles in border camera snapshot images.

Method: YOLO detection + ReID appearance matching + distance constraint.

For low-frequency snapshots (~3s intervals), position-only matching fails because:
  - Same vehicle moves far between frames → IoU ≈ 0 → overcounted
  - Different vehicle at same position → IoU high → undercounted

This script uses vehicle appearance (ReID embedding from pretrained ResNet) combined
with a maximum distance constraint to match vehicles across frames:
  - Same vehicle, moved: ReID similar + distance OK → matched ✓
  - Different vehicle, same spot: ReID dissimilar → new vehicle ✓
  - Two same-model cars far apart: distance too large → not merged ✓

Usage:
    pip install ultralytics torch torchvision Pillow

    # With project's custom YOLO model:
    python count_vehicles_in_images.py \
        --image-dir Macau_project-main/CV_part/crawler/border_cam_data/image5_test \
        --model Macau_project-main/CV_part/bach2.pt

    # With standard YOLOv8 (auto-downloaded):
    python count_vehicles_in_images.py \
        --image-dir Macau_project-main/CV_part/crawler/border_cam_data/image5_test

    # Preview counting zone:
    python count_vehicles_in_images.py --image-dir IMAGE_DIR --preview
"""

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as T
from torchvision import models
from PIL import Image
from ultralytics import YOLO


# COCO class IDs for vehicles (standard yolov8)
COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Custom model class IDs (project: 0=car, 1=bus, 2=truck, 3=motorcycle)
CUSTOM_VEHICLE_CLASSES = {0: "car", 1: "bus", 2: "truck", 3: "motorcycle"}


# ---------------------------------------------------------------------------
# ReID feature extractor
# ---------------------------------------------------------------------------
class VehicleReID:
    # 这个类专门负责“车辆重识别”特征提取：
    # 输入是一张原图和 YOLO 检测出来的车辆框，输出该车辆外观的 512 维特征向量。
    """Extract appearance embeddings using a pretrained ResNet18."""

    def __init__(self, device=None):
        # 初始化函数；device 用来指定模型运行在 CPU 还是 GPU 上。
        if device is None:
            # 如果调用者没有手动指定 device，就自动判断当前机器是否支持 CUDA GPU。
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # 如果有可用 GPU，就使用 "cuda"；否则退回到 "cpu"。
        self.device = device
        # 把最终选择的运行设备保存到对象属性里，后面模型和张量都会放到这个设备上。

        # Use ResNet18 as feature backbone (512-dim embedding)
        # 加载 torchvision 官方提供的 ResNet18 预训练权重。
        weights = models.ResNet18_Weights.DEFAULT
        # 使用上面的预训练权重构建 ResNet18 模型。
        backbone = models.resnet18(weights=weights)
        # 这里的 backbone 原本是一个完整的图像分类模型。
        # Remove classification head → output is (batch, 512, 1, 1)
        # list(backbone.children()) 会取出 ResNet18 的所有子模块。
        # [:-1] 表示去掉最后一层分类器，只保留前面的特征提取部分。
        # torch.nn.Sequential(...) 把这些保留下来的层重新组装成一个新模型。
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        # eval() 把模型切换到推理模式，关闭训练时才需要的行为。
        # to(self.device) 把模型移动到前面选择的 CPU 或 GPU 上。
        self.model.eval().to(self.device)

        # 定义输入车辆裁剪图的预处理流程；必须和 ResNet18 预训练时的输入格式保持一致。
        self.transform = T.Compose([
            # 把车辆裁剪图统一缩放到 128x128，方便批量送入模型。
            T.Resize((128, 128)),
            # 把 PIL 图片转换成 PyTorch Tensor，像素值也会从 0~255 转成 0~1。
            T.ToTensor(),
            # 使用 ImageNet 的均值和标准差做归一化，因为 ResNet18 是在 ImageNet 上预训练的。
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        # 分别对应 RGB 三个通道的标准差。
                        std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    # no_grad 表示这个函数只做推理，不计算梯度，可以减少显存/内存占用并加快速度。
    def extract(self, pil_image, bbox):
        # 从单个 bbox 中提取一辆车的外观特征。
        """Crop bbox from image and return L2-normalized embedding (np array)."""
        # bbox 是 YOLO 给出的边界框，格式通常是 (x1, y1, x2, y2)。
        # int(v) 把坐标转成整数；max(0, ...) 防止坐标小于图片边界。
        x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
        # 根据边界框从原图中裁剪出车辆区域。
        crop = pil_image.crop((x1, y1, x2, y2))
        # 如果裁剪出来的区域太小，说明检测框异常或无效。
        if crop.size[0] < 2 or crop.size[1] < 2:
            # 返回一个全 0 的 512 维向量作为占位，避免后续代码报错。
            return np.zeros(512, dtype=np.float32)
        # 对车辆裁剪图做预处理，并通过 unsqueeze(0) 增加 batch 维度。
        # 原本形状是 (C, H, W)，加 batch 后变成 (1, C, H, W)。
        # to(self.device) 把输入张量移动到和模型相同的设备上。
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        # 把车辆图送入 ResNet18 特征提取器，得到 512 维外观特征。
        feat = self.model(tensor).squeeze()
        # 对特征做 L2 归一化，使向量长度变成 1。
        # 这样后面用 np.dot(a, b) 就相当于计算 cosine similarity。
        # 1e-8 是为了避免极端情况下除以 0。
        feat = feat / (feat.norm() + 1e-8)
        # 把特征从 GPU/CPU Tensor 转回 NumPy 数组，方便后续用 numpy 做相似度计算。
        return feat.cpu().numpy()

    @torch.no_grad()
    # 这个函数同样只做推理，不需要梯度。
    def extract_batch(self, pil_image, bboxes):
        # 一次性为多个 bbox 提取外观特征，比逐个调用 extract 更高效。
        """Extract embeddings for multiple bboxes in one forward pass."""
        # 如果当前图片没有检测框，就直接返回空列表。
        if not bboxes:
            return []
        # tensors 用来存放所有有效车辆裁剪图预处理后的 Tensor。
        tensors = []
        # valid_indices 记录哪些 bbox 是有效的，方便之后把特征放回原来的位置。
        valid_indices = []
        # results 先准备一个结果列表，长度和 bboxes 一样。
        # 对无效 bbox，默认保留 512 维全 0 向量作为占位。
        results = [np.zeros(512, dtype=np.float32)] * len(bboxes)

        # 逐个处理当前图片里的所有检测框。
        for i, bbox in enumerate(bboxes):
            # 把 bbox 坐标转成非负整数，避免 PIL crop 时出现负坐标。
            x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
            # 从原图中裁剪出第 i 个检测框对应的车辆区域。
            crop = pil_image.crop((x1, y1, x2, y2))
            # 如果裁剪图宽或高太小，就认为这个检测框无效。
            if crop.size[0] < 2 or crop.size[1] < 2:
                # 跳过这个无效框；它在 results 里会继续保持全 0 向量。
                continue
            # 对有效裁剪图做 Resize、ToTensor、Normalize 等预处理。
            tensors.append(self.transform(crop))
            # 记录这个有效裁剪图原本对应 bboxes 里的哪个下标。
            valid_indices.append(i)

        # 如果所有 bbox 都无效，没有任何可送入模型的图片。
        if not tensors:
            # 直接返回默认的全 0 特征列表。
            return results

        # 把多个单张图片 Tensor 堆叠成一个 batch。
        # 形状从多个 (C, H, W) 变成 (N, C, H, W)。
        # 然后把 batch 移动到 CPU 或 GPU 上。
        batch = torch.stack(tensors).to(self.device)
        # 一次前向传播提取所有车辆的 ResNet 特征。
        # 原始输出形状是 (N, 512, 1, 1)，两个 squeeze(-1) 去掉最后两个大小为 1 的维度。
        # 最终 feats 的形状是 (N, 512)。
        feats = self.model(batch).squeeze(-1).squeeze(-1)
        # 对 batch 中每一辆车的 512 维特征分别做 L2 归一化。
        # norm(dim=1, keepdim=True) 会计算每一行向量的长度，并保持形状方便广播相除。
        feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-8)
        # 把 PyTorch Tensor 转成 NumPy 数组，方便后面和其他 detection 一起保存、比较。
        feats_np = feats.cpu().numpy()

        # 把模型输出的特征写回 results 中对应的原始 bbox 位置。
        for idx, feat in zip(valid_indices, feats_np):
            # idx 是原始 bboxes 的下标，feat 是该 bbox 对应的 512 维特征。
            results[idx] = feat
        # 返回和 bboxes 一一对应的特征列表。
        return results


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def detect_vehicles(model, image_path, conf_threshold, is_custom_model):
    """Run YOLO on a single image, return detections list."""
    results = model(image_path, conf=conf_threshold, verbose=False)
    cls_map = CUSTOM_VEHICLE_CLASSES if is_custom_model else COCO_VEHICLE_CLASSES

    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in cls_map:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                detections.append({
                    "class": cls_map[cls_id],
                    "conf": conf,
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                })
    return detections


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
def cosine_sim(a, b):
    return float(np.dot(a, b))


def center_dist(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def in_zone(det, img_h, y_min_frac, y_max_frac):
    cy = det["center"][1]
    return y_min_frac * img_h <= cy <= y_max_frac * img_h


def count_unique_vehicles(all_frame_data, img_h, y_min_frac, y_max_frac,
                          reid_thresh, max_dist, lookback=3):
    """
    Count unique vehicles using sliding-window matching.

    Each current-frame detection is compared against raw detections from the
    previous `lookback` frames (no embedding updates — entries naturally expire).
    A detection is "new" only if it doesn't match ANY detection in the window.

    This avoids two failure modes:
      - Frame-to-frame only (lookback=1): missed detection → double-count
      - Persistent gallery with updates: entry absorbs different vehicles → undercount
    """
    unique_count = 0
    class_counts = defaultdict(int)
    # Sliding window: list of recent frames' zone detections (oldest first)
    window = []

    for frame_data in all_frame_data:
        curr_zone = [d for d in frame_data
                     if in_zone(d, img_h, y_min_frac, y_max_frac)]

        for det in curr_zone:
            matched = False
            # Search window from most recent to oldest
            for prev_zone in reversed(window):
                for prev_det in prev_zone:
                    if det["class"] != prev_det["class"]:
                        continue
                    dist = center_dist(det["center"], prev_det["center"])
                    if dist > max_dist:
                        continue
                    sim = cosine_sim(det["embedding"], prev_det["embedding"])
                    if sim > reid_thresh:
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                unique_count += 1
                class_counts[det["class"]] += 1

        window.append(curr_zone)
        if len(window) > lookback:
            window.pop(0)

    return unique_count, dict(class_counts)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------
def preview_zone(image_path, y_min_frac, y_max_frac):
    from PIL import ImageDraw
    img = Image.open(image_path)
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    y_min, y_max = int(y_min_frac * h), int(y_max_frac * h)
    draw.rectangle([0, y_min, w, y_max],
                   fill=(0, 255, 0, 50), outline=(0, 255, 0, 200))
    draw.line([(0, y_min), (w, y_min)], fill=(0, 255, 0), width=2)
    draw.line([(0, y_max), (w, y_max)], fill=(0, 255, 0), width=2)
    preview_path = image_path.parent / "zone_preview.jpg"
    img.save(preview_path)
    print(f"Zone preview saved: {preview_path}")
    print(f"Zone: y={y_min_frac:.0%}~{y_max_frac:.0%} (px {y_min}~{y_max} / {h})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 创建命令行参数解析器，用来读取用户在终端运行脚本时传入的参数。
    parser = argparse.ArgumentParser(
        # description 是脚本说明，用户运行 --help 时会看到这句话。
        description="Count unique vehicles in snapshot images (ReID + distance)")
    # 添加必填参数 --image-dir，用来指定待统计图片所在的文件夹路径。
    parser.add_argument("--image-dir", required=True)
    # 添加可选参数 --model，用来指定 YOLO 模型文件路径；如果不传，就使用默认模型。
    parser.add_argument("--model", default=None,
                        # 这里说明默认模型是 yolov8n.pt。
                        help="YOLO model path. Default: yolov8n.pt")
    # 添加可选参数 --conf，表示 YOLO 检测框的置信度阈值。
    parser.add_argument("--conf", type=float, default=0.25,
                        # 如果检测置信度低于这个阈值，就会被 YOLO 过滤掉。
                        help="YOLO confidence threshold")
    # 添加可选参数 --output-csv，用来指定结果 CSV 文件保存到哪里。
    parser.add_argument("--output-csv", default=None)
    # 添加开关参数 --car-only；用户写了这个参数时，只统计 car 类别。
    parser.add_argument("--car-only", action="store_true")
    # 添加可选参数 --zone-y-min，表示统计区域顶部在图片高度中的比例位置。
    parser.add_argument("--zone-y-min", type=float, default=0.10,
                        # 默认从图片高度的 10% 位置开始统计。
                        help="Zone top (fraction of height, default 0.10)")
    # 添加可选参数 --zone-y-max，表示统计区域底部在图片高度中的比例位置。
    parser.add_argument("--zone-y-max", type=float, default=0.80,
                        # 默认到图片高度的 80% 位置结束统计。
                        help="Zone bottom (fraction of height, default 0.80)")
    # 添加可选参数 --reid-thresh，表示车辆外观相似度匹配阈值。
    parser.add_argument("--reid-thresh", type=float, default=0.50,
                        # 两辆车的 cosine similarity 大于该阈值时，才可能被认为是同一辆。
                        help="Min cosine similarity to match (default 0.50)")
    # 添加可选参数 --max-dist，限制两次检测中心点之间允许匹配的最大像素距离。
    parser.add_argument("--max-dist", type=float, default=300,
                        # 如果两辆车中心点距离超过该值，即使外观相似也不合并。
                        help="Max center distance in pixels to allow match (default 300)")
    # 添加可选参数 --lookback，表示当前帧会往前看多少帧做匹配。
    parser.add_argument("--lookback", type=int, default=3,
                        # 默认和最近 3 帧里的检测结果比较，判断是否已经出现过。
                        help="Match against detections in last N frames (default 3)")
    # 添加开关参数 --preview；用户写了这个参数时，只生成统计区域预览图，不做车辆统计。
    parser.add_argument("--preview", action="store_true",
                        # 这句说明 preview 模式会在第一张图片上画出统计区域然后退出。
                        help="Draw zone on first image and exit")
    # 真正解析终端传入的参数，并把结果保存到 args 对象里。
    args = parser.parse_args()

    # 把用户传入的图片文件夹路径转换成 Path 对象，方便后面做路径操作。
    image_dir = Path(args.image_dir)
    # 检查 image_dir 是否确实是一个存在的文件夹。
    if not image_dir.is_dir():
        # 如果路径不存在，或者不是文件夹，就抛出错误并停止程序。
        raise FileNotFoundError(f"Not found: {image_dir}")

    # 定义允许读取的图片后缀名集合。
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    # 遍历图片文件夹，把后缀属于 exts 的文件筛选出来，并按文件名排序。
    image_files = sorted(f for f in image_dir.iterdir()
                         # suffix.lower() 把后缀转成小写，避免 .JPG 这种大写后缀被漏掉。
                         if f.suffix.lower() in exts)
    # 如果筛选后没有任何图片文件。
    if not image_files:
        # 在终端提示没有找到图片。
        print("No images found.")
        # 提前结束 main()，不再继续加载模型或统计。
        return
    # 输出找到的图片数量，方便用户确认读取目录是否正确。
    print(f"Found {len(image_files)} images")

    # 如果用户启用了 --preview 模式。
    if args.preview:
        # 用第一张图片画出统计区域，保存成 zone_preview.jpg。
        preview_zone(image_files[0], args.zone_y_min, args.zone_y_max)
        # preview 模式只负责看区域，不继续做 YOLO 检测和车辆统计。
        return

    # Load models
    # 判断是否使用了用户自己传入的 YOLO 模型路径。
    is_custom = args.model is not None
    # 如果用户传了 --model，就用用户的模型；否则默认使用 yolov8n.pt。
    model_path = args.model or "yolov8n.pt"
    # 打印当前加载的 YOLO 模型路径，以及类别映射会按 custom 还是 COCO 处理。
    print(f"Loading YOLO: {model_path} ({'custom' if is_custom else 'COCO'})")
    # 创建 YOLO 模型对象；这一步会加载模型权重。
    yolo = YOLO(model_path)

    # 打印提示，说明接下来开始加载 ReID 外观特征提取器。
    print("Loading ReID feature extractor (ResNet18)...")
    # 创建 VehicleReID 对象，内部会加载预训练 ResNet18。
    reid = VehicleReID()

    # Image dimensions
    # 打开第一张图片，用来读取图片宽度和高度。
    with Image.open(image_files[0]) as img0:
        # img0.size 返回 (宽度, 高度)。
        img_w, img_h = img0.size
    # 打印图片尺寸，方便用户知道后面像素距离和统计区域对应的尺度。
    print(f"Image: {img_w}x{img_h}")
    # 打印统计区域的相对比例和换算后的像素范围。
    print(f"Zone: y={args.zone_y_min:.0%}~{args.zone_y_max:.0%} "
          # 这里把比例乘以图片高度，得到实际的 y 坐标上下边界。
          f"(px {int(args.zone_y_min * img_h)}~{int(args.zone_y_max * img_h)})")
    # 打印 ReID 相似度阈值和最大中心点距离，方便记录本次实验参数。
    print(f"ReID thresh: {args.reid_thresh}, Max dist: {args.max_dist}px")

    # Process all images
    # all_frame_data 用来保存所有帧的完整检测结果，后面唯一车辆计数会用到。
    all_frame_data = []
    # per_frame_results 用来保存每一张图片的统计结果，最后写入 CSV。
    per_frame_results = []
    # total_detections 统计所有图片中 YOLO 检测到的车辆框总数。
    total_detections = 0
    # class_totals 按类别累计所有图片中的检测框数量。
    class_totals = defaultdict(int)

    # skipped 用来记录无法读取或检测失败的图片文件名。
    skipped = []
    # 按顺序遍历所有图片；i 是图片下标，img_path 是图片路径。
    for i, img_path in enumerate(image_files):
        # Verify image is readable
        # 尝试打开当前图片，防止遇到损坏文件时程序直接崩溃。
        try:
            # Image.open 读取图片；convert("RGB") 统一转换成 RGB 三通道格式。
            pil_img = Image.open(img_path).convert("RGB")
        # 如果图片无法读取，PIL 会抛出异常。
        except Exception:
            # 打印跳过提示，说明当前图片不可读。
            print(f"  [SKIP] Unreadable image: {img_path.name}")
            # 把不可读图片的文件名记录下来，最后统一汇报。
            skipped.append(img_path.name)
            # 跳过当前图片，继续处理下一张。
            continue

        # Detect
        # 尝试对当前图片执行 YOLO 车辆检测。
        try:
            # detect_vehicles 会返回当前图片中属于车辆类别的检测结果列表。
            detections = detect_vehicles(yolo, str(img_path), args.conf, is_custom)
        # 如果 YOLO 推理过程中出错，就进入异常处理。
        except Exception:
            # 打印跳过提示，说明当前图片检测失败。
            print(f"  [SKIP] Detection failed: {img_path.name}")
            # 把检测失败的文件名记录下来。
            skipped.append(img_path.name)
            # 跳过当前图片，继续下一张。
            continue

        # 如果用户启用了 --car-only 参数。
        if args.car_only:
            # 只保留 class 等于 "car" 的检测框，过滤掉 bus、truck、motorcycle。
            detections = [d for d in detections if d["class"] == "car"]

        # Extract ReID embeddings for all detections in this frame
        # 从当前图片的每个 detection 中取出 bbox，准备送给 ReID 模型裁剪车辆图。
        bboxes = [d["bbox"] for d in detections]
        # 批量提取当前图片所有检测车辆的 512 维外观特征。
        embeddings = reid.extract_batch(pil_img, bboxes)
        # 把 detection 和对应的 embedding 一一配对。
        for det, emb in zip(detections, embeddings):
            # 将外观特征写入 detection 字典，后面跨帧匹配时会用到。
            det["embedding"] = emb

        # 把当前帧的所有检测结果加入 all_frame_data，供最后唯一车辆计数使用。
        all_frame_data.append(detections)

        # Per-frame stats
        # frame_cls 用来统计当前这一张图片里每个类别各有多少个检测框。
        frame_cls = defaultdict(int)
        # 遍历当前图片中的所有检测结果。
        for d in detections:
            # 当前图片中该类别数量加 1。
            frame_cls[d["class"]] += 1
            # 全部图片范围内该类别总数也加 1。
            class_totals[d["class"]] += 1
        # 把当前图片的检测框数量加入全局检测总数。
        total_detections += len(detections)

        # 统计当前图片中有多少检测框的中心点落在指定统计区域内。
        zone_count = sum(1 for d in detections
                         # in_zone 会根据车辆中心点 y 坐标判断是否处在 zone-y-min 到 zone-y-max 之间。
                         if in_zone(d, img_h, args.zone_y_min, args.zone_y_max))
        # 把当前图片的逐帧统计结果保存成一个字典，后面会写入 CSV。
        per_frame_results.append({
            # 当前图片文件名。
            "filename": img_path.name,
            # 当前图片中检测到的车辆总数。
            "total": len(detections),
            # 当前图片中落在统计区域内的车辆数量。
            "in_zone": zone_count,
            # 下面这段字典推导式会生成 car、bus、truck、motorcycle 四个类别的数量字段。
            **{c: frame_cls.get(c, 0)
               # 如果某个类别在当前图片中没有出现，就通过 get(c, 0) 记为 0。
               for c in ["car", "bus", "truck", "motorcycle"]},
        })

        # 每处理 50 张图片打印一次进度；如果到了最后一张，也打印进度。
        if (i + 1) % 50 == 0 or i == len(image_files) - 1:
            # i 从 0 开始，所以显示进度时用 i + 1 表示已处理图片数。
            print(f"  Processed {i + 1}/{len(image_files)}")

    # Count unique vehicles
    # 在所有帧检测完成后，进行跨帧匹配，估算唯一车辆数量。
    unique_total, unique_by_class = count_unique_vehicles(
        # 传入所有图片的检测结果，每个 detection 里面已经包含 bbox、class、center、embedding。
        all_frame_data, img_h,
        # 传入统计区域的 y 方向比例范围。
        args.zone_y_min, args.zone_y_max,
        # 传入 ReID 相似度阈值和最大中心点距离。
        args.reid_thresh, args.max_dist,
        # 传入向前回看多少帧做匹配。
        args.lookback,
    )

    # Results
    # 打印一个空行和分隔线，让结果区域更清楚。
    print("\n" + "=" * 60)
    # 打印结果标题。
    print("RESULTS")
    # 再打印一条分隔线。
    print("=" * 60)
    # 打印图片总数；注意这里是文件夹中找到的图片数量，不是成功处理数量。
    print(f"Images processed: {len(image_files)}")
    # 打印所有帧所有区域内的原始检测框总数；同一辆车出现在多帧会被重复算。
    print(f"\nRaw detections (all frames, full image): {total_detections}")
    # 依次检查四种车辆类别的累计检测数量。
    for cls in ["car", "bus", "truck", "motorcycle"]:
        # 只有当该类别出现次数大于 0 时才打印，避免输出一堆 0。
        if class_totals[cls] > 0:
            # 打印该类别在所有图片中的原始检测框数量。
            print(f"  {cls}: {class_totals[cls]}")

    # 打印唯一车辆估算结果标题。
    print(f"\nEstimated unique vehicles (ReID + distance matching):")
    # 打印估算出的唯一车辆总数。
    print(f"  Total: {unique_total}")
    # 依次检查四种车辆类别的唯一车辆估算数量。
    for cls in ["car", "bus", "truck", "motorcycle"]:
        # 只有该类别的唯一车辆数大于 0 时才打印。
        if unique_by_class.get(cls, 0) > 0:
            # 打印该类别估算出的唯一车辆数量。
            print(f"  {cls}: {unique_by_class[cls]}")

    # Save CSV
    # 如果用户传了 --output-csv，就使用用户指定路径；否则默认保存在图片目录下。
    output_path = Path(args.output_csv) if args.output_csv else image_dir / "vehicle_count_results.csv"
    # 定义 CSV 文件的列名和顺序。
    fieldnames = ["filename", "total", "in_zone", "car", "bus", "truck", "motorcycle"]
    # 以写入模式打开 CSV 文件；newline="" 可以避免某些平台出现多余空行。
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        # 创建 DictWriter，用字典的 key 对应 CSV 的列名。
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # 先写入 CSV 表头。
        writer.writeheader()
        # 再把每一帧的统计结果逐行写入 CSV。
        writer.writerows(per_frame_results)
    # 打印 CSV 文件保存路径。
    print(f"\nPer-frame CSV: {output_path}")

    # 如果 skipped 列表不为空，说明有图片被跳过。
    if skipped:
        # 打印警告信息和被跳过的图片数量。
        print(f"\n[WARNING] Skipped {len(skipped)} unreadable images:")
        # 遍历所有被跳过的图片文件名。
        for name in skipped:
            # 逐行打印被跳过的文件名，方便用户检查问题图片。
            print(f"  - {name}")


if __name__ == "__main__":
    main()
