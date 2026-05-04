"""
Count unique vehicles in border camera snapshot images.

Method: YOLO detection + ReID appearance matching + distance constraint.

For low-frequency snapshots (~3s intervals), position-only matching fails because:
  - Same vehicle moves far between frames → IoU ≈ 0 → overcounted
  - Different vehicle at same position → IoU high → undercounted

This script uses vehicle appearance (OpenVINO vehicle-reid-0001 embedding)
combined with a maximum distance constraint to match vehicles across frames:
  - Same vehicle, moved: ReID similar + distance OK → matched ✓
  - Different vehicle, same spot: ReID dissimilar → new vehicle ✓
  - Two same-model cars far apart: distance too large → not merged ✓

Usage:
    pip install ultralytics openvino Pillow

    # With project's custom YOLO model:
    python count_vehicles_in_images_openvino.py \
        --image-dir Macau_project-main/CV_part/crawler/border_cam_data/image5_test \
        --model Macau_project-main/CV_part/bach2.pt \
        --reid-model-path CV_part/models/openvino/public/vehicle-reid-0001/FP32/vehicle-reid-0001.xml

    # With standard YOLOv8 (auto-downloaded):
    python count_vehicles_in_images_openvino.py \
        --image-dir Macau_project-main/CV_part/crawler/border_cam_data/image5_test \
        --reid-model-path CV_part/models/openvino/public/vehicle-reid-0001/FP32/vehicle-reid-0001.xml

    # Preview counting zone:
    python count_vehicles_in_images_openvino.py --image-dir IMAGE_DIR --preview
"""

import argparse
import csv
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from ultralytics import YOLO


# COCO class IDs for vehicles (standard yolov8)
COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Custom model class IDs (project: 0=car, 1=bus, 2=truck, 3=motorcycle)
CUSTOM_VEHICLE_CLASSES = {0: "car", 1: "bus", 2: "truck", 3: "motorcycle"}

DEFAULT_OPENVINO_REID_MODEL = (
    Path(__file__).resolve().parent
    / "models"
    / "openvino"
    / "public"
    / "vehicle-reid-0001"
    / "FP32"
    / "vehicle-reid-0001.xml"
)


# ---------------------------------------------------------------------------
# ReID feature extractor
# ---------------------------------------------------------------------------
class VehicleReID:
    """Extract vehicle embeddings using OpenVINO vehicle-reid-0001."""

    EMBEDDING_DIM = 512

    def __init__(self, model_path, device="AUTO"):
        model_path = Path(model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                "OpenVINO ReID model .xml not found: "
                f"{model_path}\n"
                "Download/convert vehicle-reid-0001 first, then pass "
                "--reid-model-path /path/to/vehicle-reid-0001.xml"
            )

        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError(
                "OpenVINO is required for this script. Install it with: "
                "pip install openvino"
            ) from exc

        self.model_path = model_path
        self.device = device
        self.core = ov.Core()
        self.compiled_model = self.core.compile_model(str(model_path), device)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.input_name = self.input_layer.any_name
        self.input_h, self.input_w = self._resolve_input_hw()

    def extract(self, pil_image, bbox):
        """Crop bbox from image and return L2-normalized embedding (np array)."""
        tensor = self._crop_to_tensor(pil_image, bbox)
        if tensor is None:
            return self._zero_embedding()

        output = self.compiled_model({self.input_name: tensor})[self.output_layer]
        feat = np.asarray(output, dtype=np.float32).reshape(-1)
        return self._normalize_embedding(feat)

    def extract_batch(self, pil_image, bboxes):
        """Extract embeddings for multiple bboxes.

        vehicle-reid-0001 is distributed with a static batch-1 input, so this
        keeps inference sequential for correctness and simpler deployment.
        """
        if not bboxes:
            return []
        return [self.extract(pil_image, bbox) for bbox in bboxes]

    def _resolve_input_hw(self):
        shape = list(self.input_layer.shape)
        if len(shape) == 4:
            try:
                height = int(shape[2])
                width = int(shape[3])
                if height > 0 and width > 0:
                    return height, width
            except (TypeError, ValueError):
                pass
        return 208, 208

    def _crop_to_tensor(self, pil_image, bbox):
        img_w, img_h = pil_image.size
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(img_w, int(x1)))
        y1 = max(0, min(img_h, int(y1)))
        x2 = max(0, min(img_w, int(x2)))
        y2 = max(0, min(img_h, int(y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None

        crop = pil_image.crop((x1, y1, x2, y2))
        crop = crop.resize((self.input_w, self.input_h), Image.Resampling.BILINEAR)
        rgb = np.asarray(crop, dtype=np.float32)
        bgr = rgb[:, :, ::-1]
        chw = bgr.transpose(2, 0, 1)
        return np.ascontiguousarray(chw[None, :, :, :], dtype=np.float32)

    def _normalize_embedding(self, feat):
        if feat.size != self.EMBEDDING_DIM or not np.all(np.isfinite(feat)):
            return self._zero_embedding()
        norm = np.linalg.norm(feat)
        if norm <= 1e-8:
            return self._zero_embedding()
        return (feat / norm).astype(np.float32)

    def _zero_embedding(self):
        return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)


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
        description="Count unique vehicles in snapshot images (OpenVINO ReID + distance)")
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
    parser.add_argument(
        "--reid-model-path",
        default=str(DEFAULT_OPENVINO_REID_MODEL),
        help=(
            "OpenVINO vehicle-reid-0001 .xml path. Default: "
            f"{DEFAULT_OPENVINO_REID_MODEL}"
        ),
    )
    parser.add_argument(
        "--openvino-device",
        default="AUTO",
        help="OpenVINO device name, e.g. AUTO, CPU, GPU. Default: AUTO",
    )
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
                        help=(
                            "Min cosine similarity to match (default 0.50; "
                            "retune after switching ReID models)"
                        ))
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

    # 打印提示，说明接下来开始加载 OpenVINO ReID 外观特征提取器。
    print(f"Loading OpenVINO ReID: {args.reid_model_path} ({args.openvino_device})")
    # 创建 VehicleReID 对象，内部会加载 vehicle-reid-0001 OpenVINO IR 模型。
    reid = VehicleReID(args.reid_model_path, device=args.openvino_device)

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
    print(f"\nEstimated unique vehicles (OpenVINO ReID + distance matching):")
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
    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else image_dir / "vehicle_count_results_openvino.csv"
    )
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
