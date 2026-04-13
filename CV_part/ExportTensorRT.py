"""
将 YOLOv8 模型导出为 TensorRT FP16 引擎文件。
导出后的 .engine 文件用于替代原始 .pt 模型，推理速度提升 2-3 倍。

分两步：
  1. ultralytics 导出 ONNX（如果 .onnx 已存在则跳过）
  2. TensorRT Python API 直接从 ONNX 构建引擎（绕过 ultralytics 的 TRT 导出 bug）

用法:
    pip install tensorrt-cu12 onnx onnxslim
    python ProjectScripts/ExportTensorRT.py
"""

import os

MODEL_DIR = "ProjectScripts/ProjectTextDocument"
PT_PATH = os.path.join(MODEL_DIR, "bach2.pt")
ONNX_PATH = os.path.join(MODEL_DIR, "bach2.onnx")
ENGINE_PATH = os.path.join(MODEL_DIR, "bach2.engine")

# 图像尺寸与 batch 参数
IMG_H, IMG_W = 288, 352
ONNX_BATCH = 128      # 这个Batch 服务于计算图导出的过程，可以不用设置特别大，因为主要是用来构建逻辑计算图(神经网
                    #络结构)，后面通过dynamic=True告诉逻辑计算图网络结构可能传入的batch不定
MAX_BATCH = 512       # TensorRT 引擎支持的最大 batch，这个参数对计算图的影响很大，在构建引擎的时候，会以这个
#参数大小为最常用的批次，去做算子融合，选择最优的逻辑图方案

WORKSPACE_GIB = 4     # 构建Tensor RT引擎的时候，给定的最大临时显存工作区

# ── Step 1: 导出 ONNX（如果不存在）──
if not os.path.exists(ONNX_PATH):
    import torch, torch.onnx

    # PyTorch 2.9+ 默认走 dynamo ONNX 导出器，与 YOLO head 的 aten.item 不兼容
    _orig_export = torch.onnx.export
    def _patched_export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return _orig_export(*args, **kwargs)
    torch.onnx.export = _patched_export

    from ultralytics import YOLO
    model = YOLO(PT_PATH)
    model.export(
        format="onnx",
        device=0,
        imgsz=[IMG_H, IMG_W],
        half=False,          # ONNX 用 FP32 导出，FP16 由 TensorRT 引擎构建时处理
        dynamic=True,
        batch=ONNX_BATCH,
    )
    print(f"ONNX 导出完成: {ONNX_PATH}")
    # 释放 PyTorch / ultralytics 占用的 GPU 内存
    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU 显存已释放")
else:
    print(f"ONNX 已存在，跳过导出: {ONNX_PATH}")

# ── Step 2: TensorRT Python API 直接构建引擎 ──
import tensorrt as trt

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

print(f"解析 ONNX: {ONNX_PATH}")
with open(ONNX_PATH, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f"  ONNX 解析错误: {parser.get_error(i)}")
        raise RuntimeError("ONNX 解析失败")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GIB * (1 << 30))
config.set_flag(trt.BuilderFlag.FP16)

# 动态 batch 优化 profile：min=1, opt=MAX_BATCH, max=MAX_BATCH
profile = builder.create_optimization_profile()
input_name = network.get_input(0).name
profile.set_shape(input_name,
                  min=(1, 3, IMG_H, IMG_W),
                  opt=(MAX_BATCH, 3, IMG_H, IMG_W),
                  max=(MAX_BATCH, 3, IMG_H, IMG_W))
config.add_optimization_profile(profile)

print(f"构建 TensorRT FP16 引擎 (workspace={WORKSPACE_GIB} GiB, batch=1-{MAX_BATCH})...")
engine_bytes = builder.build_serialized_network(network, config)
if engine_bytes is None:
    raise RuntimeError("TensorRT 引擎构建失败")

with open(ENGINE_PATH, "wb") as f:
    f.write(engine_bytes)

print(f"TensorRT 导出完成: {ENGINE_PATH} ({os.path.getsize(ENGINE_PATH) / 1024 / 1024:.1f} MB)")
