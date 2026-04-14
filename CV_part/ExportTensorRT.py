"""
将 YOLOv8 模型导出为 TensorRT FP16 引擎文件。

分两步：
  1. ultralytics 导出 ONNX（如果 .onnx 已存在则跳过）
  2. TensorRT Python API 直接从 ONNX 构建引擎

用法:
    pip install tensorrt-cu12 onnx onnxslim
    python CV_part/ExportTensorRT.py
"""

from __future__ import annotations

import gc
from pathlib import Path

from cv_paths import CVPaths


PATHS = CVPaths.from_file(__file__)
PT_PATH = PATHS.model_pt_path
ONNX_PATH = PATHS.model_onnx_path
ENGINE_PATH = PATHS.model_engine_path

# 图像尺寸与 batch 参数
IMG_H, IMG_W = 288, 352
ONNX_BATCH = 128      # 这个 batch 主要服务于 ONNX 图导出，不必与最终引擎最优 batch 完全一致
MAX_BATCH = 1024      # TensorRT 引擎支持的最大 batch
WORKSPACE_GIB = 18    # 构建 TensorRT 引擎时允许使用的最大临时显存工作区


def export_onnx() -> None:
    if ONNX_PATH.exists():
        print(f"ONNX 已存在，跳过导出: {ONNX_PATH}")
        return

    import torch
    import torch.onnx

    # PyTorch 2.9+ 默认走 dynamo ONNX 导出器，与 YOLO head 的 aten.item 不兼容
    orig_export = torch.onnx.export

    def patched_export(*args, **kwargs):
        kwargs.setdefault("dynamo", False)
        return orig_export(*args, **kwargs)

    torch.onnx.export = patched_export

    from ultralytics import YOLO

    model = YOLO(str(PT_PATH))
    model.export(
        format="onnx",
        device=0,
        imgsz=[IMG_H, IMG_W],
        half=False,  # ONNX 用 FP32 导出，FP16 由 TensorRT 构建时处理
        dynamic=True,
        batch=ONNX_BATCH,
    )
    print(f"ONNX 导出完成: {ONNX_PATH}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU 显存已释放")


def build_engine() -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f"解析 ONNX: {ONNX_PATH}")
    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for idx in range(parser.num_errors):
                print(f"  ONNX 解析错误: {parser.get_error(idx)}")
            raise RuntimeError("ONNX 解析失败")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_GIB * (1 << 30))
    config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    profile.set_shape(
        input_name,
        min=(1, 3, IMG_H, IMG_W),
        opt=(MAX_BATCH, 3, IMG_H, IMG_W),
        max=(MAX_BATCH, 3, IMG_H, IMG_W),
    )
    config.add_optimization_profile(profile)

    print(f"构建 TensorRT FP16 引擎 (workspace={WORKSPACE_GIB} GiB, batch=1-{MAX_BATCH})...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT 引擎构建失败")

    with open(ENGINE_PATH, "wb") as f:
        f.write(engine_bytes)

    print(f"TensorRT 导出完成: {ENGINE_PATH} ({ENGINE_PATH.stat().st_size / 1024 / 1024:.1f} MB)")


def main() -> None:
    PATHS.artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"模型权重路径: {PT_PATH}")
    print(f"ONNX 输出路径: {ONNX_PATH}")
    print(f"Engine 输出路径: {ENGINE_PATH}")

    export_onnx()
    build_engine()


if __name__ == "__main__":
    main()
