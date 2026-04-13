"""
SUMO 仿真后端模块 (sumo_backend.py)
====================================

本模块是项目中连接 贝叶斯OD估计 与 SUMO微观交通仿真 的桥梁。

整体职责:
    1. 发现并验证本地 SUMO 安装环境 (SUMO_HOME, netconvert, od2trips 等工具)
    2. 将项目中的 23 个交通分区 (zone) 映射到 SUMO 路网的 TAZ (Traffic Assignment Zone)
    3. 将摄像头 gate (检测点) 映射到 SUMO 路网中最匹配的 edge (路段)
    4. 将贝叶斯后验 OD 矩阵转换为 SUMO 的 OD 需求文件, 运行 DUA (Dynamic User Assignment)
    5. 从 SUMO 仿真路径结果中反向构建分配矩阵 H (gate × OD pair), 用于与解析式 H 对比
    6. 提取仿真中每个 gate 的转向流量 (turning count), 用于验证

核心概念:
    - TAZ (Traffic Assignment Zone): SUMO 中将多条 edge 聚合为一个"出行起讫区域"的概念,
      对应项目中的 23 个交通分区
    - Gate: 摄像头检测线, 每个 gate 有 from_edge (车辆驶入方向) 和 to_edge (车辆驶出方向)
    - DUA (Dynamic User Assignment): SUMO 的动态用户均衡分配算法, 通过 duaIterate.py 迭代求解
    - H 矩阵: 分配矩阵, 形状为 (n_gates × n_od_pairs), 表示每个OD对对每个gate的贡献比例

注意: SUMO 仿真是后处理验证步骤, 不参与贝叶斯估计链。
      解析式 H (K-shortest paths + Logit) 驱动贝叶斯估计;
      SUMO 在之后运行, 用于模拟后验OD并产出诊断转向统计。
"""

from __future__ import annotations

# === 标准库导入 ===
import csv            # 读写 CSV 文件 (OD矩阵、edge流量)
import hashlib        # 生成缓存文件的哈希签名
import json           # 读写 JSON 配置和元数据
import logging        # 日志记录
import math           # 数学计算 (角度、四舍五入等)
import os             # 环境变量访问 (SUMO_HOME)
import re             # 正则表达式 (解析 gap 日志)
import shutil         # 文件操作 (复制路线文件、查找可执行文件)
import subprocess     # 调用外部 SUMO 命令行工具
import sys            # Python 解释器路径、sys.path 操作
import time           # 计时 SUMO 仿真各阶段耗时
import xml.etree.ElementTree as ET  # 读写 SUMO XML 文件 (TAZ, OD, 路线等)
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# === 第三方库导入 ===
import networkx as nx                                      # 图论分析 (强连通分量检测)
import numpy as np                                         # 数值计算 (OD矩阵、H矩阵)
import scipy.sparse as sp                                  # 稀疏矩阵 (H矩阵存储)
from shapely.geometry import LineString, MultiPolygon, Point, Polygon  # 几何运算 (zone多边形、edge线段)

# === 项目内部导入 ===
from .network import load_zones_geojson    # 加载 23 个交通分区的 GeoJSON 边界
from .paths import ProjectPaths, default_paths  # 项目文件路径管理
from .types import (
    GateInfo,                  # 摄像头 gate 信息 (camera_id, 方向, 关联道路等)
    SumoAssignmentArtifacts,   # SUMO 仿真产出的文件路径和诊断信息集合
    SumoAssignmentConfig,      # SUMO 仿真配置 (仿真时间、迭代次数、随机种子等)
    SumoGateMapping,           # gate → SUMO edge 的映射结果
    SumoSimulationResult,      # 仿真最终结果 (H矩阵 + 映射 + artifacts)
    SumoNativeMappings,        # zone映射 + gate映射 的综合容器
    SumoZoneMapping,           # zone → SUMO TAZ 的映射结果
)

logger = logging.getLogger(__name__)

# ============================================================================
# 常量定义
# ============================================================================

# 项目车型名称 → SUMO 内部 vClass 名称 的映射
# SUMO 用 vClass 来决定车辆允许通行的 edge (例如 bus 只能走允许 bus 的道路)
VEHICLE_TO_VCLASS = {
    "car": "passenger",     # 小汽车 → SUMO 的 "passenger" 类
    "bus": "bus",           # 公交车
    "truck": "truck",       # 货车
    "motorcycle": "motorcycle",  # 摩托车
}

# 方向名称 → 方位角 (bearing) 的映射, 以正北为 0°, 顺时针递增
# 用于将摄像头配置中的方向描述 (如 "north", "southeast") 转换为数值角度,
# 再与 SUMO edge 的几何方位角进行匹配
DIRECTION_BEARINGS = {
    "north": 0,
    "northeast": 45,
    "east": 90,
    "southeast": 135,
    "south": 180,
    "southwest": 225,
    "west": 270,
    "northwest": 315,
}

# 方向名称拼写纠错映射 (处理摄像头配置 JSON 中的拼写错误)
DIRECTION_ALIASES = {
    "nortwest": "northwest",
    "northweast": "northeast",
    "southest": "southeast",
    "souhtwest": "southwest",
    "nort": "north",
    "weast": "east",
}

# gate 匹配 SUMO edge 时的搜索参数
DEFAULT_EDGE_SEARCH_RADIUS = 80.0   # 搜索半径 (米): 在摄像头位置周围 80m 内寻找候选 edge
DEFAULT_EDGE_SEARCH_LIMIT = 20      # 当半径内候选不足时, 取距离最近的前 20 条 edge


# ============================================================================
# 内部数据结构
# ============================================================================


@dataclass
class _SumoEdgeRecord:
    """SUMO 路网中一条 edge (路段) 的精简记录。

    从 sumolib 的 net 对象中提取核心属性, 便于后续几何计算和匹配。
    SUMO 中的 edge 是有向的, 从 from_node 指向 to_node, shape 记录了路段的几何折线。

    Attributes:
        edge_id: SUMO edge 的唯一标识 (如 "123456789#0")
        from_node_id: 起始节点 ID
        to_node_id: 终止节点 ID
        shape: edge 的几何形状, 由 (x, y) 坐标点序列组成 (SUMO 投影坐标系, 单位: 米)
        name: 道路名称 (用于与摄像头配置中的路名进行匹配)
        length: 路段长度 (米)
        speed: 路段限速 (m/s)
        is_internal: 是否为交叉口内部 edge (以 ":" 开头的 edge, 通常被过滤掉)
    """
    edge_id: str
    from_node_id: str
    to_node_id: str
    shape: list[tuple[float, float]]
    name: str = ""
    length: float = 0.0
    speed: float = 0.0
    is_internal: bool = False

    @property
    def line(self) -> LineString:
        """将 shape 转为 Shapely LineString, 用于几何空间查询 (距离计算、相交判断等)。"""
        return LineString(self.shape)


# ============================================================================
# 方向与角度辅助函数
# ============================================================================


def direction_to_bearing(direction: str) -> float | None:
    """将文本方向描述转换为方位角 (度)。

    支持三种输入格式:
      - 单一方向: "north" → 0.0, "southeast" → 135.0
      - 拼写纠错: "nortwest" → "northwest" → 315.0
      - 复合方向 (斜杠分隔): "north/east" → 对两个方位角取圆周均值 → 45.0

    Args:
        direction: 方向文本 (来自摄像头配置 JSON)

    Returns:
        方位角 (0~360 度), 若无法识别则返回 None
    """
    direction = direction.strip().lower()
    # 先尝试拼写纠错
    direction = DIRECTION_ALIASES.get(direction, direction)
    if direction in DIRECTION_BEARINGS:
        return float(DIRECTION_BEARINGS[direction])
    # 处理 "north/east" 这类复合方向
    if "/" in direction:
        parts = []
        for part in direction.split("/"):
            part = DIRECTION_ALIASES.get(part.strip(), part.strip())
            if part in DIRECTION_BEARINGS:
                parts.append(float(DIRECTION_BEARINGS[part]))
        if parts:
            return _circular_mean(parts)
    return None


def _circular_mean(angles_deg: list[float]) -> float:
    """计算角度的圆周均值 (circular mean)。

    普通算术平均不适用于角度 (例如 350° 和 10° 的平均应为 0°, 而非 180°)。
    此函数将角度转换为单位圆上的向量, 对向量求和后取 atan2, 得到正确的平均方向。

    Args:
        angles_deg: 角度列表 (度)

    Returns:
        圆周均值角度 (0~360 度)
    """
    rads = [math.radians(angle) for angle in angles_deg]
    sin_sum = sum(math.sin(angle) for angle in rads)
    cos_sum = sum(math.cos(angle) for angle in rads)
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360


def _angular_diff(a: float, b: float) -> float:
    """计算两个方位角之间的最小角度差 (0~180 度)。

    例如: _angular_diff(350, 10) → 20, _angular_diff(90, 270) → 180
    用于评估 SUMO edge 方向与摄像头期望方向的吻合程度。
    """
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def _normalize_road_name(value: str) -> str:
    """标准化道路名称: 去除首尾空白、转小写、删除内部空格。

    用于将摄像头配置中的路名与 SUMO edge 的 name 属性进行模糊匹配,
    避免因大小写或空格差异导致匹配失败。
    例如: "Avenida de Almeida Ribeiro" → "avenidadealmeidaribeiro"
    """
    return "".join(value.strip().lower().split())


def _shape_bearing(shape: list[tuple[float, float]]) -> float:
    """计算一条几何折线从起点到终点的总体方位角。

    以起点→终点的向量方向为准, 返回 0~360 度的方位角 (正北=0, 顺时针递增)。
    用于判断 SUMO edge 的行驶方向, 以便与摄像头的方向描述匹配。

    注意: atan2(dx, dy) 而非 atan2(dy, dx), 因为此处 y 轴指向北方,
    x 轴指向东方, 需要以北为基准顺时针计算。
    """
    if len(shape) < 2:
        return 0.0
    start_x, start_y = shape[0]
    end_x, end_y = shape[-1]
    dx = end_x - start_x
    dy = end_y - start_y
    return math.degrees(math.atan2(dx, dy)) % 360


def _serialize_path(path: Path | None) -> str | None:
    """将 Path 对象序列化为字符串, 用于 JSON 输出。"""
    return str(path) if path else None


# ============================================================================
# SUMO 环境发现与外部命令执行
# ============================================================================


def _run_command(cmd: list[str], cwd: Path, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    """执行外部命令 (SUMO 工具链), 捕获输出并可选地写入日志文件。

    本模块通过 subprocess 调用的 SUMO 命令包括:
      - netconvert: 将 OSM 路网转换为 SUMO .net.xml
      - od2trips: 将 OD 矩阵转换为个体出行 (trips)
      - sumo: 运行微观仿真
      - duaIterate.py: 动态用户均衡迭代 (Python 脚本)

    Args:
        cmd: 命令及其参数列表
        cwd: 工作目录
        log_path: 若提供, 将 stdout+stderr 写入该文件 (用于事后调试)

    Returns:
        已完成的子进程对象

    Raises:
        RuntimeError: 命令返回非零退出码时抛出, 包含完整 stdout/stderr
    """
    logger.info("Running command: %s", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write(completed.stdout)
            handle.write("\n")
            handle.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    return completed


def _ensure_sumo_home() -> Path:
    """检查并返回 SUMO_HOME 环境变量指向的路径。

    SUMO 安装目录下需包含 bin/ (可执行文件) 和 tools/ (Python 工具) 两个子目录。
    """
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise RuntimeError("SUMO_HOME is not set. SUMO backend requires a working SUMO installation.")
    return Path(sumo_home).resolve()


def discover_sumo_tools() -> tuple[dict[str, str], str]:
    """发现并验证本地 SUMO 安装, 返回所有工具路径和 SUMO 版本号。

    检查流程:
      1. 确认 SUMO_HOME 环境变量已设置
      2. 确认 tools/ 目录存在 (包含 sumolib 等 Python 库)
      3. 逐一解析必要的可执行文件路径 (优先 $SUMO_HOME/bin/, 其次 PATH)
      4. 确认 duaIterate.py 存在 (DUA 迭代的入口脚本)
      5. 调用 sumo --version 获取版本号

    Returns:
        (tool_paths, version) 元组:
          - tool_paths: 各工具路径的字典, 包含 netconvert, od2trips, sumo,
            python, duaIterate.py 等
          - version: SUMO 版本字符串 (如 "Eclipse SUMO Version 1.19.0")
    """
    sumo_home = _ensure_sumo_home()
    tools_dir = sumo_home / "tools"
    if not tools_dir.exists():
        raise RuntimeError(f"SUMO tools directory not found: {tools_dir}")

    def _resolve_binary(name: str) -> str:
        """先在 SUMO_HOME/bin/ 下查找, 找不到则从系统 PATH 中查找。"""
        candidate = sumo_home / "bin" / name
        if candidate.exists():
            return str(candidate)
        resolved = shutil.which(name)
        if resolved:
            return resolved
        raise RuntimeError(f"Required SUMO executable '{name}' is not available")

    # duaIterate.py 是 SUMO 的动态用户均衡迭代脚本, 位于 tools/assign/ 下
    dua_iterate = tools_dir / "assign" / "duaIterate.py"
    if not dua_iterate.exists():
        raise RuntimeError(f"duaIterate.py was not found under {dua_iterate}")

    tool_paths = {
        "SUMO_HOME": str(sumo_home),
        "tools_dir": str(tools_dir),
        "netconvert": _resolve_binary("netconvert"),   # OSM → net.xml 转换器
        "od2trips": _resolve_binary("od2trips"),       # OD矩阵 → trips.rou.xml
        "sumo": _resolve_binary("sumo"),               # SUMO 微观仿真引擎
        "python": sys.executable,                       # 当前 Python 解释器 (运行 duaIterate.py)
        "duaIterate.py": str(dua_iterate),             # DUA 迭代脚本
    }
    # 获取 SUMO 版本号 (取 --version 输出的第一行)
    version = _run_command([tool_paths["sumo"], "--version"], cwd=sumo_home).stdout.strip().splitlines()[0]
    return tool_paths, version


def _import_sumolib(tools_dir: str):
    """动态导入 sumolib (SUMO 的 Python 路网解析库)。

    sumolib 不是通过 pip 安装的, 而是随 SUMO 安装包附带在 tools/ 目录下。
    需要将 tools/ 添加到 sys.path 后才能 import。

    sumolib 提供的关键功能:
      - net.readNet(): 解析 .net.xml 路网文件
      - net.convertLonLat2XY(): 经纬度 → SUMO 投影坐标的转换
      - edge.allows(vClass): 检查 edge 是否允许某类车辆通行
    """
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    try:
        import sumolib  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Could not import sumolib from SUMO tools. "
            "Make sure SUMO_HOME points to a full SUMO installation."
        ) from exc
    return sumolib


# ============================================================================
# SUMO 路网加载与 edge 记录提取
# ============================================================================


def _vehicle_vclass(vehicle_type: str) -> str:
    """将项目车型名称转换为 SUMO vClass 名称。"""
    try:
        return VEHICLE_TO_VCLASS[vehicle_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported vehicle type for SUMO backend: {vehicle_type}") from exc


def _build_od_pairs_from_zone_mappings(
    zone_mappings: dict[int, SumoZoneMapping],
) -> list[tuple[int, int]]:
    """根据 zone 映射生成所有 OD 对 (排除 origin==dest 的自环)。

    23 个 zone → 23×22 = 506 个有效 OD 对。
    """
    zone_ids = sorted(zone_mappings.keys())
    return [(origin, dest) for origin in zone_ids for dest in zone_ids if origin != dest]


def _build_od_pairs_from_zone_ids(zone_ids: list[int]) -> list[tuple[int, int]]:
    """同 _build_od_pairs_from_zone_mappings, 但直接接受 zone_id 列表。"""
    return [(origin, dest) for origin in zone_ids for dest in zone_ids if origin != dest]


def _ensure_sumo_net(paths: ProjectPaths, tool_paths: dict[str, str]) -> Path:
    """确保 SUMO 路网文件 (.net.xml) 存在。

    如果已有 .net.xml 则直接返回; 否则从 OSM 文件通过 netconvert 生成。
    netconvert 是 SUMO 的路网转换工具, 可以将 OSM 道路数据转为 SUMO 仿真路网。
    """
    if paths.network_net_xml.exists():
        return paths.network_net_xml
    if not paths.network_osm.exists():
        raise FileNotFoundError(
            f"Missing SUMO net and OSM source. Expected {paths.network_net_xml} or {paths.network_osm}"
        )
    paths.network_net_xml.parent.mkdir(parents=True, exist_ok=True)
    _run_command(
        [
            tool_paths["netconvert"],
            "--osm-files",
            str(paths.network_osm),
            "-o",
            str(paths.network_net_xml),
        ],
        cwd=paths.root,
    )
    return paths.network_net_xml


def _edge_allows_vehicle(edge: Any, vehicle_type: str) -> bool:
    """检查一条 SUMO edge 是否允许指定车型通行。

    SUMO 中每条 edge 有 allow/disallow 属性, 限制特定 vClass 的通行。
    例如, 人行道不允许 "passenger", 高速公路不允许 "motorcycle" (某些设置下)。
    """
    vclass = _vehicle_vclass(vehicle_type)
    try:
        return bool(edge.allows(vclass))
    except Exception:
        # 如果 allows() 失败 (旧版 sumolib), 保守地假设允许通行
        return True


def _load_sumo_edge_records(net: Any, vehicle_type: str) -> tuple[list[_SumoEdgeRecord], dict[str, _SumoEdgeRecord]]:
    """从 SUMO 路网中提取所有可用的 edge 记录。

    过滤规则:
      1. 跳过内部 edge (以 ":" 开头, 这是交叉口内部的连接 edge)
      2. 跳过不允许该车型通行的 edge
      3. 跳过 shape 不足两点的 edge (无法计算方位角)

    Returns:
        (records, by_id) 元组:
          - records: 所有有效 edge 的列表
          - by_id: edge_id → record 的字典, 用于快速查找
    """
    records: list[_SumoEdgeRecord] = []
    by_id: dict[str, _SumoEdgeRecord] = {}
    for edge in net.getEdges():
        edge_id = str(edge.getID())
        # 跳过交叉口内部 edge (SUMO 中以 ":" 前缀标识)
        if edge_id.startswith(":"):
            continue
        # 跳过不允许该车型的 edge
        if not _edge_allows_vehicle(edge, vehicle_type):
            continue
        # 提取 edge 的几何形状 (投影坐标)
        shape = [(float(x), float(y)) for x, y in edge.getShape()]
        if len(shape) < 2:
            continue
        record = _SumoEdgeRecord(
            edge_id=edge_id,
            from_node_id=str(edge.getFromNode().getID()),
            to_node_id=str(edge.getToNode().getID()),
            shape=shape,
            name=str(getattr(edge, "getName", lambda: "")() or ""),
            length=float(getattr(edge, "getLength", lambda: 0.0)() or 0.0),
            speed=float(getattr(edge, "getSpeed", lambda: 0.0)() or 0.0),
            is_internal=edge_id.startswith(":"),
        )
        records.append(record)
        by_id[record.edge_id] = record
    return records, by_id


def _transform_geometry_to_xy(geometry: Any, net: Any) -> Any:
    """将 GeoJSON 的经纬度多边形转换为 SUMO 投影坐标系下的多边形。

    项目的 zone 边界文件 (macau_zones_23.geojson) 使用 WGS84 经纬度坐标,
    而 SUMO 路网使用投影坐标 (通常是 UTM 或自定义投影)。
    此函数通过 sumolib 的 convertLonLat2XY 逐点转换多边形的坐标,
    使 zone 多边形与 SUMO edge 在同一坐标系下, 便于做几何相交判断。

    支持 Polygon 和 MultiPolygon 两种几何类型。
    """
    if isinstance(geometry, Polygon):
        # 转换外环坐标
        shell = [net.convertLonLat2XY(float(x), float(y)) for x, y in geometry.exterior.coords]
        # 转换内环 (孔洞) 坐标
        holes = [
            [net.convertLonLat2XY(float(x), float(y)) for x, y in ring.coords]
            for ring in geometry.interiors
        ]
        return Polygon(shell, holes)
    if isinstance(geometry, MultiPolygon):
        return MultiPolygon([_transform_geometry_to_xy(poly, net) for poly in geometry.geoms])
    raise TypeError(f"Unsupported geometry type for TAZ conversion: {geometry.geom_type}")


# ============================================================================
# Zone → TAZ 映射 和 Gate → Edge 映射
# ============================================================================


def build_zone_mappings_from_records(
    zone_shapes: list[tuple[int, Any]],
    edge_records: list[_SumoEdgeRecord],
    centroid_xy_lookup: dict[int, tuple[float, float]],
) -> dict[int, SumoZoneMapping]:
    """将 23 个交通分区映射到 SUMO TAZ, 确定每个 zone 包含哪些 edge。

    SUMO 的 TAZ (Traffic Assignment Zone) 概念:
      每个 TAZ 关联一组 edge, 当车辆的 OD 起终点是某个 TAZ 时,
      SUMO 的 od2trips 工具会随机选择 TAZ 内的一条 edge 作为实际出发/到达点。

    映射逻辑 (对每个 zone):
      1. 找出所有与 zone 多边形相交的 edge, 或者中点落在 zone 内的 edge
      2. 如果没有任何 edge 被匹配 (通常是因为 zone 面积很小或边缘),
         则启用"质心回退": 找距离 zone 质心最近的 edge

    Args:
        zone_shapes: [(zone_id, 投影坐标系下的多边形)] 列表
        edge_records: 所有可用的 SUMO edge 记录
        centroid_xy_lookup: {zone_id: (x, y)} zone 质心的投影坐标

    Returns:
        {zone_id: SumoZoneMapping} 映射字典
    """
    # 预计算每条 edge 的 LineString, 避免重复构造
    line_cache = {edge.edge_id: edge.line for edge in edge_records}
    mappings: dict[int, SumoZoneMapping] = {}
    for zone_id, geometry_xy in zone_shapes:
        # 方法一: 找与 zone 多边形几何相交的 edge, 或中点在多边形内的 edge
        matching_edges = [
            edge.edge_id
            for edge in edge_records
            if line_cache[edge.edge_id].intersects(geometry_xy)
            or geometry_xy.contains(line_cache[edge.edge_id].centroid)
        ]
        used_centroid_fallback = False
        if not matching_edges:
            # 方法二 (回退): 若无交集, 取距离 zone 质心最近的单条 edge
            centroid = Point(*centroid_xy_lookup[zone_id])
            fallback_edge = min(edge_records, key=lambda edge: line_cache[edge.edge_id].distance(centroid))
            matching_edges = [fallback_edge.edge_id]
            used_centroid_fallback = True
        mappings[zone_id] = SumoZoneMapping(
            zone_id=zone_id,
            taz_id=f"taz_{zone_id}",             # SUMO TAZ 的 ID 命名约定
            edge_ids=sorted(set(matching_edges)),  # 去重并排序
            centroid_xy=centroid_xy_lookup[zone_id],
            used_centroid_fallback=used_centroid_fallback,
        )
    return mappings


def map_gates_to_sumo_records(
    gates: list[GateInfo],
    edge_records: list[_SumoEdgeRecord],
    camera_xy_lookup: dict[str, tuple[float, float]],
) -> dict[int, SumoGateMapping]:
    """将每个摄像头 gate 映射到 SUMO 路网中最匹配的 from_edge 和 to_edge。

    每个 gate 描述了一个检测线上的转向运动:
      - origin_direction + origin_road → 车辆驶来的方向/道路 → from_edge
      - dest_direction + dest_road → 车辆驶去的方向/道路 → to_edge

    映射流程 (对每个 gate):
      1. 以摄像头位置为中心, 计算所有 edge 到该点的距离, 按距离排序
      2. 从距离最近的候选 edge 中, 综合考虑 道路名称匹配 和 方位角匹配,
         分别选出最佳的 from_edge 和 to_edge
      3. 根据匹配结果设置状态: mapped_turning / mapped_from_only / mapped_to_only / unmapped

    关于 from_edge 的方位角:
      摄像头配置中 origin_direction 描述的是"车辆从哪个方向来",
      而 SUMO edge 的方位角描述的是"这条路的行驶方向"。
      所以 from_edge 的期望方位角 = origin_direction + 180° (反向)。
      例如: 车从北方来 (origin_direction=north=0°), 那 from_edge 应朝南行驶 (180°)。

    Args:
        gates: 所有 gate 的信息列表
        edge_records: 所有可用的 SUMO edge 记录
        camera_xy_lookup: {camera_id: (x, y)} 摄像头位置的投影坐标

    Returns:
        {gate_index: SumoGateMapping} 映射字典
    """
    line_cache = {edge.edge_id: edge.line for edge in edge_records}
    gate_mappings: dict[int, SumoGateMapping] = {}
    for gate in gates:
        camera_xy = camera_xy_lookup[gate.camera_id]
        point = Point(*camera_xy)
        # 按距离排序所有 edge, 选出候选集
        candidates_with_distance = sorted(
            ((edge, line_cache[edge.edge_id].distance(point)) for edge in edge_records),
            key=lambda item: item[1],
        )
        # 优先使用 80m 半径内的候选; 不足2条时用距离最近的前20条
        close_candidates = [item for item in candidates_with_distance if item[1] <= DEFAULT_EDGE_SEARCH_RADIUS]
        if len(close_candidates) >= 2:
            candidates = close_candidates
        else:
            candidates = candidates_with_distance[:DEFAULT_EDGE_SEARCH_LIMIT]

        # 选择 from_edge: 期望方位角 = origin_direction + 180° (车来的反方向)
        from_edge, from_distance = _pick_best_edge(
            candidates=candidates,
            expected_bearing=((direction_to_bearing(gate.origin_direction) or 0.0) + 180.0) % 360
            if direction_to_bearing(gate.origin_direction) is not None
            else None,
            expected_road_name=gate.origin_road,
        )
        # 选择 to_edge: 期望方位角 = dest_direction (车去的方向), 且排除 from_edge
        to_edge, to_distance = _pick_best_edge(
            candidates=candidates,
            expected_bearing=direction_to_bearing(gate.dest_direction),
            expected_road_name=gate.dest_road,
            exclude_edge_id=from_edge.edge_id if from_edge is not None else None,
        )
        # 根据匹配结果设置状态
        status = "unmapped"
        if from_edge is not None and to_edge is not None:
            status = "mapped_turning"      # 完整转向映射 (最佳)
        elif from_edge is not None:
            status = "mapped_from_only"    # 只匹配到驶入 edge
        elif to_edge is not None:
            status = "mapped_to_only"      # 只匹配到驶出 edge
        gate_mappings[gate.gate_index] = SumoGateMapping(
            gate_index=gate.gate_index,
            camera_id=gate.camera_id,
            gate_id=gate.gate_id,
            from_edge_id=from_edge.edge_id if from_edge is not None else None,
            to_edge_id=to_edge.edge_id if to_edge is not None else None,
            status=status,
            from_distance=from_distance,
            to_distance=to_distance,
        )
    return gate_mappings


def _pick_best_edge(
    candidates: list[tuple[_SumoEdgeRecord, float]],
    expected_bearing: float | None,
    expected_road_name: str = "",
    exclude_edge_id: str | None = None,
) -> tuple[_SumoEdgeRecord | None, float | None]:
    """从候选 edge 中选出与期望方向和路名最匹配的 edge。

    选择策略 (优先级从高到低):
      1. 如果有路名匹配的 edge, 优先从中选择
      2. 在候选集中, 按 (方位角差, 距离) 联合排序, 取最小值
      3. 如果最佳候选的方位角差 > 120°, 认为方向不匹配, 返回 None

    120° 阈值的含义:
      允许一定的角度偏差 (如 edge 弯曲、道路局部方向变化),
      但超过 120° 说明候选 edge 的方向与期望方向严重不符。

    Args:
        candidates: [(edge, distance)] 候选列表
        expected_bearing: 期望方位角 (度), None 则仅按距离选
        expected_road_name: 期望路名 (用于优先匹配)
        exclude_edge_id: 需排除的 edge ID (防止 from 和 to 选到同一条)
    """
    if not candidates:
        return None, None
    expected_road_name = _normalize_road_name(expected_road_name)
    # 排除指定 edge (通常是已选为 from_edge 的那条)
    filtered = [
        (edge, dist)
        for edge, dist in candidates
        if exclude_edge_id is None or edge.edge_id != exclude_edge_id
    ]
    if not filtered:
        return None, None
    # 优先使用路名匹配的子集
    road_matched = [
        (edge, dist)
        for edge, dist in filtered
        if expected_road_name and _normalize_road_name(edge.name) == expected_road_name
    ]
    working = road_matched or filtered
    # 若无期望方位角, 仅按距离选最近的
    if expected_bearing is None:
        edge, dist = min(working, key=lambda item: item[1])
        return edge, float(dist)
    # 按 (方位角差, 距离) 联合排序
    best_edge, best_dist = min(
        working,
        key=lambda item: (_angular_diff(_shape_bearing(item[0].shape), expected_bearing), item[1]),
    )
    # 方位角差超过 120° 则拒绝匹配
    diff = _angular_diff(_shape_bearing(best_edge.shape), expected_bearing)
    if diff > 120.0:
        return None, None
    return best_edge, float(best_dist)


# ============================================================================
# TAZ 文件生成与序列化
# ============================================================================


def _write_taz_file(path: Path, zone_mappings: dict[int, SumoZoneMapping]) -> Path:
    """生成 SUMO TAZ 定义文件 (.taz.xml)。

    TAZ 文件格式示例:
      <tazs>
        <taz id="taz_1" edges="12345 12346 12347"/>
        <taz id="taz_2" edges="22345 22346"/>
        ...
      </tazs>

    SUMO 的 od2trips 工具读取此文件, 将 OD 矩阵中的 zone-level 需求
    分配到具体的 edge 上。
    """
    root = ET.Element(
        "tazs",
        {
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:noNamespaceSchemaLocation": "http://sumo.dlr.de/xsd/taz_file.xsd",
        },
    )
    for zone_id in sorted(zone_mappings):
        zone_mapping = zone_mappings[zone_id]
        ET.SubElement(
            root,
            "taz",
            {
                "id": zone_mapping.taz_id,
                "edges": " ".join(zone_mapping.edge_ids),  # 空格分隔的 edge ID 列表
            },
        )
    tree = ET.ElementTree(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def _serialize_zone_mappings(zone_mappings: dict[int, SumoZoneMapping]) -> dict[str, Any]:
    """将 zone 映射序列化为可 JSON 化的字典 (用于写入元数据文件)。"""
    return {
        str(zone_id): {
            "taz_id": mapping.taz_id,
            "edge_ids": mapping.edge_ids,
            "centroid_xy": list(mapping.centroid_xy),
            "used_centroid_fallback": mapping.used_centroid_fallback,
        }
        for zone_id, mapping in zone_mappings.items()
    }


def _serialize_gate_mappings(gate_mappings: dict[int, SumoGateMapping]) -> list[dict[str, Any]]:
    """将 gate 映射序列化为可 JSON 化的列表 (按 gate_index 排序)。"""
    return [asdict(gate_mappings[idx]) for idx in sorted(gate_mappings)]


# ============================================================================
# ensure_sumo_assets: SUMO 资产准备的主入口
# ============================================================================


def ensure_sumo_assets(
    gates: list[GateInfo],
    vehicle_type: str,
    paths: ProjectPaths | None = None,
    sumo_config: SumoAssignmentConfig | None = None,
) -> tuple[SumoNativeMappings, SumoAssignmentArtifacts]:
    """准备 SUMO 仿真所需的所有前置资产 (TAZ 文件、zone/gate 映射), 并执行验证。

    这是 SUMO 后端的核心编排函数, 在运行仿真前必须调用。
    它完成以下步骤:
      1. 发现 SUMO 环境 → 加载路网 → 提取 edge
      2. 加载 23 个 zone 的 GeoJSON, 转换坐标系, 建立 zone→TAZ 映射
      3. 将摄像头 gate 映射到 SUMO edge
      4. 写出 TAZ 文件 (供 od2trips 使用)
      5. 验证映射质量 (空 zone 检查、edge 存在性检查、图连通性检查)
      6. 将所有映射和验证结果写入 JSON 元数据文件

    Args:
        gates: 所有摄像头 gate 信息
        vehicle_type: 车型 ("car", "bus", "truck", "motorcycle")
        paths: 项目路径配置 (默认自动推导)
        sumo_config: SUMO 仿真配置 (默认参数)

    Returns:
        (mappings, artifacts) 元组:
          - mappings: zone/gate 映射的综合对象
          - artifacts: 产出文件路径和诊断统计的集合
    """
    paths = paths or default_paths()
    sumo_config = sumo_config or SumoAssignmentConfig()
    paths.ensure_output_dir()

    # ---- 步骤 1: 发现 SUMO 环境, 加载路网, 提取可用 edge ----
    tool_paths, version = discover_sumo_tools()
    net_file = _ensure_sumo_net(paths, tool_paths)
    sumolib = _import_sumolib(tool_paths["tools_dir"])
    net = sumolib.net.readNet(str(net_file), withInternal=True)
    edge_records, edge_by_id = _load_sumo_edge_records(net, vehicle_type)
    if not edge_records:
        raise RuntimeError(f"No usable SUMO edges were found for vehicle type '{vehicle_type}'")

    # ---- 步骤 2: 加载 zone GeoJSON, 坐标转换, 建立 zone→TAZ 映射 ----
    zones = load_zones_geojson(paths.zones_geojson)
    zone_shapes_xy: list[tuple[int, Any]] = []
    centroid_xy_lookup: dict[int, tuple[float, float]] = {}
    for zone_id, geometry in zones:
        # 将经纬度多边形转为投影坐标
        geometry_xy = _transform_geometry_to_xy(geometry, net)
        # 计算 zone 质心的投影坐标 (用于回退匹配和输出)
        centroid_lonlat = geometry.representative_point()
        centroid_xy_lookup[zone_id] = tuple(
            map(float, net.convertLonLat2XY(float(centroid_lonlat.x), float(centroid_lonlat.y)))
        )
        zone_shapes_xy.append((zone_id, geometry_xy))
    zone_mappings = build_zone_mappings_from_records(zone_shapes_xy, edge_records, centroid_xy_lookup)

    # ---- 步骤 3: 将摄像头位置从经纬度转为投影坐标, 映射 gate→edge ----
    camera_xy_lookup = {
        gate.camera_id: tuple(map(float, net.convertLonLat2XY(gate.camera_lon, gate.camera_lat)))
        for gate in gates
    }
    gate_mappings = map_gates_to_sumo_records(gates, edge_records, camera_xy_lookup)

    # ---- 步骤 4: 写出 TAZ 文件和映射元数据 ----
    period_dir = paths.sumo_period_dir(sumo_config.period_name)
    vehicle_dir = paths.sumo_vehicle_dir(sumo_config.period_name, vehicle_type)
    vehicle_dir.mkdir(parents=True, exist_ok=True)
    taz_file = vehicle_dir / f"macau_zones_23__{vehicle_type}.taz.xml"
    mapping_file = vehicle_dir / f"sumo_native_mappings__{vehicle_type}.json"
    _write_taz_file(taz_file, zone_mappings)

    # ---- 步骤 5: 计算映射质量统计并执行验证 ----
    gate_from_mapped = sum(1 for mapping in gate_mappings.values() if mapping.from_edge_id)
    gate_turning_mapped = sum(
        1 for mapping in gate_mappings.values() if mapping.from_edge_id and mapping.to_edge_id
    )
    validation = {
        "vehicle_type": vehicle_type,
        "n_zones": len(zone_mappings),                # zone 总数
        "n_zone_fallbacks": sum(1 for mapping in zone_mappings.values() if mapping.used_centroid_fallback),  # 使用质心回退的 zone 数
        "n_gates": len(gate_mappings),                # gate 总数
        "n_gate_from_mapped": gate_from_mapped,       # 成功映射 from_edge 的 gate 数
        "n_gate_turning_mapped": gate_turning_mapped, # 成功映射完整转向 (from+to) 的 gate 数
    }

    _validate_zone_mappings(zone_mappings)                      # 检查是否有 zone 没有关联任何 edge
    _validate_gate_mappings(gate_mappings, edge_by_id)          # 检查映射的 edge 是否在路网中存在
    scc_stats = _compute_scc_stats(edge_records, zone_mappings) # 检查所有 TAZ 是否在路网图的最大强连通分量内
    validation.update(scc_stats)

    # ---- 步骤 6: 将映射和验证结果写入 JSON ----
    with open(mapping_file, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "vehicle_type": vehicle_type,
                "sumo_version": version,
                "tool_paths": tool_paths,
                "net_file": str(net_file),
                "taz_file": str(taz_file),
                "zone_mappings": _serialize_zone_mappings(zone_mappings),
                "gate_mappings": _serialize_gate_mappings(gate_mappings),
                "validation": validation,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # 封装返回值
    mappings = SumoNativeMappings(
        zone_mappings=zone_mappings,
        gate_mappings=gate_mappings,
        taz_file=taz_file,
        gate_mapping_file=mapping_file,
        n_zone_fallbacks=validation["n_zone_fallbacks"],
        n_gate_from_mapped=gate_from_mapped,
        n_gate_turning_mapped=gate_turning_mapped,
    )
    artifacts = SumoAssignmentArtifacts(
        vehicle_type=vehicle_type,
        period_name=sumo_config.period_name,
        output_dir=vehicle_dir,
        sumo_version=version,
        tool_paths=tool_paths,
        files={
            "net_file": str(net_file),
            "taz_file": str(taz_file),
            "gate_mapping_file": str(mapping_file),
        },
        diagnostics=validation,
    )
    return mappings, artifacts


# ============================================================================
# 映射验证与路网连通性检查
# ============================================================================


def _validate_zone_mappings(zone_mappings: dict[int, SumoZoneMapping]) -> None:
    """验证所有 zone 都至少关联了一条 edge (TAZ 不为空)。"""
    empty = [zone_id for zone_id, mapping in zone_mappings.items() if not mapping.edge_ids]
    if empty:
        raise RuntimeError(f"TAZ generation failed for zones without candidate edges: {empty}")


def _validate_gate_mappings(gate_mappings: dict[int, SumoGateMapping], edge_by_id: dict[str, _SumoEdgeRecord]) -> None:
    """验证 gate 映射引用的 edge 确实存在于路网中。

    这是一个一致性检查: 如果 gate 的 from_edge_id 指向了一个不存在的 edge,
    说明映射过程有 bug (理论上不应发生)。
    """
    missing = [
        mapping.gate_index
        for mapping in gate_mappings.values()
        if mapping.from_edge_id and mapping.from_edge_id not in edge_by_id
    ]
    if missing:
        raise RuntimeError(f"Gate mappings reference SUMO edges that are absent from net.xml: {missing}")


def _compute_scc_stats(
    edge_records: list[_SumoEdgeRecord],
    zone_mappings: dict[int, SumoZoneMapping],
) -> dict[str, int]:
    """检查路网图的强连通性, 确保所有 TAZ 在同一个强连通分量 (SCC) 中。

    为什么需要强连通分量检查?
      SUMO DUA 要求任意 OD 对之间必须存在可达路径。
      如果某个 zone 的 TAZ edge 全部位于主强连通分量之外,
      那么从该 zone 出发 (或到达该 zone) 的车辆将无法完成路径查找,
      导致仿真失败。

    算法:
      1. 将所有 edge 构建为有向图 (from_node → to_node)
      2. 用 NetworkX 找出最大强连通分量 (largest SCC)
      3. 检查每个 zone 的 TAZ edge 是否至少有一条在最大 SCC 中
    """
    # 构建有向图
    graph = nx.DiGraph()
    for edge in edge_records:
        graph.add_edge(edge.from_node_id, edge.to_node_id, edge_id=edge.edge_id)
    if graph.number_of_nodes() == 0:
        raise RuntimeError("SUMO graph is empty after filtering edges")
    # 找最大强连通分量
    largest_scc = max(nx.strongly_connected_components(graph), key=len)
    # 找出最大 SCC 中包含的 edge
    scc_edge_ids = {
        data["edge_id"]
        for u, v, data in graph.edges(data=True)
        if u in largest_scc and v in largest_scc
    }
    # 检查是否有 zone 的 edge 完全不在最大 SCC 中
    disconnected_zones = [
        zone_id
        for zone_id, mapping in zone_mappings.items()
        if not set(mapping.edge_ids).intersection(scc_edge_ids)
    ]
    if disconnected_zones:
        raise RuntimeError(
            f"TAZ edges for zones {disconnected_zones} are outside the main strongly connected component"
        )
    return {
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "largest_scc_nodes": len(largest_scc),
        "largest_scc_edges": len(scc_edge_ids),
    }


# ============================================================================
# simulate_posterior_sumo: 后验 OD 矩阵的 SUMO 仿真主流程
# ============================================================================


def simulate_posterior_sumo(
    gates: list[GateInfo],
    vehicle_type: str,
    zone_ids: list[int] | None = None,
    posterior_od_matrix: np.ndarray | None = None,
    od_csv_path: Path | None = None,
    paths: ProjectPaths | None = None,
    sumo_config: SumoAssignmentConfig | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
) -> SumoSimulationResult:
    """运行后验 OD 矩阵的 SUMO 微观仿真, 返回仿真产出的分配矩阵 H。

    这是 SUMO 后端对外的主入口函数, 完整流程:
      1. 读取后验 OD 矩阵 (从内存 ndarray 或 CSV 文件)
      2. 调用 ensure_sumo_assets() 准备 TAZ 文件和映射
      3. 将 OD 矩阵转换为 SUMO 需求并四舍五入为整数出行数
      4. 检查缓存 — 若已有相同参数的仿真结果, 直接加载
      5. 运行 SUMO DUA 仿真 (_run_posterior_sumo)
      6. 从仿真路线结果中构建 H 矩阵 (_build_h_from_sumo_routes)
      7. 将结果缓存到磁盘

    Args:
        gates: 所有 gate 信息
        vehicle_type: 车型
        zone_ids: zone ID 列表 (与 OD 矩阵行列对应)
        posterior_od_matrix: 后验 OD 矩阵 (23×23 ndarray), 与 od_csv_path 二选一
        od_csv_path: 后验 OD 矩阵的 CSV 文件路径, 与 posterior_od_matrix 二选一
        paths: 项目路径配置
        sumo_config: SUMO 仿真参数
        use_cache: 是否尝试从磁盘加载缓存
        write_cache: 是否将结果写入磁盘缓存

    Returns:
        SumoSimulationResult, 包含:
          - H: 基于 SUMO 仿真路线构建的分配矩阵 (n_gates × n_od_pairs, 稀疏)
          - od_pairs: OD 对列表
          - mappings: zone/gate 映射
          - artifacts: 仿真过程文件和诊断信息
    """
    paths = paths or default_paths()
    sumo_config = sumo_config or SumoAssignmentConfig()

    # ---- 读取后验 OD 矩阵 ----
    if posterior_od_matrix is None:
        if od_csv_path is None:
            raise ValueError("posterior_od_matrix or od_csv_path is required for posterior SUMO simulation")
        csv_zone_ids, posterior_od_matrix = _load_posterior_od_matrix_csv(od_csv_path)
        if zone_ids is None:
            zone_ids = csv_zone_ids
        elif list(zone_ids) != csv_zone_ids:
            raise ValueError("zone_ids does not match the ordering in od_csv_path")
    elif zone_ids is None:
        raise ValueError("zone_ids is required when posterior_od_matrix is provided")

    zone_ids = [int(zone_id) for zone_id in zone_ids]

    # ---- 准备 SUMO 资产 (TAZ、映射) ----
    mappings, artifacts = ensure_sumo_assets(
        gates=gates,
        vehicle_type=vehicle_type,
        paths=paths,
        sumo_config=sumo_config,
    )
    missing_zones = [zone_id for zone_id in zone_ids if zone_id not in mappings.zone_mappings]
    if missing_zones:
        raise ValueError(f"Zones {missing_zones} are not available in SUMO TAZ mappings")

    # ---- 准备需求元数据 (将浮点 OD 值四舍五入为整数出行数) ----
    demand_metadata = _prepare_posterior_demand_metadata(
        zone_ids=zone_ids,
        posterior_od_matrix=posterior_od_matrix,
        vehicle_type=vehicle_type,
        period_name=sumo_config.period_name,
        od_csv_path=od_csv_path,
    )
    if int(demand_metadata["total_rounded_trips"]) <= 0:
        raise ValueError(
            f"Posterior OD for vehicle '{vehicle_type}' rounds to zero trips; SUMO simulation cannot proceed"
        )
    od_pairs = [(int(origin), int(dest)) for origin, dest in demand_metadata["od_pairs"]]

    # ---- 计算缓存路径 (基于输入参数的 SHA1 哈希) ----
    matrix_path, metadata_path = _posterior_sumo_cache_paths(
        paths=paths,
        gates=gates,
        od_pairs=od_pairs,
        vehicle_type=vehicle_type,
        mappings=mappings,
        sumo_config=sumo_config,
        rounded_counts=demand_metadata["rounded_counts"],
    )
    artifacts.files["matrix_path"] = str(matrix_path)
    artifacts.files["metadata_path"] = str(metadata_path)

    # ---- 缓存命中: 直接加载已有结果 ----
    if use_cache and matrix_path.exists() and metadata_path.exists():
        logger.info("Loading cached posterior SUMO matrix from %s", matrix_path)
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        cached_artifacts = metadata.get("artifacts", {})
        artifacts.files.update(cached_artifacts.get("files", {}))
        artifacts.gap_history = list(cached_artifacts.get("gap_history", []))
        artifacts.diagnostics.update(cached_artifacts.get("diagnostics", {}))
        H = sp.load_npz(matrix_path)
        return SumoSimulationResult(
            vehicle_type=vehicle_type,
            od_pairs=od_pairs,
            H=H,
            mappings=mappings,
            artifacts=artifacts,
        )

    # ---- 缓存未命中: 运行完整 SUMO 仿真 ----
    run_outputs = _run_posterior_sumo(
        vehicle_type=vehicle_type,
        od_pairs=od_pairs,
        mappings=mappings,
        paths=paths,
        sumo_config=sumo_config,
        artifacts=artifacts,
        demand_metadata=demand_metadata,
    )
    # 从仿真路线结果反向构建 H 矩阵
    H = _build_h_from_sumo_routes(
        trips_file=Path(run_outputs["trips_file"]),
        route_file=Path(run_outputs["final_route_file"]),
        gate_mappings=mappings.gate_mappings,
        od_pairs=od_pairs,
        n_gates=len(gates),
    )
    # 将 H 矩阵和元数据写入缓存
    if write_cache:
        paths.ensure_output_dir()
        sp.save_npz(matrix_path, H)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "assignment_backend": "posterior_sumo",
                    "vehicle_type": vehicle_type,
                    "period_name": sumo_config.period_name,
                    "sumo_config": asdict(sumo_config),
                    "zone_ids": zone_ids,
                    "od_pairs": od_pairs,
                    "demand_metadata": demand_metadata,
                    "gate_mappings": _serialize_gate_mappings(mappings.gate_mappings),
                    "artifacts": {
                        "sumo_version": artifacts.sumo_version,
                        "files": artifacts.files,
                        "tool_paths": artifacts.tool_paths,
                        "gap_history": artifacts.gap_history,
                        "diagnostics": artifacts.diagnostics,
                    },
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
    return SumoSimulationResult(
        vehicle_type=vehicle_type,
        od_pairs=od_pairs,
        H=H,
        mappings=mappings,
        artifacts=artifacts,
    )


def _load_posterior_od_matrix_csv(od_csv_path: Path) -> tuple[list[int], np.ndarray]:
    """从 CSV 文件加载后验 OD 矩阵。

    CSV 格式:
      ,1,2,3,...,23      ← 列头: zone ID
      1,0.5,1.2,...,0.3  ← 第一行: zone 1 到各 zone 的出行量
      2,0.8,...          ← 第二行: zone 2 到各 zone 的出行量
      ...

    Returns:
        (zone_ids, matrix): zone ID 列表和 23×23 的 OD 矩阵 ndarray
    """
    with open(od_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None or len(header) < 2:
            raise ValueError(f"OD CSV is missing a valid header: {od_csv_path}")
        zone_ids = [int(value) for value in header[1:]]  # 跳过第一列 (空或标签)
        matrix_rows: list[list[float]] = []
        row_zone_ids: list[int] = []
        for row in reader:
            if not row:
                continue
            row_zone_ids.append(int(row[0]))              # 行标签 = zone ID
            matrix_rows.append([float(value) for value in row[1:]])
    # 验证行列 zone ID 一致
    if row_zone_ids != zone_ids:
        raise ValueError(f"OD CSV row/column zone ordering does not match in {od_csv_path}")
    matrix = np.array(matrix_rows, dtype=float)
    if matrix.shape != (len(zone_ids), len(zone_ids)):
        raise ValueError(
            f"OD CSV matrix shape {matrix.shape} does not match zone count {len(zone_ids)} in {od_csv_path}"
        )
    return zone_ids, matrix


def _round_trip_count(value: float) -> int:
    """将浮点出行量四舍五入为非负整数。

    SUMO 只能处理整数出行数, 贝叶斯后验的 OD 值是浮点数,
    因此需要四舍五入。负值先截断为 0。
    """
    return int(math.floor(max(float(value), 0.0) + 0.5))


def _prepare_posterior_demand_metadata(
    zone_ids: list[int],
    posterior_od_matrix: np.ndarray,
    vehicle_type: str,
    period_name: str,
    od_csv_path: Path | None = None,
) -> dict[str, Any]:
    """将后验 OD 矩阵转换为 SUMO 需求的元数据结构。

    关键操作:
      - 将 23×23 矩阵展开为 506 个 OD 对 (排除对角线)
      - 对每个 OD 对的浮点出行量四舍五入为整数
      - 统计总出行数和有效 (>0) 的 OD 对数

    Returns:
        包含 demand_entries, rounded_counts, total_rounded_trips 等字段的字典,
        后续供 _write_tazrelation_od() 和缓存签名使用
    """
    matrix = np.asarray(posterior_od_matrix, dtype=float)
    if matrix.shape != (len(zone_ids), len(zone_ids)):
        raise ValueError(
            f"posterior_od_matrix shape {matrix.shape} does not match zone_ids length {len(zone_ids)}"
        )

    od_pairs = _build_od_pairs_from_zone_ids(zone_ids)
    zone_idx = {zone_id: idx for idx, zone_id in enumerate(zone_ids)}
    demand_entries: list[dict[str, Any]] = []
    raw_counts: list[float] = []
    rounded_counts: list[int] = []
    for origin, dest in od_pairs:
        raw_value = max(float(matrix[zone_idx[origin], zone_idx[dest]]), 0.0)
        rounded_value = _round_trip_count(raw_value)
        raw_counts.append(raw_value)
        rounded_counts.append(rounded_value)
        demand_entries.append(
            {
                "origin": origin,
                "dest": dest,
                "raw_count": raw_value,
                "rounded_count": rounded_value,
            }
        )
    return {
        "vehicle_type": vehicle_type,
        "period_name": period_name,
        "source_od_csv": str(od_csv_path) if od_csv_path is not None else None,
        "zone_ids": zone_ids,
        "od_pairs": [[origin, dest] for origin, dest in od_pairs],
        "raw_counts": raw_counts,
        "rounded_counts": rounded_counts,
        "demand_entries": demand_entries,
        "total_raw_trips": float(sum(raw_counts)),       # 四舍五入前的总出行量
        "total_rounded_trips": int(sum(rounded_counts)),  # 四舍五入后的总出行数
        "positive_od_pairs": sum(1 for value in rounded_counts if value > 0),  # 有效 OD 对数
    }


def _posterior_sumo_cache_paths(
    paths: ProjectPaths,
    gates: list[GateInfo],
    od_pairs: list[tuple[int, int]],
    vehicle_type: str,
    mappings: SumoNativeMappings,
    sumo_config: SumoAssignmentConfig,
    rounded_counts: list[int],
) -> tuple[Path, Path]:
    """计算仿真结果的缓存文件路径 (基于输入参数的内容哈希)。

    缓存机制:
      将所有影响仿真结果的参数 (车型、配置、gate映射、OD需求) 序列化为 JSON,
      取 SHA1 哈希的前 12 位作为缓存键。只要参数不变, 哈希相同, 可直接复用结果。

    Returns:
        (matrix_path, metadata_path): H 矩阵的 .npz 文件和元数据的 .json 文件路径
    """
    vehicle_dir = paths.sumo_vehicle_dir(sumo_config.period_name, vehicle_type)
    # 构造缓存签名: 包含 gate 映射的 edge ID, 确保映射变化时缓存失效
    gate_signature = [
        [
            gate.gate_index,
            gate.camera_id,
            gate.gate_id,
            mappings.gate_mappings[gate.gate_index].from_edge_id,
            mappings.gate_mappings[gate.gate_index].to_edge_id,
        ]
        for gate in gates
    ]
    payload = {
        "assignment_backend": "posterior_sumo",
        "vehicle_type": vehicle_type,
        "period_name": sumo_config.period_name,
        "config": asdict(sumo_config),
        "gate_signature": gate_signature,
        "od_pairs": od_pairs,
        "rounded_counts": rounded_counts,
    }
    # 取 SHA1 前 12 位作为文件名后缀
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    base = (
        f"assignment_matrix__backend-posterior_sumo__period-{sumo_config.period_name}"
        f"__veh-{vehicle_type}__{digest}"
    )
    return vehicle_dir / f"{base}.npz", vehicle_dir / f"{base}.json"


def _run_posterior_sumo(
    vehicle_type: str,
    od_pairs: list[tuple[int, int]],
    mappings: SumoNativeMappings,
    paths: ProjectPaths,
    sumo_config: SumoAssignmentConfig,
    artifacts: SumoAssignmentArtifacts,
    demand_metadata: dict[str, Any],
) -> dict[str, str]:
    """执行完整的 SUMO 仿真流程: od2trips → DUA迭代 → SUMO仿真 → 后处理。

    这是整个 SUMO 仿真的核心执行函数, 依次调用四个 SUMO 工具:

    步骤 1: od2trips — OD 需求分解为个体出行
      输入: TAZ 文件 + OD 需求 XML
      输出: trips.rou.xml (每辆车一个 <trip> 元素, 包含起讫 TAZ)
      --spread.uniform: 出行出发时间在 [begin, end] 区间内均匀分布

    步骤 2: duaIterate.py — 动态用户均衡 (DUA) 迭代
      输入: 路网 + trips
      过程: 反复运行 duarouter (路径选择) + sumo (仿真), 迭代收敛到均衡状态
      输出: 每次迭代的 route 文件, gap 值逐步降低
      关键参数:
        - due_iterations: 迭代次数 (默认10)
        - route_choice_model: 路径选择模型 (Gawron)
        - gawron_beta / gawron_a: Gawron 模型参数

    步骤 3: sumo — 最终仿真
      输入: 路网 + 最终路线文件 + additional 配置 (edge 流量输出)
      输出: edge 级别的流量统计

    步骤 4: 后处理
      - 解析 edge 流量 → CSV
      - 统计 gate 级别的转向流量
      - 计算路径诊断统计

    Returns:
        各产出文件路径的字典
    """
    vehicle_dir = paths.sumo_vehicle_dir(sumo_config.period_name, vehicle_type)
    vehicle_dir.mkdir(parents=True, exist_ok=True)

    # 定义所有输出文件路径
    od_file = vehicle_dir / f"posterior_od_demand__{vehicle_type}.xml"            # OD需求XML (SUMO tazRelation格式)
    demand_metadata_file = vehicle_dir / f"posterior_od_demand__{vehicle_type}.json"  # 需求元数据
    trips_file = vehicle_dir / f"trips__posterior_sumo__{vehicle_type}.rou.xml"       # od2trips输出的个体出行
    dua_log = vehicle_dir / f"duaIterate__posterior_sumo__{vehicle_type}.log"         # DUA迭代日志
    final_route_file = vehicle_dir / f"routes_final__posterior_sumo__{vehicle_type}.rou.xml"  # DUA最终路线
    additional_file = vehicle_dir / f"outputs__posterior_sumo__{vehicle_type}.add.xml"  # SUMO附加配置 (启用edge流量输出)
    edge_flows_xml = vehicle_dir / f"edge_flows__posterior_sumo__{vehicle_type}.xml"   # SUMO输出的edge流量XML
    edge_flows_csv = vehicle_dir / f"edge_flows__posterior_sumo__{vehicle_type}.csv"   # 解析后的edge流量CSV
    sumo_log = vehicle_dir / f"sumo_final__posterior_sumo__{vehicle_type}.log"         # 最终仿真日志
    turning_file = vehicle_dir / f"turning_{vehicle_type}__posterior_sumo.json"        # gate级转向统计
    artifacts_file = vehicle_dir / f"posterior_sumo_artifacts__{vehicle_type}.json"     # 完整artifacts记录

    # ---- 写出 OD 需求文件 (SUMO tazRelation 格式) ----
    _write_tazrelation_od(
        path=od_file,
        zone_mappings=mappings.zone_mappings,
        begin=sumo_config.begin,
        end=sumo_config.end,
        demand_entries=demand_metadata["demand_entries"],
    )
    with open(demand_metadata_file, "w", encoding="utf-8") as handle:
        json.dump(demand_metadata, handle, indent=2, ensure_ascii=False)

    # 记录文件路径到 artifacts
    artifacts.files.update(
        {
            "od_file": str(od_file),
            "demand_metadata_file": str(demand_metadata_file),
            "trips_file": str(trips_file),
            "dua_log": str(dua_log),
            "additional_file": str(additional_file),
            "edge_flows_xml": str(edge_flows_xml),
            "edge_flows_csv": str(edge_flows_csv),
            "turning_file": str(turning_file),
            "artifacts_file": str(artifacts_file),
        }
    )
    artifacts.diagnostics.update(
        {
            "demand_total_raw_trips": demand_metadata["total_raw_trips"],
            "demand_total_rounded_trips": demand_metadata["total_rounded_trips"],
            "demand_positive_od_pairs": demand_metadata["positive_od_pairs"],
        }
    )

    t0 = time.time()

    # ---- 步骤 1: od2trips — 将 OD 需求分解为个体出行 ----
    _run_command(
        [
            artifacts.tool_paths["od2trips"],
            "--taz-files",
            str(mappings.taz_file),          # TAZ 定义文件
            "--tazrelation-files",
            str(od_file),                     # OD 需求文件
            "--output-file",
            str(trips_file),                  # 输出: 个体出行
            "--begin",
            str(sumo_config.begin),
            "--end",
            str(sumo_config.end),
            "--spread.uniform",               # 出发时间均匀分布
            "true",
            "--prefix",
            f"{vehicle_type}_",               # 车辆 ID 前缀 (如 "car_0001")
            "--seed",
            str(sumo_config.seed),
            "--ignore-vehicle-type",          # 不写 vtype, 避免 duarouter 报错
        ],
        cwd=vehicle_dir,
    )

    t1 = time.time()

    # ---- 步骤 2: duaIterate.py — 动态用户均衡迭代 ----
    _run_command(
        [
            artifacts.tool_paths["python"],
            artifacts.tool_paths["duaIterate.py"],
            "-n",
            artifacts.files["net_file"],      # 路网文件
            "-t",
            str(trips_file),                  # 个体出行文件
            "-l",
            str(sumo_config.due_iterations),  # 迭代次数
            "--aggregation",
            str(sumo_config.aggregation_freq),  # 流量统计聚合频率 (秒)
            "-A",                               # Gawron alpha 参数
            str(sumo_config.gawron_a),
            "-B",                               # Gawron beta 参数
            str(sumo_config.gawron_beta),
            "--continue-on-unbuild",              # 跳过无法路由的车辆继续迭代
            "sumo--seed",                         # seed 通过 remaining_args 传给 sumo
            str(sumo_config.seed),
            "duarouter--repair",                  # 尝试修复不可达的起终点
            "true",
        ],
        cwd=vehicle_dir,
        log_path=dua_log,
    )
    # 解析 DUA 日志中的 gap 收敛历史
    artifacts.gap_history = _parse_gap_history(dua_log)
    if artifacts.gap_history and artifacts.gap_history[-1] > 0.5:
        logger.warning(
            "Posterior SUMO assignment for %s ended with a relatively large final gap %.4f",
            vehicle_type,
            artifacts.gap_history[-1],
        )

    # 从 DUA 输出中找到最新的路线文件并复制为最终路线
    dua_route_file = _find_latest_dua_route_file(vehicle_dir, trips_file)
    shutil.copyfile(dua_route_file, final_route_file)
    artifacts.files["final_route_file"] = str(final_route_file)

    t2 = time.time()

    # ---- 步骤 3: sumo — 用最终路线运行一次完整仿真 (收集 edge 流量) ----
    _write_edge_data_additional(additional_file, edge_flows_xml, sumo_config.aggregation_freq)
    _run_command(
        [
            artifacts.tool_paths["sumo"],
            "--net-file",
            artifacts.files["net_file"],
            "--route-files",
            str(final_route_file),
            "--additional-files",
            str(additional_file),              # 启用 edge 流量输出
            "--begin",
            str(sumo_config.begin),
            "--end",
            str(sumo_config.end),
            "--seed",
            str(sumo_config.seed),
            "--no-warnings",
        ],
        cwd=vehicle_dir,
        log_path=sumo_log,
    )

    t3 = time.time()

    # ---- 步骤 4: 后处理 ----
    # 解析 edge 流量 XML → CSV
    _parse_edge_flows_to_csv(edge_flows_xml, edge_flows_csv)
    # 统计 gate 级转向流量
    turning_payload = _build_turning_output(
        route_file=final_route_file,
        gate_mappings=mappings.gate_mappings,
        vehicle_type=vehicle_type,
        period_name=sumo_config.period_name,
    )
    with open(turning_file, "w", encoding="utf-8") as handle:
        json.dump(turning_payload, handle, indent=2, ensure_ascii=False)
    # 计算路径诊断统计 (平均路径长度、多路径OD对数等)
    sumolib = _import_sumolib(artifacts.tool_paths["tools_dir"])
    net = sumolib.net.readNet(artifacts.files["net_file"], withInternal=True)
    edge_records, _ = _load_sumo_edge_records(net, vehicle_type)
    artifacts.diagnostics.update(
        _build_route_diagnostics(
            route_file=final_route_file,
            trips_file=trips_file,
            gate_mappings=mappings.gate_mappings,
            od_pairs=od_pairs,
            gap_history=artifacts.gap_history,
            edge_records=edge_records,
        )
    )
    t4 = time.time()
    artifacts.diagnostics.update(
        {
            "timing_od2trips_sec": round(t1 - t0, 2),
            "timing_dua_iterate_sec": round(t2 - t1, 2),
            "timing_sumo_final_sec": round(t3 - t2, 2),
            "timing_postprocess_sec": round(t4 - t3, 2),
            "timing_simulation_total_sec": round(t4 - t0, 2),
        }
    )
    logger.info(
        "SUMO simulation timing for %s: od2trips=%.1fs, DUA=%.1fs, sumo=%.1fs, post=%.1fs, total=%.1fs",
        vehicle_type,
        t1 - t0,
        t2 - t1,
        t3 - t2,
        t4 - t3,
        t4 - t0,
    )
    # 写出完整 artifacts 记录
    with open(artifacts_file, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "vehicle_type": artifacts.vehicle_type,
                "period_name": artifacts.period_name,
                "sumo_version": artifacts.sumo_version,
                "tool_paths": artifacts.tool_paths,
                "files": artifacts.files,
                "gap_history": artifacts.gap_history,
                "diagnostics": artifacts.diagnostics,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    return {
        "od_file": str(od_file),
        "demand_metadata_file": str(demand_metadata_file),
        "trips_file": str(trips_file),
        "final_route_file": str(final_route_file),
        "edge_flows_xml": str(edge_flows_xml),
        "edge_flows_csv": str(edge_flows_csv),
        "turning_file": str(turning_file),
        "artifacts_file": str(artifacts_file),
    }


# ============================================================================
# SUMO XML 文件生成与解析
# ============================================================================


def _write_tazrelation_od(
    path: Path,
    zone_mappings: dict[int, SumoZoneMapping],
    begin: int,
    end: int,
    demand_entries: list[dict[str, Any]],
) -> Path:
    """生成 SUMO OD 需求文件 (tazRelation 格式)。

    输出格式:
      <data>
        <interval id="0" begin="0" end="3600">
          <tazRelation from="taz_1" to="taz_2" count="15"/>
          <tazRelation from="taz_1" to="taz_3" count="8"/>
          ...
        </interval>
      </data>

    SUMO 的 od2trips 工具读取此文件, 为每个 OD 对生成 count 辆车的 trip。
    四舍五入后 count=0 的 OD 对会被跳过。
    """
    root = ET.Element("data")
    interval = ET.SubElement(
        root,
        "interval",
        {
            "id": "0",
            "begin": str(begin),
            "end": str(end),
        },
    )
    for entry in demand_entries:
        if int(entry["rounded_count"]) <= 0:
            continue  # 跳过零出行的 OD 对
        origin = int(entry["origin"])
        dest = int(entry["dest"])
        ET.SubElement(
            interval,
            "tazRelation",
            {
                "from": zone_mappings[origin].taz_id,
                "to": zone_mappings[dest].taz_id,
                "count": str(int(entry["rounded_count"])),
            },
        )
    tree = ET.ElementTree(root)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def _find_latest_dua_route_file(vehicle_dir: Path, trips_file: Path) -> Path:
    """从 DUA 迭代输出中找到最新的路线文件。

    duaIterate.py 每次迭代产出一个 .rou.xml 文件 (如 xxx_000.rou.xml, xxx_001.rou.xml, ...),
    最后一个迭代的文件即为收敛后的最终路线。
    通过文件修改时间排序找到最新的那个, 排除 trips 文件、alt 文件和已有的 final 文件。
    """
    route_files = [
        path
        for path in vehicle_dir.rglob("*.rou.xml")
        if path != trips_file
        and not path.name.endswith(".rou.alt.xml")  # 排除备选路线文件
        and "routes_final" not in path.name         # 排除已有的 final 文件
    ]
    if not route_files:
        raise RuntimeError(f"Could not locate the final route file produced by duaIterate under {vehicle_dir}")
    return max(route_files, key=lambda path: path.stat().st_mtime)


def _parse_gap_history(log_path: Path) -> list[float]:
    """从 DUA 迭代日志中提取 gap 收敛历史。

    DUA 每次迭代会输出 "gap" 或 "relative gap" 值,
    该值越小说明路径选择越接近均衡状态。
    理想情况下 gap 应逐步下降并趋近于 0。
    """
    if not log_path.exists():
        return []
    gap_values: list[float] = []
    # 匹配日志中 "gap 0.1234" 或 "relative gap 0.0567" 等模式
    pattern = re.compile(r"(?i)(?:relative\s+gap|gap)\D+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                try:
                    gap_values.append(float(match.group(1)))
                except ValueError:
                    continue
    return gap_values


def _write_edge_data_additional(path: Path, edge_flows_xml: Path, aggregation_freq: int) -> None:
    """生成 SUMO additional 配置文件, 启用 edge 级流量输出。

    该配置告诉 SUMO 在仿真过程中每隔 aggregation_freq 秒统计一次每条 edge 的通过车辆数,
    结果写入 edge_flows_xml 文件。
    """
    root = ET.Element("additional")
    ET.SubElement(
        root,
        "edgeData",
        {
            "id": "assignment_matrix",
            "file": str(edge_flows_xml),
            "freq": str(aggregation_freq),
        },
    )
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _parse_edge_flows_to_csv(edge_flows_xml: Path, edge_flows_csv: Path) -> None:
    """将 SUMO 输出的 edge 流量 XML 解析为更易读的 CSV 格式。

    输入 XML 结构: <meandata><interval begin="0"><edge id="xxx" entered="15"/>...
    输出 CSV 列: interval_begin, edge_id, entered (进入该 edge 的车辆数)
    """
    root = ET.parse(edge_flows_xml).getroot()
    rows: list[tuple[str, str, str]] = []
    for interval in root.findall("interval"):
        begin = interval.get("begin", "")
        for edge in interval.findall("edge"):
            rows.append((begin, edge.get("id", ""), edge.get("entered", "0")))
    with open(edge_flows_csv, "w", encoding="utf-8") as handle:
        handle.write("interval_begin,edge_id,entered\n")
        for begin, edge_id, entered in rows:
            handle.write(f"{begin},{edge_id},{entered}\n")


# ============================================================================
# 路线解析与 H 矩阵构建
# ============================================================================


def _parse_trip_od_map(trips_file: Path) -> dict[str, tuple[int, int]]:
    """从 trips.rou.xml 中提取每辆车的 OD 信息。

    trips 文件中每个 <trip> 元素包含:
      - id: 车辆 ID (如 "car_0001")
      - fromTaz: 起始 TAZ (如 "taz_1")
      - toTaz: 目的 TAZ (如 "taz_5")

    Returns:
        {vehicle_id: (origin_zone_id, dest_zone_id)} 字典
    """
    root = ET.parse(trips_file).getroot()
    result: dict[str, tuple[int, int]] = {}
    for trip in root.findall("trip"):
        trip_id = trip.get("id")
        from_taz = trip.get("fromTaz")
        to_taz = trip.get("toTaz")
        if not trip_id or not from_taz or not to_taz:
            continue
        try:
            # "taz_1" → 1, "taz_23" → 23
            result[trip_id] = (int(from_taz.replace("taz_", "")), int(to_taz.replace("taz_", "")))
        except ValueError:
            continue
    return result


def _parse_route_edges(route_file: Path) -> dict[str, list[str]]:
    """从路线文件中提取每辆车的路线 (经过的 edge 序列)。

    SUMO 路线文件有两种格式:
      格式 A — 独立 route 定义 + vehicle 引用:
        <route id="route_0" edges="e1 e2 e3"/>
        <vehicle id="car_0001" route="route_0"/>

      格式 B — 内嵌 route:
        <vehicle id="car_0001">
          <route edges="e1 e2 e3"/>
        </vehicle>

    本函数两种格式都支持。

    Returns:
        {vehicle_id: [edge_id_1, edge_id_2, ...]} 字典
    """
    root = ET.parse(route_file).getroot()
    # 先解析独立定义的 route (格式 A)
    route_defs: dict[str, list[str]] = {}
    vehicle_routes: dict[str, list[str]] = {}
    for route in root.findall("route"):
        route_id = route.get("id")
        edges = route.get("edges")
        if route_id and edges:
            route_defs[route_id] = edges.split()
    # 再解析每辆车的路线
    for vehicle in root.findall("vehicle"):
        vehicle_id = vehicle.get("id")
        if not vehicle_id:
            continue
        route_ref = vehicle.get("route")
        route_edges = None
        if route_ref and route_ref in route_defs:
            # 格式 A: 通过 route 属性引用
            route_edges = route_defs[route_ref]
        else:
            # 格式 B: 内嵌 <route> 子元素
            route_elem = vehicle.find("route")
            if route_elem is not None and route_elem.get("edges"):
                route_edges = route_elem.get("edges", "").split()
        if route_edges is not None:
            vehicle_routes[vehicle_id] = route_edges
    return vehicle_routes


def _build_h_from_sumo_routes(
    trips_file: Path,
    route_file: Path,
    gate_mappings: dict[int, SumoGateMapping],
    od_pairs: list[tuple[int, int]],
    n_gates: int,
) -> sp.csr_matrix:
    """从 SUMO 仿真的路线结果中反向构建分配矩阵 H。

    这是 SUMO 后端最核心的输出。H 矩阵的含义:
      H[g, j] = 在 OD 对 j 的所有车辆中, 经过 gate g 对应 edge 的比例

    构建过程:
      1. 从 trips 文件获取每辆车的 OD 对
      2. 从 route 文件获取每辆车的路线 (edge 序列)
      3. 遍历每辆车: 对其所属 OD 对的总数 +1, 同时检查其路线是否经过每个 gate 的 from_edge
      4. 归一化: gate_counts / od_totals → H (每列归一化, 使 H[g,j] 表示概率)

    与解析式 H 的对比:
      解析式 H 基于 K-shortest paths + Logit 模型计算路径概率;
      SUMO H 基于微观仿真的均衡路径, 考虑了拥堵和动态路径选择。
      两者的差异可用于评估解析模型的准确性。

    Args:
        trips_file: od2trips 输出的 trips 文件 (用于获取车辆→OD 映射)
        route_file: DUA 最终路线文件 (用于获取车辆的 edge 序列)
        gate_mappings: gate→edge 映射
        od_pairs: OD 对列表 (确定 H 矩阵的列顺序)
        n_gates: gate 总数 (确定 H 矩阵的行数)

    Returns:
        稀疏 CSR 格式的 H 矩阵, 形状 (n_gates, n_od_pairs)
    """
    vehicle_to_od = _parse_trip_od_map(trips_file)
    vehicle_routes = _parse_route_edges(route_file)
    od_to_col = {pair: idx for idx, pair in enumerate(od_pairs)}  # OD对 → 列索引
    od_totals = np.zeros(len(od_pairs), dtype=float)              # 每个OD对的总车辆数
    gate_counts = np.zeros((n_gates, len(od_pairs)), dtype=float) # 经过每个gate的车辆数 (按OD对分)
    gate_to_from_edge = {gate_idx: mapping.from_edge_id for gate_idx, mapping in gate_mappings.items()}

    # 遍历每辆车, 统计其对各 gate 的贡献
    for vehicle_id, route_edges in vehicle_routes.items():
        od_pair = vehicle_to_od.get(vehicle_id)
        if od_pair is None or od_pair not in od_to_col:
            continue
        col = od_to_col[od_pair]
        od_totals[col] += 1.0  # 该 OD 对的总车辆数 +1
        route_edge_set = set(route_edges)  # 转为 set 加速查询
        # 检查这辆车的路线是否经过每个 gate 的 from_edge
        for gate_idx, from_edge_id in gate_to_from_edge.items():
            if from_edge_id and from_edge_id in route_edge_set:
                gate_counts[gate_idx, col] += 1.0

    # 归一化: H[g,j] = 经过 gate g 的该 OD 对车辆数 / 该 OD 对的总车辆数
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(
            gate_counts,
            od_totals[np.newaxis, :],
            out=np.zeros_like(gate_counts),
            where=od_totals[np.newaxis, :] > 0,  # 避免除以零
        )
    return sp.csr_matrix(normalized)


# ============================================================================
# 转向流量统计与路径诊断
# ============================================================================


def _build_turning_output(
    route_file: Path,
    gate_mappings: dict[int, SumoGateMapping],
    vehicle_type: str,
    period_name: str,
) -> dict[str, Any]:
    """从 SUMO 路线结果中统计每个 gate 的转向流量 (仿真侧)。

    "转向流量" 指从 from_edge 驶入、经 to_edge 驶出的车辆数。
    这是 SUMO 仿真的关键验证指标, 可与摄像头实际观测的转向计数对比。

    统计过程:
      1. 解析所有车辆的路线, 统计路线中相邻 edge 对的出现次数
         (忽略内部 edge, 即以 ":" 开头的交叉口连接)
      2. 对每个 gate, 查找其 (from_edge, to_edge) 对在所有路线中的出现次数
      3. 按摄像头分组, 计算每个 gate 在该摄像头内的流量占比

    Returns:
        包含 movements (逐 gate 转向计数) 和 cameras (按摄像头分组) 的字典
    """
    vehicle_routes = _parse_route_edges(route_file)
    # 统计所有路线中相邻 edge 对的出现次数
    pair_counts: dict[tuple[str, str], int] = {}
    for route_edges in vehicle_routes.values():
        # 过滤掉交叉口内部 edge (以 ":" 开头)
        external_edges = [edge_id for edge_id in route_edges if not edge_id.startswith(":")]
        for first, second in zip(external_edges, external_edges[1:]):
            pair_counts[(first, second)] = pair_counts.get((first, second), 0) + 1

    # 对每个 gate, 查找其转向对的流量
    movements = []
    camera_movements: dict[str, list[dict[str, Any]]] = {}
    matched = 0
    for gate_idx in sorted(gate_mappings):
        mapping = gate_mappings[gate_idx]
        # 只处理有完整转向映射的 gate
        if not mapping.from_edge_id or not mapping.to_edge_id:
            continue
        if mapping.from_edge_id == mapping.to_edge_id:
            continue  # 排除掉头 (from == to)
        count = pair_counts.get((mapping.from_edge_id, mapping.to_edge_id), 0)
        movement = {
            "gate_index": gate_idx,
            "camera_id": mapping.camera_id,
            "gate_id": mapping.gate_id,
            "from_edge_id": mapping.from_edge_id,
            "to_edge_id": mapping.to_edge_id,
            "count": count,
        }
        movements.append(movement)
        camera_movements.setdefault(mapping.camera_id, []).append(movement)
        matched += 1
    # 按摄像头分组, 计算各 gate 在该摄像头流量中的占比
    cameras = []
    for camera_id in sorted(camera_movements):
        camera_items = camera_movements[camera_id]
        total_count = sum(int(item["count"]) for item in camera_items)
        movements_with_proportions = []
        for item in camera_items:
            movement = dict(item)
            movement["proportion"] = (
                float(item["count"]) / float(total_count) if total_count > 0 else 0.0
            )
            movements_with_proportions.append(movement)
        cameras.append(
            {
                "camera_id": camera_id,
                "n_movements": len(movements_with_proportions),
                "total_count": total_count,
                "movements": movements_with_proportions,
            }
        )
    return {
        "vehicle_type": vehicle_type,
        "period_name": period_name,
        "matched_movements": matched,
        "movements": movements,
        "cameras": cameras,
    }


def build_observed_turning_sumo(
    gate_mappings: dict[int, SumoGateMapping],
    raw_obs: dict[str, dict[int, list[float]]],
    vehicle_type: str,
) -> dict[str, Any]:
    """将观测侧原始 gate 计数映射到 SUMO edge 体系，产出与 _build_turning_output 格式一致的转向数据。

    每个 gate 在摄像头配置中定义了 origin_direction 和 dest_direction，
    ensure_sumo_assets() 已将它们映射为 SUMO 的 (from_edge_id, to_edge_id)。
    本函数将原始观测多时段计数取均值，关联到该映射上，输出 gate 级别的转向计数。

    参数:
        gate_mappings: ensure_sumo_assets() 返回的 SumoNativeMappings.gate_mappings
        raw_obs: parse_observations() 返回的原始数据 {vehicle_type: {gate_index: [counts]}}
        vehicle_type: 车型

    返回:
        与 _build_turning_output() 格式一致的 dict，可直接和 SUMO 仿真转向数据对比
    """
    vehicle_data = raw_obs.get(vehicle_type, {})

    movements: list[dict[str, Any]] = []
    camera_movements: dict[str, list[dict[str, Any]]] = {}
    matched = 0

    for gate_idx in sorted(gate_mappings):
        mapping = gate_mappings[gate_idx]
        if not mapping.from_edge_id or not mapping.to_edge_id:
            continue
        if mapping.from_edge_id == mapping.to_edge_id:
            continue

        gate_counts = vehicle_data.get(gate_idx, [])
        count = float(np.mean(gate_counts)) if gate_counts else 0.0

        movement: dict[str, Any] = {
            "gate_index": gate_idx,
            "camera_id": mapping.camera_id,
            "gate_id": mapping.gate_id,
            "from_edge_id": mapping.from_edge_id,
            "to_edge_id": mapping.to_edge_id,
            "count": count,
        }
        movements.append(movement)
        camera_movements.setdefault(mapping.camera_id, []).append(movement)
        matched += 1

    cameras: list[dict[str, Any]] = []
    for camera_id in sorted(camera_movements):
        camera_items = camera_movements[camera_id]
        total_count = sum(item["count"] for item in camera_items)
        movements_with_proportions = []
        for item in camera_items:
            movement = dict(item)
            movement["proportion"] = (
                item["count"] / total_count if total_count > 0 else 0.0
            )
            movements_with_proportions.append(movement)
        cameras.append(
            {
                "camera_id": camera_id,
                "n_movements": len(movements_with_proportions),
                "total_count": total_count,
                "movements": movements_with_proportions,
            }
        )

    return {
        "vehicle_type": vehicle_type,
        "matched_movements": matched,
        "movements": movements,
        "cameras": cameras,
    }


def _build_route_diagnostics(
    route_file: Path,
    trips_file: Path,
    gate_mappings: dict[int, SumoGateMapping],
    od_pairs: list[tuple[int, int]],
    gap_history: list[float],
    edge_records: list[_SumoEdgeRecord],
) -> dict[str, Any]:
    """计算仿真路径的诊断统计信息, 用于评估仿真质量。

    统计指标:
      - mean_path_edge_count: 平均路径经过的 edge 数 (衡量路径长度)
      - mean_path_travel_time: 平均路径旅行时间 (edge长度/限速 的累加, 仅为粗略估算)
      - max_path_edge_count: 最长路径的 edge 数 (异常长路径可能暗示路网问题)
      - multi_path_od_pairs: 有多条不同路径的 OD 对数 (DUA 均衡的效果指标)
      - single_path_od_pairs: 只有一条路径的 OD 对数
      - gap_history_count: DUA 迭代次数
      - turning_candidate_count: 有完整转向映射的 gate 数
    """
    vehicle_routes = _parse_route_edges(route_file)
    vehicle_to_od = _parse_trip_od_map(trips_file)
    # 收集每个 OD 对的不同路径签名 (用于判断多路径多样性)
    route_signatures_per_od: dict[tuple[int, int], set[tuple[str, ...]]] = {pair: set() for pair in od_pairs}
    path_lengths = []
    # 预计算每条 edge 的自由流旅行时间 = 长度 / 限速
    edge_time_by_id = {
        edge.edge_id: (edge.length / edge.speed if edge.length > 0.0 and edge.speed > 0.0 else 0.0)
        for edge in edge_records
    }
    path_times = []
    for vehicle_id, route_edges in vehicle_routes.items():
        od_pair = vehicle_to_od.get(vehicle_id)
        if od_pair in route_signatures_per_od:
            route_signatures_per_od[od_pair].add(tuple(route_edges))
        external_edges = [edge for edge in route_edges if not edge.startswith(":")]
        path_lengths.append(len(external_edges))
        path_times.append(sum(edge_time_by_id.get(edge, 0.0) for edge in external_edges))
    # 有多条不同路径的 OD 对数 (DUA 均衡使不同车辆选择不同路径)
    multi_path_ods = sum(1 for routes in route_signatures_per_od.values() if len(routes) > 1)
    # 有完整转向映射 (from_edge + to_edge) 的 gate 数
    turning_candidates = sum(
        1
        for mapping in gate_mappings.values()
        if mapping.from_edge_id is not None and mapping.to_edge_id is not None and mapping.from_edge_id != mapping.to_edge_id
    )
    return {
        "mean_path_edge_count": float(np.mean(path_lengths)) if path_lengths else 0.0,
        "mean_path_travel_time": float(np.mean(path_times)) if path_times else 0.0,
        "max_path_edge_count": int(max(path_lengths)) if path_lengths else 0,
        "multi_path_od_pairs": multi_path_ods,
        "single_path_od_pairs": len(od_pairs) - multi_path_ods,
        "gap_history_count": len(gap_history),
        "turning_candidate_count": turning_candidates,
    }
