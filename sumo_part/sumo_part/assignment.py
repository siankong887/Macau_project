"""
assignment.py — 分配矩阵构建模块

本模块的核心任务是构建 **分配矩阵 H**（allocation matrix），它描述了
"每个 OD 对的交通流量有多大概率经过每个观测 gate"。

H 矩阵是贝叶斯 OD 估计的关键桥梁：
    观测模型:  g = H · x + ε
    其中:
      - g: 各 gate 的观测流量向量 (n_gates,)
      - H: 分配矩阵 (n_gates × n_od)，本模块构建的核心产物
      - x: OD 流量向量 (n_od,)，待估计的目标
      - ε: 观测噪声

构建 H 的流程：
  1. 加载摄像头配置 → 获取 gate 列表及其方向信息
  2. 将 gate 映射到路网边 (edge) → 通过方位角匹配
  3. 枚举所有 OD 对 (origin-destination pairs)
  4. 对每个 OD 对，计算 K 条最短路径
  5. 用 Logit 模型计算每条路径的选择概率
  6. 如果路径经过某个 gate 所在的 edge，将概率累加到 H 矩阵对应位置

数据流向：
  road network (GraphML) + camera config (JSON)
      ↓ load_camera_config()     — 解析摄像头/gate 配置
      ↓ map_gates_to_edges()     — 将 gate 映射到最近的路网边
      ↓ build_od_pairs()         — 生成 23×22=506 个 OD 对
      ↓ compute_k_shortest_paths — 对每个 OD 对求 K 条最短路径
      ↓ logit_probabilities()    — Logit 路径选择模型分配概率
      → build_assignment_matrix  — 组装稀疏矩阵 H (n_gates × n_od)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from itertools import islice

import networkx as nx
import numpy as np
import scipy.sparse as sp

from .paths import ProjectPaths, default_paths
from .types import GateInfo, PathCache

logger = logging.getLogger(__name__)

# ============================================================================
# 默认参数
# ============================================================================

DEFAULT_K_PATHS = 3        # 每个 OD 对默认计算的候选最短路径数量
DEFAULT_LOGIT_THETA = 0.1  # Logit 模型的温度参数 (越小 → 概率越均匀；越大 → 越集中于最短路)

# ============================================================================
# 方向与方位角 (Bearing) 工具
# ============================================================================
# 摄像头配置文件中每个 gate 有 origin_direction / dest_direction 字段，
# 用文字描述车辆来自哪个方向（如 "north", "southwest"）。
# 这里将方向文字转成数值方位角 (0°=北, 顺时针增加)，
# 以便后续与路网边的几何方位角做匹配，判断 gate 对应哪条路网边。

# 八个基本方向对应的方位角（度），以正北为 0°，顺时针递增
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

# 拼写错误修正映射表：配置数据中可能存在的方向拼写错误 → 正确写法
DIRECTION_ALIASES = {
    "nortwest": "northwest",
    "northweast": "northeast",
    "southest": "southeast",
    "souhtwest": "southwest",
    "nort": "north",
    "weast": "east",
}


def direction_to_bearing(direction: str) -> float | None:
    """
    将方向文字转换为方位角（度）。

    处理三种情况：
    1. 单个方向词：如 "north" → 0.0, "southwest" → 225.0
    2. 组合方向：如 "north/east" → 取两个方位角的圆周均值 → 45.0
    3. 无法识别：返回 None

    先做拼写纠错（DIRECTION_ALIASES），再查找方位角。

    参数:
        direction: 方向文字，如 "north", "southwest", "north/east"
    返回:
        方位角（度），0~360，或 None
    """
    direction = direction.strip().lower()
    # 先修正拼写错误
    direction = DIRECTION_ALIASES.get(direction, direction)
    # 情况 1：直接匹配单个方向
    if direction in DIRECTION_BEARINGS:
        return float(DIRECTION_BEARINGS[direction])
    # 情况 2：组合方向，如 "north/east"，取圆周均值
    if "/" in direction:
        parts = []
        for part in direction.split("/"):
            part = DIRECTION_ALIASES.get(part.strip(), part.strip())
            if part in DIRECTION_BEARINGS:
                parts.append(float(DIRECTION_BEARINGS[part]))
        if parts:
            return _circular_mean(parts)
    # 情况 3：无法识别
    return None


def _circular_mean(angles_deg: list[float]) -> float:
    """
    计算角度的圆周均值（circular mean）。

    普通算术均值对角度不适用，例如 avg(350°, 10°) = 180°，显然不对。
    圆周均值的做法是：将角度转为单位向量，求向量和的方向。

    例如: _circular_mean([350, 10]) → 0.0  (正北方向，符合直觉)

    参数:
        angles_deg: 角度列表（度）
    返回:
        圆周均值角度（度），范围 [0, 360)
    """
    rads = [math.radians(angle) for angle in angles_deg]
    sin_sum = sum(math.sin(angle) for angle in rads)
    cos_sum = sum(math.cos(angle) for angle in rads)
    return math.degrees(math.atan2(sin_sum, cos_sum)) % 360


def _angular_diff(a: float, b: float) -> float:
    """
    计算两个方位角之间的最小夹角（度），范围 [0, 180]。

    处理了跨越 0°/360° 的情况，例如:
      _angular_diff(10, 350) → 20  (不是 340)
      _angular_diff(90, 270) → 180

    参数:
        a, b: 两个方位角（度）
    返回:
        最小夹角（度），范围 [0, 180]
    """
    diff = abs(a - b) % 360
    return min(diff, 360 - diff)


def _edge_bearing(G: nx.MultiDiGraph, u: str, v: str) -> float:
    """
    计算路网中一条边 (u → v) 的地理方位角。

    使用节点的 x (经度) 和 y (纬度) 坐标，通过 atan2 计算从 u 到 v 的方位角。
    注意：atan2(dx, dy) 而非 atan2(dy, dx)，是因为方位角以正北 (y轴正方向) 为 0°，
    顺时针增加，所以 x 对应 sin 分量，y 对应 cos 分量。

    参数:
        G: 路网图
        u, v: 边的起点和终点节点 ID
    返回:
        方位角（度），范围 [0, 360)
    """
    ux, uy = float(G.nodes[u]["x"]), float(G.nodes[u]["y"])
    vx, vy = float(G.nodes[v]["x"]), float(G.nodes[v]["y"])
    dx = vx - ux  # 经度差 → 东西方向
    dy = vy - uy  # 纬度差 → 南北方向
    return math.degrees(math.atan2(dx, dy)) % 360


# ============================================================================
# 摄像头配置加载与 Gate 到路网边的映射
# ============================================================================


def load_camera_config(config_path: Path) -> list[GateInfo]:
    """
    从摄像头配置 JSON 文件加载所有 gate 信息。

    配置文件 (a1_copy_2.json) 结构示例：
    {
      "list": [
        {
          "camera": "cam_a1",
          "GPS": [113.5435, 22.1987],         ← 摄像头经纬度
          "gate": [
            {
              "gate_id": "gate_1",
              "origin_direction": "north",     ← 车辆来自哪个方向
              "dest_direction": "south",       ← 车辆去往哪个方向
              "origin_road_name": "友谊大馬路",
              "dest_road_name": "罅些喇提督大馬路"
            },
            { "gate_id": "gate_2", ... }
          ]
        },
        ...
      ]
    }

    每个 gate 被分配一个全局递增索引 (gate_index)，这个索引在整个系统中
    唯一标识一个 gate，也对应 H 矩阵的行索引。

    参数:
        config_path: 摄像头配置 JSON 文件路径
    返回:
        所有 gate 的 GateInfo 列表，按全局索引排列
    """
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    gates: list[GateInfo] = []
    idx = 0  # 全局 gate 索引计数器
    for camera in config["list"]:
        camera_id = camera["camera"]
        lon, lat = camera["GPS"]
        # 一个摄像头可能有多个 gate（多个监控车道/方向）
        for gate in camera["gate"]:
            gates.append(
                GateInfo(
                    camera_id=camera_id,
                    gate_id=gate["gate_id"],
                    gate_index=idx,               # 全局唯一索引
                    camera_lon=float(lon),
                    camera_lat=float(lat),
                    origin_direction=gate.get("origin_direction", ""),
                    dest_direction=gate.get("dest_direction", ""),
                    origin_road=gate.get("origin_road_name", ""),
                    dest_road=gate.get("dest_road_name", ""),
                )
            )
            idx += 1
    return gates
"""
gates 是一个列表，每一个元素对应一个gate的信息，GateInfo类用了@dataclass修饰符，打印列表的元素会
输出：
 GateInfo(
                    camera_id=camera_id,
                    gate_id=gate["gate_id"],
                    gate_index=idx,               # 全局唯一索引
                    camera_lon=float(lon),
                    camera_lat=float(lat),
                    origin_direction=gate.get("origin_direction", ""),
                    dest_direction=gate.get("dest_direction", ""),
                    origin_road=gate.get("origin_road_name", ""),
                    dest_road=gate.get("dest_road_name", ""),
                )
访问元素也非常方便，通过索引访问列表中的单个元素，然后通过"."的方式访问具体的数据
gates[0].camera_id      # 第一个 gate 的摄像头 ID
gates[3].edge           # 第四个 gate 映射到的路网边
gates[i].gate_index     # 第 i 个 gate 的全局索引

"""


def _nearest_graph_node(
    G: nx.MultiDiGraph,
    lon: float,
    lat: float,
    node_cache: list[tuple[str, float, float]],
) -> str:
    """
    找到路网中距离给定经纬度最近的节点。

    使用欧氏距离的平方（省略开方，不影响排序）来比较距离。
    注意：这里直接用经纬度做欧氏距离是简化处理，严格来说经度和纬度
    的尺度不同，但在澳门这样的小区域内误差可忽略。

    参数:
        G: 路网图（未直接使用，保留接口一致性）
        lon, lat: 目标点的经度和纬度
        node_cache: 预构建的节点坐标缓存 [(node_id, lon, lat), ...]
    返回:
        最近节点的 ID
    """
    best_node = None
    best_dist = float("inf")
    for node, node_lon, node_lat in node_cache:
        # 使用距离平方比较，省略开方运算
        dist = (node_lon - lon) ** 2 + (node_lat - lat) ** 2
        if dist < best_dist:
            best_dist = dist
            best_node = node
    if best_node is None:
        raise RuntimeError("No graph node found")
    return best_node


def map_gates_to_edges(G: nx.MultiDiGraph, gates: list[GateInfo]) -> list[GateInfo]:
    """
    将每个 gate 映射到路网中最匹配的边 (edge)。

    映射逻辑：
    1. 对每个摄像头，找到路网中距离最近的节点
    2. 收集该节点所有相连的边（出边 + 入边）作为候选
    3. 对摄像头下的每个 gate：
       a. 将 gate 的 origin_direction（车辆来源方向）转换为方位角
       b. 加 180° 得到车辆的「接近方位角」（approach_bearing）
          例如：车从北方来 (0°) → 接近方向是从北向南 → 接近方位角 = 180°
       c. 在候选边中找方位角最接近的边
       d. 如果最小夹角 < 90°，认为匹配成功；否则回退到默认边

    参数:
        G: 路网多重有向图
        gates: gate 列表（会被原地修改，写入 edge 属性）
    返回:
        修改后的 gates 列表，每个 gate 的 .edge 字段被赋值为 (u, v, key) 元组
    """
    # 按摄像头分组，同一摄像头下的 gate 共享同一个最近节点，通过集合实现去重
    cameras: dict[str, list[GateInfo]] = {}
    for gate in gates:
        cameras.setdefault(gate.camera_id, []).append(gate)
    """
    前面这三段的逻辑是先给cmaeras这个变量初始化一个空的字典
    循环访问gates这个列表，逐个取出包含gate信息的对象，
    最后一行，.setdefalut()的作用是首先访问一下cameras这个字典，有没有gate.camera_id这个key，如果没有就先初始化一个
    gate.camera_id:[],这样一个初始的key value对
    然后一个关键的点你要知道，不然你会感觉很奇怪，就是.setdefalut()最后都会返回value本身而不是值，命中了那个key，也会直接
    返回value本身，没有命中，创建后也会返回空列表
    """

    # 预构建节点坐标缓存，避免反复访问图属性
    node_cache = [(str(node), float(data["x"]), float(data["y"])) for node, data in G.nodes(data=True)]
    """
    node_cache 大概长这样：
    [
    ("12345", 113.5435, 22.1987),
    ("12346", 113.5512, 22.2001),
    ("12347", 113.5389, 22.1953),
    ...
]
    列表中的每一个元素都是(节点ID，经度x，纬度y)
    """
    mapped = 0    # 通过方位角成功匹配的计数
    fallback = 0  # 回退到默认边的计数

    for camera_id, camera_gates in cameras.items():
        # 用该摄像头第一个 gate 的坐标找最近的路网节点（同一摄像头坐标相同）
        #这里输入了一个gate，里面含有经纬度，返回一个最近点的node id
        node = _nearest_graph_node(G, camera_gates[0].camera_lon, camera_gates[0].camera_lat, node_cache)

        # 收集该节点的所有相连边（出边 + 入边），连带计算每条边的方位角
        candidate_edges: list[tuple[str, str, str, float]] = []
        """
        G → MultiDiGraph 实例（图对象）
        G.edges → OutMultiEdgeView 实例（视图对象）
        G.edges即可以是属性，也可以也可以加括号变成一个方法
        G.edges 本身是一个 视图对象（OutMultiEdgeView），既可以当属性用，也可以像方法一样传参调用：
        # 当属性用 — 返回图中所有边
        G.edges

        # 像方法一样调用 — 传入节点，只返回该节点的出边
        G.edges("A", keys=True)
        这是因为 NetworkX 在这个视图对象上实现了 __call__() 方法，
        所以加括号传参时实际上是调用了 OutMultiEdgeView.__call__(node, keys=True)，内部做了筛选只返回指定节点的边。
        类似的，G.nodes 也是同样的设计，既是属性也可以当方法调用。
        """
        for u, v, key in G.edges(node, keys=True):       # 出边：node → v
            candidate_edges.append((str(u), str(v), str(key), _edge_bearing(G, str(u), str(v))))
        for u, v, key in G.in_edges(node, keys=True):    # 入边：u → node
            candidate_edges.append((str(u), str(v), str(key), _edge_bearing(G, str(u), str(v))))

        if not candidate_edges:
            logger.warning("Camera %s has no nearby candidate edges", camera_id)
            continue

        # 默认边：当方位角匹配失败时使用第一条候选边
        default_edge = candidate_edges[0][:3]

        for gate in camera_gates:
            # 将 origin_direction（如 "north"）转换为方位角
            origin_bearing = direction_to_bearing(gate.origin_direction)
            if origin_bearing is None:
                # 方向信息无法解析，使用默认边
                gate.edge = default_edge
                fallback += 1
                continue

            # 关键转换：origin_direction 是车辆「来自」的方向，
            # 而我们需要的是车辆「行驶」的方向（即路网边的方向）。
            # 车从北方来 = 车向南行驶，所以加 180°。
            approach_bearing = (origin_bearing + 180.0) % 360.0

            # 在候选边中找方位角最接近 approach_bearing 的边
            best = None
            best_diff = 360.0
            for u, v, key, bearing in candidate_edges:
                diff = _angular_diff(bearing, approach_bearing)
                if diff < best_diff:
                    best_diff = diff
                    best = (u, v, key)

            # 夹角 < 90° 视为匹配成功（大致同方向），否则回退
            if best is not None and best_diff < 90.0:
                gate.edge = best
                mapped += 1
            else:
                gate.edge = default_edge
                fallback += 1

    logger.info(
        "Mapped %d gates by bearing and %d by fallback",
        mapped,
        fallback,
    )
    return gates


def _serialize_gate_mapping(gates: list[GateInfo]) -> list[dict[str, object]]:
    """
    将 gate-edge 映射关系序列化为可 JSON 存储的格式。
    用于缓存写入，避免重复执行耗时的方位角匹配。
    """
    rows = []
    for gate in gates:
        edge = None
        if gate.edge is not None:
            edge = [gate.edge[0], gate.edge[1], gate.edge[2]]  # tuple → list (JSON 兼容)
        rows.append(
            {
                "gate_index": gate.gate_index,
                "camera_id": gate.camera_id,
                "gate_id": gate.gate_id,
                "edge": edge,
            }
        )
    return rows


def _apply_gate_mapping_cache(gates: list[GateInfo], mapping_rows: list[dict[str, object]]) -> bool:
    """
    尝试从缓存中恢复 gate-edge 映射关系。

    会做一致性校验：gate 数量、camera_id、gate_id 必须完全一致，
    否则返回 False 表示缓存失效（可能配置文件已更改）。

    参数:
        gates: 当前加载的 gate 列表（会被原地修改 .edge 字段）
        mapping_rows: 缓存中读取的映射数据
    返回:
        True 表示缓存成功应用，False 表示缓存失效需要重新计算
    """
    if len(gates) != len(mapping_rows):
        return False
    mapping_by_index = {int(row["gate_index"]): row for row in mapping_rows}
    for gate in gates:
        row = mapping_by_index.get(gate.gate_index)
        if row is None:
            return False
        # 校验 camera_id 和 gate_id 一致，确保配置没变
        if row.get("camera_id") != gate.camera_id or row.get("gate_id") != gate.gate_id:
            return False
        edge = row.get("edge")
        gate.edge = tuple(edge) if edge else None  # list → tuple 还原
    return True


def prepare_gates(
    G: nx.MultiDiGraph,
    paths: ProjectPaths | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
) -> tuple[list[GateInfo], dict[int, tuple[str, str, str] | None]]:
    """
    准备 gate 数据：加载配置 → 映射到路网边 → 返回映射表。

    这是 gate 处理的主入口函数，带缓存机制：
    1. 先加载摄像头配置，得到 gate 列表
    2. 尝试读取缓存的 gate-edge 映射
    3. 缓存命中则直接使用，否则重新执行方位角匹配并写入缓存

    参数:
        G: 路网图
        paths: 项目路径配置
        use_cache: 是否尝试使用缓存
        write_cache: 缓存未命中时是否写入新缓存
    返回:
        (gates, gate_edge_mapping) 元组:
          - gates: GateInfo 列表，每个 gate 已有 .edge 属性
          - gate_edge_mapping: {gate全局索引: (u, v, key) 或 None}
    """
    paths = paths or default_paths()
    gates = load_camera_config(paths.camera_config_json)

    # 尝试从缓存恢复
    cache_applied = False
    if use_cache and paths.gate_edge_mapping_json.exists():
        with open(paths.gate_edge_mapping_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        cache_applied = _apply_gate_mapping_cache(gates, payload.get("gates", []))

    if not cache_applied:
        # 缓存未命中，执行完整的方位角匹配
        map_gates_to_edges(G, gates)
        if write_cache:
            paths.ensure_output_dir()
            with open(paths.gate_edge_mapping_json, "w", encoding="utf-8") as handle:
                json.dump({"gates": _serialize_gate_mapping(gates)}, handle, indent=2)

    # 构建简洁的映射字典：gate_index → edge
    gate_edge_mapping = {gate.gate_index: gate.edge for gate in gates}
    return gates, gate_edge_mapping


# ============================================================================
# OD 对生成与图工具函数
# ============================================================================


def build_od_pairs(zone_node_map: dict[int, str]) -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    """
    生成所有可能的 OD 对（排除 origin == dest 的情况）。

    对于 23 个 zone，生成 23 × 22 = 506 个 OD 对。
    每个 OD 对被赋予一个唯一索引，这个索引对应 H 矩阵的列索引和 OD 向量 x 的元素位置。

    排列顺序：按 zone_id 升序，先固定 origin 遍历所有 dest。
    例如 zones=[1,2,3] → [(1,2),(1,3),(2,1),(2,3),(3,1),(3,2)]

    参数:
        zone_node_map: {zone_id: 路网节点ID} 的映射
    返回:
        (od_pairs, od_to_idx) 元组:
          - od_pairs: OD 对列表 [(origin, dest), ...]
          - od_to_idx: OD 对到列索引的映射 {(origin, dest): idx}
    """
    zone_ids = sorted(zone_node_map.keys())
    od_pairs: list[tuple[int, int]] = []
    od_to_idx: dict[tuple[int, int], int] = {}
    idx = 0
    for origin in zone_ids:
        for dest in zone_ids:
            if origin == dest:
                continue  # 排除自身到自身的 OD 对
            od_pairs.append((origin, dest))
            od_to_idx[(origin, dest)] = idx
            idx += 1
    return od_pairs, od_to_idx

"""
这个函数的主要作用是生成od对，以及建立一个索引，一个od对对应唯一一个整数
od_pairs 这个列表中的元组，第一个元素是(1,2),往后的都是类似的，对应一个起点一个终点
之所以od对是23 \times 22个，是因为每一个质心点对应的那个节点，都要去掉和自己的od组合，一共去掉23个
"""


def multi_to_digraph(G: nx.MultiDiGraph, weight: str = "travel_time") -> nx.DiGraph:
    """
    将多重有向图 (MultiDiGraph) 转换为简单有向图 (DiGraph)。

    NetworkX 的 shortest_simple_paths 不支持 MultiDiGraph（两点间可能有多条平行边），
    所以需要先把平行边合并：当 u→v 有多条边时，只保留权重（travel_time）最小的那条。

    参数:
        G: 路网多重有向图
        weight: 用于比较的权重属性名
    返回:
        简单有向图 D，每对 (u, v) 之间最多一条边（最快的那条）
    """
    D = nx.DiGraph()
    D.add_nodes_from(G.nodes(data=True))  # 保留所有节点及其属性
    for u, v, data in G.edges(data=True):
        w = float(data.get(weight, 1.0))
        if D.has_edge(u, v):
            # u→v 已有边，仅在新边更快时替换
            if w < float(D[u][v].get(weight, float("inf"))):
                D[u][v].update(data)
        else:
            D.add_edge(u, v, **data)
    return D


def compute_k_shortest_paths(
    G: nx.DiGraph,
    source: str,
    target: str,
    K: int = DEFAULT_K_PATHS,
    weight: str = "travel_time",
) -> list[tuple[list[str], float]]:
    """
    计算从 source 到 target 的 K 条最短简单路径。

    使用 NetworkX 的 Yen's K-shortest paths 算法（通过 shortest_simple_paths 实现）。
    "简单路径"意味着路径中不会重复经过同一节点。

    用 islice(paths, K) 只取前 K 条，避免枚举所有路径。
    如果实际可达路径不足 K 条，则返回所有可达路径。

    参数:
        G: 简单有向图
        source, target: 起终点节点 ID
        K: 最多返回的路径数量
        weight: 路径代价的边属性名
    返回:
        [(path_nodes, cost), ...] 列表，每个元素是 (节点列表, 路径总代价)
        如果不可达，返回空列表
    """
    if source == target:
        return []
    try:
        # shortest_simple_paths 返回一个惰性生成器，按代价从小到大产出
        paths = nx.shortest_simple_paths(G, source, target, weight=weight)
        result = []
        for path in islice(paths, K):  # 只取前 K 条
            # 手动计算路径总代价（沿路径各边权重求和）
            cost = 0.0
            for idx in range(len(path) - 1):
                cost += float(G[path[idx]][path[idx + 1]].get(weight, 1.0))
            result.append(([str(node) for node in path], float(cost)))
        return result
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def logit_probabilities(costs: list[float], theta: float = DEFAULT_LOGIT_THETA) -> np.ndarray:
    """
    用 Logit 模型将路径代价转换为选择概率。

    Logit 路径选择模型公式：
        P(路径 k) = exp(-θ · cost_k) / Σ_j exp(-θ · cost_j)

    直觉：
    - 代价越低的路径，概率越高
    - θ 越大，概率越集中在最短路径上（极端情况：θ → ∞ 时退化为全有全无）
    - θ 越小，概率越均匀（极端情况：θ → 0 时各路径等概率）
    - 默认 θ = 0.1 表示较温和的选择偏好

    数值稳定性处理：
      utilities -= max(utilities) 是经典的 log-sum-exp trick，
      防止 exp() 溢出，不影响最终的 softmax 概率。

    参数:
        costs: 各路径的代价列表
        theta: 温度参数（越大越确定性）
    返回:
        概率数组，长度与 costs 相同，元素之和为 1.0
    """
    if not costs:
        return np.array([])
    costs_array = np.array(costs, dtype=float)
    utilities = -theta * costs_array          # 代价越低 → 效用越高
    utilities -= np.max(utilities)             # 数值稳定性：减去最大值
    exp_u = np.exp(utilities)                  # 指数化
    return exp_u / np.sum(exp_u)               # 归一化为概率


# ============================================================================
# 核心：分配矩阵 H 的构建
# ============================================================================
#
# H 矩阵的含义：
#   H[g, od] = "OD 对 od 的单位流量经过 gate g 的概率"
#
# 例如 H[5, 42] = 0.7 表示：
#   第 42 个 OD 对的流量，有 70% 的概率经过第 5 个 gate。
#
# H 矩阵通常是非常稀疏的（大部分 OD 对不经过大部分 gate），
# 所以使用 scipy 稀疏矩阵存储以节省内存。


def build_edge_to_gates_mapping(gates: list[GateInfo]) -> dict[tuple[str, str], list[int]]:
    """
    构建「路网边 → gate 列表」的反向索引。

    一条路网边上可能安装了多个 gate（多个摄像头车道），
    这个映射让我们在遍历路径时快速查找：这条边上有没有 gate，有哪些。

    注意：edge_key 只用 (u, v) 不含 key，因为路径遍历时只知道 (u, v) 节点对。

    返回示例:
        {("node_1", "node_2"): [0, 3], ("node_5", "node_6"): [7]}
    """
    edge_to_gates: dict[tuple[str, str], list[int]] = {}
    for gate in gates:
        if gate.edge is None:
            continue
        edge_key = (gate.edge[0], gate.edge[1])  # 只取 (u, v)，忽略 key
        edge_to_gates.setdefault(edge_key, []).append(gate.gate_index)
    return edge_to_gates


def _assignment_cache_payload(
    gates: list[GateInfo],
    od_pairs: list[tuple[int, int]],
    K: int,
    theta: float,
) -> dict[str, object]:
    """
    构建 H 矩阵缓存的「指纹」数据。

    包含所有影响 H 矩阵结果的输入参数（K、theta、gate 配置、OD 对列表），
    用于判断缓存是否仍然有效：只要这些参数没变，H 矩阵就可以复用。
    """
    gate_signature = [
        [gate.gate_index, gate.camera_id, gate.gate_id, list(gate.edge) if gate.edge else None]
        for gate in gates
    ]
    return {
        "K": int(K),
        "theta": float(theta),
        "gate_signature": gate_signature,
        "od_pairs": od_pairs,
    }


def _assignment_cache_paths(
    paths: ProjectPaths,
    gates: list[GateInfo],
    od_pairs: list[tuple[int, int]],
    K: int,
    theta: float,
) -> tuple[Path, Path]:
    """
    根据参数指纹生成 H 矩阵的缓存文件路径。

    文件名格式: assignment_matrix__k3__theta0p100000__<sha1前12位>.npz
    SHA1 哈希确保参数任何变化都会产生不同的文件名，避免误用过期缓存。

    返回:
        (npz文件路径, json元数据路径) 元组
    """
    payload = _assignment_cache_payload(gates, od_pairs, K, theta)
    # 对参数指纹做 SHA1 哈希，取前 12 位作为文件名的一部分
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    theta_token = f"{theta:.6f}".replace(".", "p")  # 小数点替换为 p，方便文件命名
    base = f"assignment_matrix__k{K}__theta{theta_token}__{digest}"
    return paths.output_dir / f"{base}.npz", paths.output_dir / f"{base}.json"


def build_assignment_matrix(
    G: nx.MultiDiGraph,
    zone_node_map: dict[int, str],
    gates: list[GateInfo],
    K: int = DEFAULT_K_PATHS,
    theta: float = DEFAULT_LOGIT_THETA,
    paths: ProjectPaths | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
) -> tuple[sp.csr_matrix, list[tuple[int, int]]]:
    """
    构建分配矩阵 H —— 本模块的核心函数。

    H 的维度: (n_gates × n_od)
    H[g, od] = OD 对 od 的单位流量经过 gate g 的总概率

    构建算法（对每个 OD 对）：
    1. 找到 origin zone 和 dest zone 对应的路网节点
    2. 计算 K 条最短路径及各自的代价
    3. 用 Logit 模型将代价转为选择概率 P(路径k)
    4. 遍历每条路径的每条边：
       如果该边上有 gate，则 H[gate, od] += P(路径k)

    示意图（K=2 的情况）：
      OD 对: zone_1 → zone_5
      路径 1 (代价 100, 概率 0.7): A → B → C → D
      路径 2 (代价 150, 概率 0.3): A → E → F → D
      如果 gate_3 在边 B→C 上 → H[3, od_idx] += 0.7
      如果 gate_7 在边 E→F 上 → H[7, od_idx] += 0.3

    缓存机制：H 矩阵计算耗时（需要对 506 个 OD 对各求 K 条最短路径），
    所以计算结果以 NPZ 格式缓存到磁盘。

    参数:
        G: 路网多重有向图
        zone_node_map: {zone_id: 路网节点ID}
        gates: gate 列表（已映射到路网边）
        K: 候选路径数量
        theta: Logit 温度参数
        paths: 项目路径配置
        use_cache / write_cache: 缓存控制
    返回:
        (H, od_pairs) 元组:
          - H: 稀疏 CSR 矩阵 (n_gates × n_od)
          - od_pairs: OD 对列表，H 的列顺序与此列表对应
    """
    paths = paths or default_paths()
    od_pairs, _ = build_od_pairs(zone_node_map)
    n_gates = len(gates)
    n_od = len(od_pairs)
    matrix_path, metadata_path = _assignment_cache_paths(paths, gates, od_pairs, K, theta)

    # ---------- 尝试加载缓存 ----------
    if use_cache and matrix_path.exists() and metadata_path.exists():
        logger.info("Loading cached assignment matrix from %s", matrix_path)
        H = sp.load_npz(matrix_path)
        return H, od_pairs

    # ---------- 从头构建 H 矩阵 ----------
    # MultiDiGraph → DiGraph（合并平行边，shortest_simple_paths 需要简单图）
    D = multi_to_digraph(G) if isinstance(G, nx.MultiDiGraph) else G
    # 构建 edge → gates 反向索引，用于快速查找路径经过了哪些 gate
    edge_to_gates = build_edge_to_gates_mapping(gates)
    # 使用 LIL (List of Lists) 格式构建稀疏矩阵，适合逐元素填充
    H = sp.lil_matrix((n_gates, n_od), dtype=np.float64)

    for od_idx, (origin, dest) in enumerate(od_pairs):
        # 找到 OD 对对应的路网节点
        source = str(zone_node_map[origin])
        target = str(zone_node_map[dest])
        # 计算 K 条最短路径
        path_list = compute_k_shortest_paths(D, source, target, K)
        if not path_list:
            continue  # 不可达的 OD 对，H 的对应列全为 0

        # 用 Logit 模型将路径代价转为选择概率
        probs = logit_probabilities([cost for _, cost in path_list], theta)

        # 遍历每条路径的每条边，累加概率到 H 矩阵
        for path_idx, (path_nodes, _) in enumerate(path_list):
            prob = float(probs[path_idx])  # 这条路径的选择概率
            for idx in range(len(path_nodes) - 1):
                edge_key = (path_nodes[idx], path_nodes[idx + 1])
                # 查找这条边上是否有 gate，有则累加概率
                for gate_index in edge_to_gates.get(edge_key, []):
                    H[gate_index, od_idx] += prob

    # LIL → CSR 格式转换（CSR 适合矩阵运算，如后续的 H · x）
    H = H.tocsr()

    # ---------- 写入缓存 ----------
    if write_cache:
        paths.ensure_output_dir()
        sp.save_npz(matrix_path, H)
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(_assignment_cache_payload(gates, od_pairs, K, theta), handle, indent=2)
    return H, od_pairs


def precompute_k_paths(
    G: nx.MultiDiGraph,
    zone_node_map: dict[int, str],
    gates: list[GateInfo],
    K: int = DEFAULT_K_PATHS,
) -> PathCache:
    """
    预计算所有 OD 对的 K 条最短路径，缓存到 PathCache 对象中。

    与 build_assignment_matrix 的区别：
    - build_assignment_matrix 直接构建 H 矩阵（固定 theta）
    - precompute_k_paths 只存路径本身（不绑定 theta），
      后续可以用 rebuild_assignment_matrix 以不同的 theta 快速重建 H

    这在需要对 theta 做参数搜索/敏感性分析时很有用：
    路径计算只需做一次，Logit 概率可以反复调整。

    参数:
        G: 路网多重有向图
        zone_node_map: {zone_id: 路网节点ID}
        gates: gate 列表
        K: 候选路径数量
    返回:
        PathCache 对象，包含所有 OD 对的路径信息和辅助映射
    """
    od_pairs, _ = build_od_pairs(zone_node_map)
    edge_to_gates = build_edge_to_gates_mapping(gates)
    D = multi_to_digraph(G) if isinstance(G, nx.MultiDiGraph) else G
    paths: dict[int, list[tuple[list[str], float]]] = {}
    for od_idx, (origin, dest) in enumerate(od_pairs):
        source = str(zone_node_map[origin])
        target = str(zone_node_map[dest])
        result = compute_k_shortest_paths(D, source, target, K)
        if result:
            paths[od_idx] = result  # 只存有可达路径的 OD 对
    return PathCache(od_pairs=od_pairs, paths=paths, edge_to_gates=edge_to_gates, n_gates=len(gates))


def rebuild_assignment_matrix(
    path_cache: PathCache,
    theta: float = DEFAULT_LOGIT_THETA,
) -> tuple[sp.csr_matrix, list[tuple[int, int]]]:
    """
    从预计算的路径缓存中，用给定的 theta 重建 H 矩阵。

    逻辑与 build_assignment_matrix 的核心循环完全相同，
    但跳过了耗时的 K 最短路径计算（直接从 PathCache 读取）。

    典型用法：
        cache = precompute_k_paths(G, zone_node_map, gates, K=3)
        H1, _ = rebuild_assignment_matrix(cache, theta=0.05)  # 较均匀的分配
        H2, _ = rebuild_assignment_matrix(cache, theta=0.5)   # 较集中的分配

    参数:
        path_cache: precompute_k_paths 返回的 PathCache 对象
        theta: Logit 温度参数
    返回:
        (H, od_pairs) 元组
    """
    H = sp.lil_matrix((path_cache.n_gates, len(path_cache.od_pairs)), dtype=np.float64)
    for od_idx, path_list in path_cache.paths.items():
        probs = logit_probabilities([cost for _, cost in path_list], theta)
        for path_idx, (path_nodes, _) in enumerate(path_list):
            prob = float(probs[path_idx])
            for idx in range(len(path_nodes) - 1):
                edge_key = (path_nodes[idx], path_nodes[idx + 1])
                for gate_index in path_cache.edge_to_gates.get(edge_key, []):
                    H[gate_index, od_idx] += prob
    return H.tocsr(), path_cache.od_pairs


def build_assignment(
    G: nx.MultiDiGraph,
    zone_node_map: dict[int, str],
    gates: list[GateInfo],
    K: int = DEFAULT_K_PATHS,
    theta: float = DEFAULT_LOGIT_THETA,
    paths: ProjectPaths | None = None,
    use_cache: bool = True,
    write_cache: bool = True,
) -> tuple[sp.csr_matrix, list[tuple[int, int]]]:
    """
    构建分配矩阵的便捷入口函数。

    直接委托给 build_assignment_matrix，提供更简短的函数名。
    pipeline.py 中通过此函数调用。
    """
    return build_assignment_matrix(
        G,
        zone_node_map,
        gates,
        K=K,
        theta=theta,
        paths=paths,
        use_cache=use_cache,
        write_cache=write_cache,
    )
