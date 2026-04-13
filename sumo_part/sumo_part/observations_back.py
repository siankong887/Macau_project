"""
observations.py — 观测数据解析与处理模块

本模块负责将摄像头原始 JSON 观测数据转化为贝叶斯推断所需的向量/矩阵格式。
整个处理分为三个层次：

1. Gate 级观测：每个摄像头 gate 的多时段流量 → 取均值得到观测向量 g
2. Edge 级聚合：将同一路网边上多个 gate 的流量加和，作为贝叶斯更新的实际输入
3. 转向比例：按摄像头路口统计各方向（edge）的流量占比

数据流向：
  time_limit.json + a1_copy_2.json
      ↓ parse_observations()        — 解析原始 JSON，按 gate 全局索引整理
      ↓ build_observation_vector()   — 多时段取均值，构造 g 向量和协方差 Σ
      ↓ aggregate_to_edge_level()    — 同一 edge 上的 gate 流量加和
      ↓ build_turning_observations() — 按摄像头路口计算转向比例
      → PreparedObservations         — 打包所有观测数据供后续使用
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .paths import ProjectPaths, default_paths
from .types import CameraTurningObs, GateInfo, ObservationData, PreparedObservations, TurningData

logger = logging.getLogger(__name__)

# 系统支持的四种车辆类型，每种车型独立进行 OD 估计
VEHICLE_TYPES = ["car", "bus", "truck", "motorcycle"]


def build_gate_index(camera_config_path: Path) -> tuple[dict[tuple[str, str], int], list[tuple[str, str]]]:
    """
    根据摄像头配置文件 (a1_copy_2.json) 构建 gate 的全局索引。

    摄像头配置文件结构示例：
      { "list": [
          { "camera": "cam_001",
            "gate": [ {"gate_id": "gate_1", ...}, {"gate_id": "gate_2", ...} ]
          }, ...
      ]}

    本函数遍历所有摄像头的所有 gate，按顺序分配一个全局唯一的整数索引。
    这个索引在后续构建观测向量 g 和分配矩阵 H 时用于统一编号。

    参数:
        camera_config_path: 摄像头配置 JSON 文件路径 (a1_copy_2.json)

    返回:
        gate_to_idx: 字典，(camera_id, gate_id) → 全局索引
        idx_to_gate: 列表，全局索引 → (camera_id, gate_id)，即反向映射
    """
    with open(camera_config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    #冒号后面只是在定义字典的结构和数据类型，第一段那个dict[...]说的是key是一个元组，比如
    #("cam_a1,"gate_1"),int是值的类型，所以说事全局唯一索引
    #idx_to_gate这个列表同样是一个映射，给定一个int类型的idx，反推出是哪一个key，原理是依靠列表
    #自带的顺序，按顺序以O(1)复杂度查询

    gate_to_idx: dict[tuple[str, str], int] = {}
    idx_to_gate: list[tuple[str, str]] = []
    idx = 0

    for camera in config["list"]:
        for gate in camera["gate"]:
            key = (camera["camera"], gate["gate_id"])
            gate_to_idx[key] = idx
            idx_to_gate.append(key)
            idx += 1

    return gate_to_idx, idx_to_gate


def parse_observations(
    obs_path: Path,
    camera_config_path: Path,
) -> dict[str, dict[int, list[float]]]:
    """
    解析原始观测 JSON (time_limit.json)，按车型和 gate 全局索引整理流量数据。

    观测 JSON 结构示例 (time_limit.json)：
      {
        "cam_001": [                          ← 摄像头 ID 作为 key
          {                                    ← 一个时段的观测
            "data": [                          ← 各 gate 的观测（按 gate 顺序排列）
              {"car": 120, "bus": 5, "truck": 3, "motorcycle": 20},  ← gate_1 的数据
              {"car": 80,  "bus": 3, "truck": 1, "motorcycle": 10},  ← gate_2 的数据
            ]
          },
          { "data": [...] },                   ← 另一个时段
        ],
        "cam_002": [...]
      }

    处理逻辑：
    1. 根据摄像头配置获取 gate 全局索引映射
    2. 遍历每个摄像头的每个时段
    3. 校验 gate 数量是否与配置一致（不一致则警告但仍处理）
    4. 按 (camera_id, "gate_{局部索引+1}") 查找全局索引
    5. 将各车型流量追加到对应 gate 的列表中（一个 gate 可能有多个时段的观测）

    注意：gate_key 的构造方式是 "gate_{局部索引+1}"（从 1 开始），
    这依赖于观测 JSON 中 data 数组的顺序与配置文件中 gate 列表的顺序一致。

    参数:
        obs_path: 观测数据 JSON 文件路径 (time_limit.json)
        camera_config_path: 摄像头配置 JSON 文件路径 (a1_copy_2.json)

    返回:
        raw_obs: 嵌套字典，结构为 {车型: {gate全局索引: [各时段流量值]}}
                 例如 {"car": {0: [120, 115, 130], 1: [80, 75]}, "bus": {...}}
    """
    with open(obs_path, "r", encoding="utf-8") as handle:
        obs_data = json.load(handle)

    gate_to_idx, _ = build_gate_index(camera_config_path)

    # 构建每个摄像头应有的 gate 数量映射，用于数据校验
    with open(camera_config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    #cam_n_gates是一个字典，key是摄像头id，比如"cam_a1","cam_a2"这种，值是这个摄像头有多少个gate
    cam_n_gates = {camera["camera"]: len(camera["gate"]) for camera in config["list"]}

    # 初始化结果字典：{车型: {gate全局索引: [各时段流量]}}
    raw_obs = {vehicle_type: {} for vehicle_type in VEHICLE_TYPES}

    #列出time_limit.json那里的key，比如“cam_a2”,值是一个列表，里面有这个摄像头各个时段的观测数据
    for camera_id, periods in obs_data.items():
        #等式右边用camera_id,比如“cam_a1”,去查这个摄像头有多少个gate并赋值
        expected_gates = cam_n_gates.get(camera_id)
        if expected_gates is None:
            # 观测数据中出现了配置文件中不存在的摄像头，跳过，因为cam_n_gates变量通过配置得到，
            #cam_id是观测数据中的摄像头名称
            continue
        #循环那个包含各个时段数据的列表
        for period in periods:
            data = period.get("data", [])
            #为空的时段就跳过
            if not data:
                continue

            # 校验：观测数据中的 gate 数量是否与摄像头配置一致
            if len(data) != expected_gates:
                logger.warning(
                    "Camera %s expected %d gates but got %d entries",
                    camera_id,
                    expected_gates,
                    len(data),
                )
            #enumerate函数会列出索引和原始数据
            for gate_idx_local, gate_data in enumerate(data):
                # 构造 gate key：假设 data 数组中的顺序对应 gate_1, gate_2, ...
                gate_key = (camera_id, f"gate_{gate_idx_local + 1}")
                #用元组去查全局的gate索引，global_idx是int类型
                global_idx = gate_to_idx.get(gate_key)
                if global_idx is None:
                    # 该 gate 在配置中不存在（可能 data 数量超出配置），跳过
                    continue

                # 提取每种车型的流量，追加到该 gate 的时序列表中
                for vehicle_type in VEHICLE_TYPES:
                    raw_obs[vehicle_type].setdefault(global_idx, []).append(
                        float(gate_data.get(vehicle_type, 0.0))
                    )
    """
    raw_obs 数据大致长这样：
    raw_obs = {
    "car": {
        0: [120.0, 95.0],   # 全局索引0的gate，两个时段的car流量
        1: [80.0, 70.0],    # 全局索引1的gate
        2: [200.0, 180.0],  # 全局索引2的gate
    },
    "bus": {
        0: [5.0, 3.0],
        1: [3.0, 2.0],
        2: [8.0, 6.0],
    },
    "truck": {
        0: [3.0, 1.0],
        1: [1.0, 0.0],
        2: [5.0, 4.0],
    },
    "motorcycle": {
        0: [20.0, 15.0],
        1: [10.0, 8.0],
        2: [30.0, 25.0],
    },
}
    第一层键是车型，第二层键是gate 全局索引，值是一个列表，
    每个元素对应一个时段的观测流量。列表会随着时段数据的遍历不断 append 增长
    """
    return raw_obs


def build_observation_vector(
    raw_obs: dict[str, dict[int, list[float]]],
    vehicle_type: str,
    min_variance: float = 1.0,
) -> ObservationData:
    """
    将某车型的原始多时段观测数据转化为 gate 级观测向量 g 和噪声协方差矩阵 Σ。

    处理逻辑：
    1. 从 raw_obs 中提取指定车型的数据
    2. 按 gate 全局索引排序（确保与分配矩阵 H 的行对齐）
    3. 对每个 gate 的多个时段的流量取 **均值** 作为观测值 g[i]
    4. 构建对角协方差矩阵 Σ：Σ[i,i] = max(g[i], min_variance)
       - 含义：流量越大的 gate，观测噪声方差越大（泊松近似）
       - min_variance 保证方差不为零，避免数值问题

    注意事项：
    - 这里对多时段取均值，隐含假设是各时段的流量模式相似（稳态假设）
    - 协方差矩阵是对角阵，即假设各 gate 的观测噪声相互独立
    - 方差直接用均值（而非样本方差），这是泊松分布的特性 (Var ≈ Mean)

    参数:
        raw_obs: parse_observations() 的返回值，{车型: {gate全局索引: [各时段流量]}}
        vehicle_type: 要处理的车型，如 "car"
        min_variance: 方差下限，防止方差为零导致矩阵奇异，默认 1.0

    返回:
        ObservationData 包含:
          - g: 观测向量，shape=(n_gates,)，每个 gate 的平均流量
          - sigma: 对角协方差矩阵，shape=(n_gates, n_gates)
          - gate_indices: 排序后的 gate 全局索引列表
          - n_time_periods: 最大时段数（各 gate 的时段数可能不同，取最大值）
          - total_flow: 所有 gate 流量之和
    """
    vehicle_data = raw_obs.get(vehicle_type, {})

    # 该车型没有任何观测数据时，返回空的 ObservationData
    if not vehicle_data:
        return ObservationData(
            g=np.array([]),
            sigma=np.array([]).reshape(0, 0),
            gate_indices=[],
            vehicle_type=vehicle_type,
            n_time_periods=0,
            total_flow=0.0,
        )

    # 按 gate 全局索引排序，确保顺序一致
    sorted_gates = sorted(vehicle_data.keys())
    g = np.zeros(len(sorted_gates))
    n_periods = 0

    for idx, gate_index in enumerate(sorted_gates):
        counts = vehicle_data[gate_index]  # 该 gate 在各时段的流量列表
        g[idx] = float(np.mean(counts))    # 取多时段均值作为观测值
        n_periods = max(n_periods, len(counts))  # 记录最大时段数

    # 对角协方差矩阵：方差 = max(均值, min_variance)
    # 这是基于泊松分布 Var≈Mean 的近似，流量大的 gate 允许更大的噪声
    sigma = np.diag(np.maximum(g, min_variance))

    return ObservationData(
        g=g,
        sigma=sigma,
        gate_indices=sorted_gates,
        vehicle_type=vehicle_type,
        n_time_periods=n_periods,
        total_flow=float(np.sum(g)),
    )


def aggregate_to_edge_level(
    gate_observations: ObservationData,
    gates: list[GateInfo],
    min_variance: float = 1.0,
) -> ObservationData:
    """
    将 gate 级观测聚合到 edge（路网边）级别。

    背景：同一条路网边上可能有多个 gate（例如同一路口的不同检测线都映射到了
    同一条路网边）。贝叶斯推断中，分配矩阵 H 的行应该与观测向量 g 对齐，
    如果 H 是按 edge 构建的，那么观测也需要聚合到 edge 级。

    聚合方式：
    - 流量：同一 edge 下所有 gate 的观测流量 **加和**
    - 方差：同一 edge 下所有 gate 的方差 **加和**（独立变量之和的方差 = 各方差之和）

    注意事项：
    - edge key 只取 (edge[0], edge[1])，即 (起始节点, 终止节点)，
      忽略了 MultiDiGraph 中的第三个 key（同向多条边只看前两个元素）
    - 每个 edge 用其 **第一个** gate 的索引作为代表 (rep_gate_indices)
    - 未映射到 edge 的 gate（edge=None）会被丢弃

    参数:
        gate_observations: gate 级的观测数据 (build_observation_vector 的输出)
        gates: 所有 gate 的信息列表 (含 edge 映射结果)
        min_variance: 聚合后方差的下限

    返回:
        ObservationData: edge 级的观测数据，维度通常小于 gate 级
    """
    # 构建 gate全局索引 → edge(u,v) 的映射（只取前两个元素，忽略 MultiDiGraph key）
    gate_idx_to_edge = {
        gate.gate_index: (gate.edge[0], gate.edge[1])
        for gate in gates
        if gate.edge is not None
    }

    # edge_positions: edge → 该 edge 下各 gate 在 gate_observations.g 中的位置索引
    # edge_gate_indices: edge → 该 edge 下各 gate 的全局索引
    edge_positions: dict[tuple[str, str], list[int]] = {}
    edge_gate_indices: dict[tuple[str, str], list[int]] = {}

    for pos, gate_index in enumerate(gate_observations.gate_indices):
        edge = gate_idx_to_edge.get(gate_index)
        if edge is None:
            # 该 gate 没有映射到任何路网边，跳过（不参与贝叶斯更新）
            continue
        edge_positions.setdefault(edge, []).append(pos)
        edge_gate_indices.setdefault(edge, []).append(gate_index)

    # 按 edge 排序，确保顺序确定性
    edges_sorted = sorted(edge_positions.keys())
    g_edge = np.zeros(len(edges_sorted))
    sigma_diag = np.zeros(len(edges_sorted))
    rep_gate_indices: list[int] = []  # 每个 edge 的代表 gate 索引

    for idx, edge in enumerate(edges_sorted):
        positions = edge_positions[edge]
        members = edge_gate_indices[edge]

        # 同一 edge 下所有 gate 流量加和
        g_edge[idx] = float(np.sum([gate_observations.g[pos] for pos in positions]))

        # 独立高斯变量之和的方差 = 各方差之和
        sigma_diag[idx] = float(np.sum([gate_observations.sigma[pos, pos] for pos in positions]))
        sigma_diag[idx] = max(sigma_diag[idx], min_variance)

        # 用第一个 gate 的索引作为该 edge 的代表
        rep_gate_indices.append(members[0])

    return ObservationData(
        g=g_edge,
        sigma=np.diag(sigma_diag),
        gate_indices=rep_gate_indices,
        vehicle_type=gate_observations.vehicle_type,
        n_time_periods=gate_observations.n_time_periods,
        total_flow=float(np.sum(g_edge)),
    )


def build_turning_observations(
    raw_obs: dict[str, dict[int, list[float]]],
    gates: list[GateInfo],
    vehicle_type: str,
    min_total_count: float = 1.0,
) -> TurningData:
    """
    构建各摄像头路口的转向比例观测数据。

    "转向比例"指的是：在一个路口，进入各个方向（edge）的车流占该路口总流量的比例。
    例如一个十字路口有 3 个方向有车流：东→西 = 40%，南→北 = 35%，西→东 = 25%。

    处理逻辑：
    1. 将所有 gate 按 camera_id 分组（同一摄像头路口的 gate）
    2. 对每个摄像头路口：
       a. 将该路口的 gate 按 edge (路网边) 再分组 — 同一 edge 上的多个 gate
          合并为一个 "supergate"
       b. 同一 edge 的流量 = 该 edge 下所有 gate 平均流量之和
       c. 计算各 edge 的流量占比 = edge流量 / 路口总流量
    3. 过滤条件：
       - 路口至少要有 2 个不同的 edge（只有 1 个方向没有"转向"的概念）
       - 路口总流量 >= min_total_count（流量太低的路口统计意义不大）

    "supergate" 的概念：同一 edge 上可能有多个 gate，将它们合并为一个
    supergate，代表该 edge/方向的总流量。n_supergates 即该路口有几个不同方向。

    独立观测数 (total_independent) = 各路口的 (n_supergates - 1) 之和，
    因为比例之和 = 1，每个路口最后一个比例由其他比例决定，不是独立的。

    参数:
        raw_obs: parse_observations() 的返回值
        gates: 所有 gate 信息列表（含 edge 映射）
        vehicle_type: 车型
        min_total_count: 路口最低总流量阈值，低于此值的路口被忽略

    返回:
        TurningData: 包含所有有效路口的转向观测数据
    """
    vehicle_data = raw_obs.get(vehicle_type, {})

    # 第一步：按 camera_id 对 gate 进行分组
    cameras_gates: dict[str, list[GateInfo]] = {}
    for gate in gates:
        cameras_gates.setdefault(gate.camera_id, []).append(gate)

    cameras: list[CameraTurningObs] = []

    for camera_id, camera_gates in cameras_gates.items():
        # edge_groups: edge(u,v) → 属于该 edge 的 gate 全局索引列表
        edge_groups: dict[tuple[str, str], list[int]] = {}
        # edge_counts: edge(u,v) → 该 edge 的合计平均流量
        edge_counts: dict[tuple[str, str], float] = {}

        for gate in camera_gates:
            # 跳过未映射到 edge 或无观测数据的 gate
            if gate.edge is None or gate.gate_index not in vehicle_data:
                continue

            edge_key = (gate.edge[0], gate.edge[1])
            edge_groups.setdefault(edge_key, []).append(gate.gate_index)

            # 该 gate 的平均流量累加到对应 edge
            edge_counts[edge_key] = edge_counts.get(edge_key, 0.0) + float(
                np.mean(vehicle_data[gate.gate_index])
            )

        # 只有一个 edge 或没有 edge 的路口，转向比例无意义，跳过
        if len(edge_groups) < 2:
            continue

        edge_keys = sorted(edge_groups.keys())
        observed_counts = np.array([edge_counts[key] for key in edge_keys], dtype=float)
        total = float(np.sum(observed_counts))

        # 总流量太低的路口，观测不可靠，跳过
        if total < min_total_count:
            continue

        cameras.append(
            CameraTurningObs(
                camera_id=camera_id,
                n_supergates=len(edge_keys),          # 该路口有几个不同方向
                edge_keys=edge_keys,                   # 各方向对应的路网边
                observed_counts=observed_counts,       # 各方向的绝对流量
                observed_proportions=observed_counts / total,  # 各方向的流量占比
                gate_indices_per_supergate=[edge_groups[key] for key in edge_keys],  # 各方向包含的 gate 索引
                total_count=total,                     # 路口总流量
            )
        )

    # 汇总统计
    total_observations = sum(camera.n_supergates for camera in cameras)
    # 独立观测数：每个路口的比例之和=1，故自由度为 n_supergates - 1
    total_independent = sum(camera.n_supergates - 1 for camera in cameras)

    return TurningData(
        cameras=cameras,
        n_cameras=len(cameras),
        total_observations=total_observations,
        total_independent=total_independent,
        vehicle_type=vehicle_type,
    )


def prepare_observations(
    gates: list[GateInfo],
    vehicle_type: str,
    paths: ProjectPaths | None = None,
) -> PreparedObservations:
    """
    观测数据处理的顶层入口函数，串联整个观测处理流水线。

    执行流程：
    1. parse_observations()         → 从 JSON 解析原始多时段观测
    2. build_observation_vector()   → 构建 gate 级观测向量 g 和协方差 Σ
    3. aggregate_to_edge_level()    → 聚合到 edge 级（贝叶斯更新的实际输入）
    4. build_turning_observations() → 构建转向比例数据（用于结果输出/诊断）

    参数:
        gates: 所有 gate 信息列表（需要已完成 edge 映射，由 network 模块提供）
        vehicle_type: 要处理的车型，如 "car"
        paths: 项目路径配置，为 None 时使用默认路径

    返回:
        PreparedObservations: 打包了三种粒度的观测数据 + 原始数据
          - gate_observations: gate 级（最细粒度）
          - edge_observations: edge 级（贝叶斯更新用）
          - turning_observations: 转向比例（结果输出用）
          - raw_observations: 原始解析数据（供调试或其他用途）
    """
    paths = paths or default_paths()

    # 第一步：解析原始 JSON 观测数据
    raw_obs = parse_observations(paths.observation_json, paths.camera_config_json)

    # 第二步：构建 gate 级观测向量（多时段取均值）
    gate_observations = build_observation_vector(raw_obs, vehicle_type)

    # 第三步：聚合到 edge 级（同一路网边上的 gate 流量加和）
    edge_observations = aggregate_to_edge_level(gate_observations, gates)

    # 第四步：构建转向比例观测
    turning_observations = build_turning_observations(raw_obs, gates, vehicle_type)

    return PreparedObservations(
        gate_observations=gate_observations,
        edge_observations=edge_observations,
        turning_observations=turning_observations,
        raw_observations=raw_obs,
    )
