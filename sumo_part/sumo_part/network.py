"""
network.py — 核心路网数据处理与交通分析小区(TAZ)空间映射模块

本模块负责 OD 矩阵估计管线中的基础空间数据准备：
1. 路网加载：支持多级降级策略（首选 GraphML 缓存，其次为本地 OSM PBF 提取，最后为 osmnx 在线抓取）加载澳门驾驶级路网。
2. 属性清洗：由于 GraphML 序列化限制（全量字符串化），需强制进行属性的类型向下还原（浮点数、布尔值等）。
3. 阻抗计算：融合路网几何长度与不同道路分类期望时速，计算路由基础代价 `travel_time`（通行时间）。
4. 区域映射：解析包含核心 23 个交通小区的 GeoJSON 边界，计算可靠质心（代表点），并在最大强连通子图 (LSCC) 中约束就近节点映射，以确保全图 OD 生成的可达性。

上层调用链路主入口为 `prepare_network()`。
"""

from __future__ import annotations

import ast
import csv
import json
import logging
from math import sqrt
from pathlib import Path

import networkx as nx
from shapely.geometry import shape

from .paths import ProjectPaths, default_paths

logger = logging.getLogger(__name__)

# 图属性映射常量：指明 GraphML 内反序列化时应严格约束为浮点类型的键名
FLOAT_EDGE_FIELDS = {"length", "speed_kph", "travel_time"}
FLOAT_NODE_FIELDS = {"x", "y"}


def load_speed_mapping(csv_path: Path) -> dict[str, float]:
    """
    加载由 OSM 道路分级至理论行驶速度映射字典。

    参数:
        csv_path (Path): 配置表 CSV 文件路径。需包含以下核心列：
            - fclass (OSM 主标记分类, 例如 primary)
            - estimated_speed_kmh (预期公里时速)

    返回:
        dict[str, float]: 包含键值对映射的字典。
    """
    speed_map: dict[str, float] = {}
    with open(csv_path, "r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                speed_map[row["fclass"]] = float(row["estimated_speed_kmh"])
            except (KeyError, TypeError, ValueError):
                continue
    return speed_map


def _coerce_value(value: object) -> object:
    """
    启发式类型推断算法，将 GraphML 的字符串强转回原生 Python 数据结构。

    解决顺序:
        1. 逻辑值判断 ("True", "False") -> bool
        2. 富数字判定(含小数点/科学计数符号e) -> float
        3. 基本整数 -> int
        4. 回退处理 -> 保留原生 str
    """
    if isinstance(value, str):
        if value in {"True", "False"}:
            return value == "True"
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            return int(value)
        except ValueError:
            return value
    return value


def _coerce_graph_attributes(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    执行路网有向图中拓扑节点和边的全属性深层次数据类型清洗。
    此行为规避了 GraphML 通用序列化规则引起的前置路由计算异常。

    参数:
        G (nx.MultiDiGraph): 反序列化后尚未确省数据类型的有向多重路网图

    返回:
        nx.MultiDiGraph: 已清洗并具备结构化确切类型的同位图指针
    """
    graph_type = G.graph.get("graphml_defaultedgetype")
    if not isinstance(G, nx.MultiDiGraph):
        G = nx.MultiDiGraph(G)

    for _, data in G.nodes(data=True):
        for key in list(data.keys()):
            if key in FLOAT_NODE_FIELDS:
                data[key] = float(data[key])
            else:
                data[key] = _coerce_value(data[key])

    for _, _, _, data in G.edges(keys=True, data=True):
        for key in list(data.keys()):
            if key in FLOAT_EDGE_FIELDS:
                try:
                    data[key] = float(data[key])
                except (TypeError, ValueError):
                    pass # 无法推断时暂存原生值
            else:
                data[key] = _coerce_value(data[key])

    if graph_type:
        G.graph["graphml_defaultedgetype"] = graph_type
    return G


def _normalize_highway(value: object) -> str | None:
    """
    规范化繁杂的 OSM 'highway' 道路标签表示法。

    消除 OSM 标签层级在多源获取（原生列表，字符串表征化数组，普通字符组合）
    出现的不等式差异，统一抽离主分类。
    """
    if value is None:
        return None
    if isinstance(value, list):
        return str(value[0]) if value else None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        return stripped
    return str(value)


def add_travel_time_weights(
    G: nx.MultiDiGraph,
    speed_dict: dict[str, float],
    fallback_speed: float = 30.0,
) -> nx.MultiDiGraph:
    """
    为路网拓扑图边属性融合通行阻抗特性 ('travel_time' 秒级测度)。
    这是全网段最短路径生成最核心的代价计算方法。

    流转判定逻辑：
        1. 首先采信拓扑原始属性中的标准设定档 (`speed_kph`)。
        2. 若缺失，引用本图级路网字典 `speed_dict` 与公路类级的对焦映射。
        3. 上述方案均落空的情况下采纳静态常量约束 `fallback_speed`。

    参数:
        G: The road network graph.
        speed_dict: 公路分级同预计均速 (km/h) 之间映射字串字典。
        fallback_speed: 默认推算阈值速度常量。

    返回:
        nx.MultiDiGraph: 完成边缘阻抗信息扩展的有向多项图对象。
    """
    for _, _, _, data in G.edges(keys=True, data=True):
        highway = _normalize_highway(data.get("highway"))
        speed = data.get("speed_kph")

        if speed in (None, "", 0, 0.0):
            speed = speed_dict.get(highway, fallback_speed)

        length = float(data.get("length", 0.0))
        data["speed_kph"] = float(speed)

        # 将速度度量元制统一推论为国际基本制秒/米进行标量时间化
        if "travel_time" not in data or not data["travel_time"]:
            meters_per_second = max(float(speed) * 1000.0 / 3600.0, 1e-6) # 阈值化以防分母置零
            data["travel_time"] = length / meters_per_second
        else:
            data["travel_time"] = float(data["travel_time"])

    return G


def load_graph_from_graphml(graphml_path: Path) -> nx.MultiDiGraph:
    """
    主线逻辑（优先级一）：挂载本地 GraphML 预构建缓存网络图，
    并实施动态序列属性规范整理，最大化运行时系统资源效能。
    """
    logger.info("Loading network graph from %s", graphml_path)
    return _coerce_graph_attributes(nx.read_graphml(graphml_path))


def _load_graph_from_local_osm(osm_path: Path, speed_map: dict[str, float]) -> nx.MultiDiGraph:
    """
    支线逻辑（优先级二）：基于标准协议直接通过 pyrosm 包将 OSM （.pbf）数据实体解压缩到
    运行内存中用以构造拓扑路网逻辑，为 GraphML 脱离服务时提供本地降级安全带。

    异常:
        RuntimeError: 当缺失依赖解析包，或协议输入尾缀非规约时报警挂起。
    """
    # 检查文件后缀是否为 .pbf（Protocolbuffer Binary Format，OSM的高效二进制格式）
    if osm_path.suffix.lower() == ".pbf":

        # 延迟导入 pyrosm 库（专门解析 .pbf 文件的工具）
        # 放在 try 里是因为 pyrosm 是可选依赖，没装的话给出友好提示
        try:
            from pyrosm import OSM  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Local .pbf loading requires pyrosm. Provide a GraphML cache instead."
            ) from exc

        # 创建 pyrosm 的 OSM 解析器对象，传入 .pbf 文件路径
        osm = OSM(str(osm_path))

        # 提取驾车路网的节点和边，返回两个 GeoDataFrame（类似带地理信息的表格）
        # nodes: 所有路口/交叉点，每行有 id、lon（经度）、lat（纬度）等
        # edges: 所有路段，每行有 u（起点）、v（终点）、id、长度、道路类型等
        nodes, edges = osm.get_network(nodes=True, network_type="driving")

        # 创建一个空的 NetworkX 有向多重图
        # MultiDiGraph: Multi=允许两点间有多条边（如平行道路）, Di=有方向（单行道）
        G = nx.MultiDiGraph()

        # ---- 逐行遍历节点表，把每个路口添加到图中 ----
        # _ 是行索引（不需要），row 是该节点的一行数据
        # 每个节点带上 x（经度）和 y（纬度）属性，id 转为字符串作为节点标识
        for _, row in nodes.iterrows():
            G.add_node(str(row["id"]), x=float(row["lon"]), y=float(row["lat"]))

        # ---- 逐行遍历边表，把每条路段添加到图中 ----
        for _, row in edges.iterrows():
            # 把这一行所有字段转成字典，方便操作
            attrs = row.to_dict()
            # 从字典中弹出起点 u 和终点 v（弹出=取值并删除，因为它们不是边的属性而是图的结构）
            u = str(attrs.pop("u"))
            v = str(attrs.pop("v"))
            # 弹出边的唯一标识 id，如果没有就用 "起点-终点" 拼接作为 key
            key = str(attrs.pop("id", f"{u}-{v}"))
            # 将边添加到图中，**attrs 把剩余的所有字段（name、lanes、highway等）作为边属性展开传入
            G.add_edge(u, v, key=key, **attrs)

        # 图构建完成后，根据速度映射表给每条边计算并添加 travel_time 权重
        # 这样图就完整了：结构（哪连哪）+ 权重（走多久）
        return add_travel_time_weights(G, speed_map)

    # 如果文件既不是 .pbf 也不是其他支持的格式，抛出异常
    raise RuntimeError(f"Unsupported local OSM file {osm_path.name}. Provide GraphML or install an adapter.")


def _load_graph_online(speed_map: dict[str, float]) -> nx.MultiDiGraph:
    """
    最底层后手降级预案（优先级三）：当所有本地特征层物理信息不可获得，
    且具备公网直连互通环境依赖时激活 OSM 服务器同步查询动作，需 osmnx 强绑定。
    """
    try:
        import osmnx as ox  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Online OSM download requires osmnx, which is not available in this environment.") from exc

    logger.info("Downloading Macau driving network from OpenStreetMap")
    ox.settings.useful_tags_way = list(set(ox.settings.useful_tags_way + ["highway"]))
    G = ox.graph_from_place("Macau", network_type="drive")
    G = add_travel_time_weights(G, speed_map)
    return G


def load_zones_geojson(geojson_path: Path) -> list[tuple[int, object]]:
    """
    抽取交通规划小区 (TAZ) GeoJSON 多边形层地理信息集。

    过滤提取带有合规化属性 `zona` 的多段性区域向量。

    返回:
        按大区标识升序排列后的区域二元组信息列表 `(Zone ID, Geom)`。
    """
    with open(geojson_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    zones: list[tuple[int, object]] = []
    for feature in data["features"]:
        props = feature.get("properties", {})
        zona = props.get("zona") or props.get("Zona")

        if zona is None:
            continue

        zones.append((int(zona), shape(feature["geometry"])))

    zones.sort(key=lambda item: item[0])
    return zones


def compute_zone_centroids(zones: list[tuple[int, object]]) -> dict[int, tuple[float, float]]:
    """
    空间计算每个 TAZ 的内部确切中心代表位坐标。

    使用 `.representative_point()` 取代质心算法 `centroid` 规避类似 U 形不规则区域边界内
    导致拓扑点暴露在有效作用区域外侧造成数据污染的情形。
    """
    return {zona: (geom.representative_point().x, geom.representative_point().y) for zona, geom in zones}


def largest_scc_nodes(G: nx.MultiDiGraph) -> set[str]:
    """
    寻找并截获网络拓扑最大的强连通分量 (Largest Strongly Connected Component)。

    仅取被圈定进入此集内的网络节点位，从而逻辑保障任意一发交通 OD (起点极-终点极) 之间必定
    通过导航规划算法具备完全双向互融连接的硬性通行需求。
    """
    return max(nx.strongly_connected_components(G), key=len)


def map_centroids_to_graph_nodes(
    G: nx.MultiDiGraph,
    centroids: dict[int, tuple[float, float]],
) -> dict[int, str]:
    """
    空间绑定操作：使交通物理区域计算得出的地理虚拟圆心与图节点间具备实体附着与逻辑联结映射机制。

    限定候选搜索池只存在于路网连通主轴 (极大强连通拓扑图)，在二维平面维度利用蛮力求解方式通过算子
    匹配欧氏坐标轴向的物理最短距离节点。

    距离存在重型偏离 (跨度超过 500M) 会上报系统级隐患日志警告信号。
    """
    scc_nodes = largest_scc_nodes(G)
    candidates = [(node, G.nodes[node]["x"], G.nodes[node]["y"]) for node in scc_nodes]
    result: dict[int, str] = {}

    for zona, (lon, lat) in centroids.items():
        best_node = None
        best_dist = float("inf")

        for node, node_lon, node_lat in candidates:
            dist = (node_lon - lon) ** 2 + (node_lat - lat) ** 2
            if dist < best_dist:
                best_dist = dist
                best_node = node

        if best_node is None:
            raise RuntimeError(f"Could not map zone {zona} to a graph node")

        # 换算单位由欧式度换维标准米级用于校验 (1弧度/经纬标 ≈ 111公里)
        approx_dist_m = sqrt(best_dist) * 111000
        if approx_dist_m > 500.0:
            logger.warning("Zone %s mapped %.0fm away from centroid", zona, approx_dist_m)

        result[zona] = str(best_node)

    return result


def load_zone_node_map(
    G: nx.MultiDiGraph,
    zone_nodes_json: Path,
    zones_geojson: Path,
    write_cache: bool = True,
) -> dict[int, str]:
    """
    调度封装层：带前置热存判断并统筹小区至交通点列对齐全周期的宏操作。

    提供结果集的硬盘缓存留档功能，节省热启动时庞大矩阵多次无谓开销计算。
    """
    if zone_nodes_json.exists():
        with open(zone_nodes_json, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return {int(key): str(value) for key, value in data.items()}

    zones = load_zones_geojson(zones_geojson)
    centroids = compute_zone_centroids(zones)
    zone_node_map = map_centroids_to_graph_nodes(G, centroids)

    if write_cache:
        zone_nodes_json.parent.mkdir(parents=True, exist_ok=True)
        with open(zone_nodes_json, "w", encoding="utf-8") as handle:
            json.dump(zone_node_map, handle, indent=2)

    return zone_node_map


def prepare_network(
    paths: ProjectPaths | None = None,
    allow_online: bool = False,
    write_cache: bool = True,
) -> tuple[nx.MultiDiGraph, dict[int, str]]:
    """
    一站式环境与元数据封装装载中心。执行 OD 估计运行管线系统中的空间前导起手预备。

    整体工作流工序编排：
      1. 获取当前场景道路默认时速表参数。
      2. 穿透优先极调度实现核心有向有权重图网络对象接管。
      3. 全图边网络推导重塑阻抗量 `travel_time`（基于路线标距和测速预期）。
      4. 将非连续性的几何物理宏观区域面转化为确切点对点运算起点极/末端极目标簇结构映射表。

    返回:
        (G, zone_node_map):
          - (nx.MultiDiGraph) 配置有基础代价与合法规范化属性组的驾驶型图主干网络实例
          - (dict[int, str]) 包含小区名(1-23)为主键定位于精确图节点唯一 ID标识索引集
    """
    paths = paths or default_paths()
    paths.ensure_output_dir()
    speed_map = load_speed_mapping(paths.speed_mapping_csv)

    # ---- 三级降级策略加载路网图：GraphML缓存 → 本地OSM文件 → 在线下载 ----

    # 优先级1：检查本地是否已有处理好的 GraphML 缓存文件（加载最快）
    if paths.network_graphml.exists():
        # 直接从 GraphML 文件加载路网图，无需再做任何转换
        G = load_graph_from_graphml(paths.network_graphml)

    # 优先级2：GraphML 不存在，但本地有原始的 OSM（OpenStreetMap）数据文件
    elif paths.network_osm.exists():
        logger.info("GraphML cache missing; trying local OSM source %s", paths.network_osm)
        # 从 OSM 原始数据解析并构建路网图（比 GraphML 慢，需要处理速度映射等）
        G = _load_graph_from_local_osm(paths.network_osm, speed_map)
        # 如果允许写缓存，把构建好的图保存为 GraphML，下次就能走优先级1的快速路径
        if write_cache:
            nx.write_graphml(G, paths.network_graphml)

    # 优先级3：本地什么文件都没有，但允许联网，则从 OSM 在线API下载路网数据
    elif allow_online:
        logger.info("Local network files missing; falling back to online OSM download")
        # 通过网络从 OpenStreetMap 下载澳门路网数据并构建图
        G = _load_graph_online(speed_map)
        # 同样缓存为 GraphML，避免每次都联网下载
        if write_cache:
            nx.write_graphml(G, paths.network_graphml)

    # 以上三种方式都不可用，抛出异常提示用户
    else:
        raise FileNotFoundError(
            "No local GraphML or OSM source found. Add data/network/macau_drive.graphml "
            "or run with allow_online=True."
        )

    if not any("travel_time" in data for _, _, _, data in G.edges(keys=True, data=True)):
        G = add_travel_time_weights(G, speed_map)

    zone_node_map = load_zone_node_map(
        G,
        zone_nodes_json=paths.zone_nodes_json,
        zones_geojson=paths.zones_geojson,
        write_cache=write_cache,
    )

    logger.info(
        "Prepared network: %d nodes, %d edges, %d zones",
        G.number_of_nodes(),
        G.number_of_edges(),
        len(zone_node_map),
    )
    return G, zone_node_map

"""
zone_node_map 大致长这样：


{
    1: "node_38921",
    2: "node_12045",
    3: "node_7823",
    ...
    23: "node_51004"
}
key = zone ID（int），对应澳门 23 个交通小区的编号（1~23）
value = 路网节点 ID（str），是 GraphML 路网中距离该 zone 几何中心最近的节点
它的构建过程是：

从 GeoJSON 加载 23 个 zone 的多边形边界
计算每个 zone 的几何中心点（centroid）经纬度
在路网的最大强连通分量中，找距离 centroid 最近的节点

G本质上是networkx.MultiDiGraph的一个实例对象，因为MutiDiGraph提供了很多实用的方法，后续最短路径计算也是
依赖其内部方法实现，我们主要传入的属性大概是边和节点，可以直接同过"."的方法进行属性访问，
节点(路口/交叉点)：
G.nodes(data=True)
是一个一个的元组
# ("12345", {"x": 113.5435, "y": 22.1987, "osmid": 12345, ...})
# ("12346", {"x": 113.5512, "y": 22.2001, "osmid": 12346, ...})
# ...
边：
G.edges(data=True, keys=True)
# ("12345", "12346", "0", {"length": 150.3, "travel_time": 12.5, "highway": "primary", ...})
# ("12345", "12347", "0", {"length": 88.7, "travel_time": 7.1, "highway": "residential", ...})
# ...


"""