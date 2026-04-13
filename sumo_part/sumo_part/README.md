# sumo_part — 澳门交通 OD 矩阵贝叶斯估计系统

## 一、项目目标

本项目的核心目标是：**根据澳门各交叉路口摄像头观测到的分车型交通流量，反推出澳门 23 个区域之间的 OD（Origin-Destination，起讫点）出行矩阵。**

简单来说就是回答这个问题：*"从 A 区到 B 区，每个时段大约有多少辆车/巴士/摩托车/货车在出行？"*

方法是 **贝叶斯推断**：先给一个均匀先验（"不知道具体分布，先假设每对 OD 差不多"），再用摄像头实际观测数据去修正，得到后验 OD 矩阵及其 95% 置信区间。

当前项目的**贝叶斯估计链只使用一套分配矩阵 `H`**：

- `analytic`：原有的 `K` 最短路 + `Logit` 路径选择模型

`SUMO` 在当前版本中不再参与贝叶斯前置 `H` 构建，而是作为**后处理仿真器**：

- 先用 `analytic/logit-H` + 贝叶斯更新得到后验 OD
- 再把后验 OD 写入 SUMO demand，运行 `od2trips + duaIterate.py + sumo`
- 从仿真 route 中重建一份 `SUMO` 产物 `H`，并输出仿真转向统计，作为诊断/对照结果

---

## 二、整体数据流向

```
┌─────────────────────────────────────────────────────────────────────┐
│                        数 据 准 备 阶 段                             │
│                                                                     │
│  macau_drive.graphml ──→ 加载路网图 G (NetworkX MultiDiGraph)        │
│  macau_drive.net.xml ──→ SUMO 路网 (不存在时由 netconvert 生成)      │
│  speed_mapping.csv   ──→ 为每条边计算 travel_time 权重               │
│  macau_zones_23.geojson → 加载 23 个区域多边形                       │
│                          → 计算每个区域的质心坐标                     │
│                          → 把质心映射到路网最近节点 (zone_node_map)   │
│                          → 或在 SUMO backend 中转为 TAZ              │
│  a1_copy_2.json      ──→ 加载摄像头配置 (位置/方向/gate 信息)        │
│                          → 把每个 gate 映射到路网的一条边 (edge)      │
│  time_limit.json     ──→ 加载各摄像头各时段的分车型观测流量           │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     分 配 矩 阵 构 建 (H)                            │
│                                                                     │
│  analytic backend:                                                  │
│    1. 用 K-最短路算法找到 K 条候选路径                                │
│    2. 用 Logit 模型计算每条路径的选择概率                             │
│    3. 遍历路径上的每条边，如果该边有摄像头 gate：                     │
│       H[gate_index, od_index] += 路径概率                            │
│                                                                     │
│  结果: 稀疏矩阵 H (n_gates × n_od_pairs)                            │
│  含义: H[i,j] = "从 OD 对 j 出发的一辆车，经过 gate i 的概率"       │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     贝 叶 斯 推 断                                   │
│                                                                     │
│  已知:                                                              │
│    g     = 摄像头观测向量 (各 gate 的平均车流量)                     │
│    Σ     = 观测噪声协方差矩阵 (对角阵, 方差 = max(观测均值, 1))     │
│    H     = 分配矩阵                                                 │
│    μ₀,V₀ = 均匀先验 (先验均值 = 总流量/OD对数, 方差 = β × 均值)    │
│                                                                     │
│  贝叶斯更新 (三种模式可选):                                          │
│    batch:      一次性矩阵求解  μ₁ = μ₀ + V₀Hᵀ(Σ+HV₀Hᵀ)⁻¹(g-Hμ₀) │
│    sequential: 逐个观测迭代更新                                      │
│    error_free: 假设无观测误差的逐个更新                               │
│                                                                     │
│  输出: 后验 OD 矩阵 (23×23) + 上下界置信区间                        │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   SUMO 后 处 理 仿 真                                │
│                                                                     │
│  1. 将后验 OD 均值矩阵写成 SUMO TAZ demand                          │
│  2. 运行 od2trips + duaIterate.py + sumo                           │
│  3. 从 route 中重建仿真产物 H（不回灌贝叶斯）                       │
│  4. 输出仿真 edge flow / turning / diagnostics                      │
└─────────────────────┬───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        输 出 结 果                                   │
│                                                                     │
│  od_{vehicle}.csv        — 后验 OD 矩阵 (均值)                      │
│  od_{vehicle}_lower.csv  — 95% 置信区间下界                          │
│  od_{vehicle}_upper.csv  — 95% 置信区间上界                          │
│  turning_{vehicle}.json  — 各摄像头路口转向比例统计                   │
│  output/sumo/...         — posterior_sumo 仿真产物                  │
│                                                                     │
│  每种车型 (car/bus/truck/motorcycle) 各生成一套                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 三、目录结构

```
sumo_part/
├── data/                          # 输入数据
│   ├── config/
│   │   └── a1_copy_2.json         # 摄像头配置 (位置、gate、方向)
│   ├── network/
│   │   ├── macau_drive.graphml    # 澳门驾车路网图 (主要数据源)
│   │   └── macau-260220.osm.pbf  # OpenStreetMap 原始数据 (备用)
│   │   └── macau_drive.net.xml   # SUMO 路网（按需生成）
│   ├── observations/
│   │   └── time_limit.json        # 各摄像头各时段的分车型交通观测量
│   ├── speeds/
│   │   └── speed_mapping.csv      # 各道路等级的限速映射表
│   └── zones/
│       ├── macau_zones_23.geojson # 澳门 23 个交通分区边界
│       ├── macau_23_districts_esri.json  # ESRI 格式的分区数据
│       └── zone_centroids.json    # 各区域质心→路网节点映射缓存
│
├── output/                        # 输出结果
│   ├── od_car.csv / od_car_lower.csv / od_car_upper.csv
│   ├── od_bus.csv / od_bus_lower.csv / od_bus_upper.csv
│   ├── od_motorcycle.csv / ...
│   ├── od_truck.csv / ...
│   ├── turning_car.json / turning_bus.json / ...
│   ├── assignment_matrix__k3__theta*__.npz   # 分配矩阵缓存
│   ├── assignment_matrix__k3__theta*__.json   # 分配矩阵元数据
│   └── gate_edge_mapping.json     # gate→路网边映射缓存
│   └── sumo/                      # posterior_sumo 的中间产物和诊断输出
│
└── sumo_part/                     # Python 源码包
    ├── __init__.py                # 公共 API 导出
    ├── __main__.py                # python -m sumo_part 入口
    ├── cli.py                     # 命令行接口
    ├── paths.py                   # 路径管理
    ├── types.py                   # 数据类型定义
    ├── network.py                 # 路网加载与区域映射
    ├── observations.py            # 观测数据解析与处理
    ├── assignment.py              # 分配矩阵构建 (K最短路 + Logit)
    ├── sumo_backend.py            # posterior OD -> SUMO 仿真（TAZ / demand / route / H / turning）
    ├── bayes.py                   # 贝叶斯推断核心算法
    ├── prior.py                   # 先验分布构建
    └── pipeline.py                # 流水线编排 (串联所有步骤)
```

---

## 四、各模块详解

### 4.1 `paths.py` — 路径管理

**职责**: 集中管理项目所有文件路径，避免硬编码。

- `ProjectPaths` 数据类：存储从 `root` 目录派生出的所有输入/输出文件路径
- `from_root(root)`: 工厂方法，给定根目录自动推导所有子路径
- `default_paths()`: 以 `sumo_part/` 目录为根的默认路径

### 4.2 `types.py` — 数据结构定义

**职责**: 定义整个项目流转的核心数据容器。

| 数据类 | 说明 |
|-------|------|
| `GateInfo` | 一个摄像头 gate 的完整信息：所属摄像头 ID、GPS 坐标、来车方向、去车方向、映射到的路网边 |
| `PathCache` | K 最短路径的预计算缓存，用于快速重建不同 θ 下的分配矩阵 |
| `ObservationData` | 一种观测粒度（gate 级/edge 级）下的观测向量 `g`、噪声协方差 `σ`、所涉及的 gate 索引 |
| `CameraTurningObs` | 单个摄像头路口的转向比例观测（各进出方向的流量及比例） |
| `TurningData` | 所有摄像头的转向观测汇总 |
| `PreparedObservations` | 一种车型的全部观测数据打包（gate级 + edge级 + 转向） |

### 4.3 `network.py` — 路网加载与区域映射

**职责**: 把原始地理数据加工成可计算的路网图和区域→节点映射。

**关键流程**:

1. **加载路网图**: 优先读 GraphML 缓存 → 退而求其次读本地 .pbf → 最后可选在线下载
2. **类型修正**: GraphML 加载后所有属性都是字符串，需转换为 float/int（`_coerce_graph_attributes`）
3. **限速赋权**: 读取 `speed_mapping.csv`，按道路等级 (highway tag) 给每条边设置 `speed_kph`，再计算 `travel_time = length / speed`
4. **区域质心映射**: 读取 23 个分区的 GeoJSON → 计算各区的代表点（representative_point）→ 在路网最大强连通分量中找最近节点

**输出**: `(G: MultiDiGraph, zone_node_map: {区域ID → 节点ID})`

### 4.4 `observations.py` — 观测数据解析

**职责**: 把摄像头原始 JSON 数据转化为贝叶斯推断需要的向量/矩阵格式。

**三层加工**:

1. **Gate 级观测** (`build_observation_vector`):
   - 读取 `time_limit.json` 中每个 gate 在多个时段的车流量
   - 对多个时段取**平均值**作为观测值 `g[i]`
   - 噪声方差设为 `max(g[i], 1.0)`（流量越大，噪声越大）

2. **Edge 级聚合** (`aggregate_to_edge_level`):
   - 同一条路网边上可能有多个 gate（同一路口不同方向）
   - 将同一 edge 上的所有 gate 流量**加和**
   - 这是实际喂给贝叶斯更新的观测粒度

3. **转向比例** (`build_turning_observations`):
   - 对每个摄像头路口，按 edge 分组统计各方向流量
   - 计算各方向的流量占比（转向比例）
   - 要求路口至少有 2 个方向才有意义

### 4.5 `assignment.py` — 分配矩阵构建（最核心的模块）

**职责**: 构建分配矩阵 **H**，建立 "OD 流量 → gate 观测" 之间的线性映射关系。

这是整个系统最关键的一步。数学上：**g = H × x + ε**，其中 g 是观测向量，x 是 OD 流量向量，H 描述了 OD 流量如何分配到各个 gate 上。

**关键步骤**:

1. **摄像头→路网边映射** (`map_gates_to_edges`):
   - 找到每个摄像头 GPS 坐标最近的路网节点
   - 获取该节点所有进出边
   - 根据 gate 标注的来车方向（origin_direction），计算**方位角**
   - 找方位角最匹配的边作为该 gate 监控的路网边
   - 匹配容差 90°，超出则用默认边

2. **K 最短路径** (`compute_k_shortest_paths`):
   - 先将 MultiDiGraph 简化为 DiGraph（同向多条边只保留 travel_time 最短的）
   - 对每对 OD，用 NetworkX 的 `shortest_simple_paths` 算法找 K 条（默认 3 条）最短路径

3. **Logit 路径选择模型** (`logit_probabilities`):
   - 对 K 条路径的 travel_time 用 Logit 模型算概率：`P(k) = exp(-θ × cost_k) / Σ exp(-θ × cost_i)`
   - θ 越大，越集中选最短路；θ 越小，分散到多条路
   - 默认 θ = 0.1

4. **填充 H 矩阵**:
   - 对于 OD 对 j 的第 k 条路径，遍历路径上每条边
   - 如果该边有 gate i，则 `H[i, j] += P(k)`
   - 最终 H 是稀疏矩阵 (scipy CSR 格式)

5. **缓存机制**: H 矩阵计算耗时较长，通过 SHA1 哈希（基于 K/θ/gate 配置）生成唯一文件名缓存为 `.npz` 文件

### 4.5.1 `sumo_backend.py` — 后验 OD 的 SUMO 仿真后处理

**职责**: 读取后验 OD（矩阵或 CSV），写成 SUMO demand，运行仿真，并从 route 中提取仿真产物 `H` 与转向统计。

**关键流程**:

1. **SUMO 资产准备**:
   - 校验 `SUMO_HOME`
   - 生成 / 读取 `macau_drive.net.xml`
   - 将 23 个 zone 转为 SUMO `TAZ`
   - 独立完成 `gate -> SUMO from_edge / to_edge` 映射

2. **后验 OD demand 写入**:
   - 按 `zone_ids` 展开 23×23 后验 OD
   - 跳过对角线
   - 将非负浮点值四舍五入为整数 trip count
   - 同时保存 demand metadata（原值 / 舍入值 / OD 顺序）

3. **DUE 运行**:
   - 调 `od2trips`
   - 调 `duaIterate.py`
   - 记录 gap 历史和诊断信息

4. **H 重建**:
   - 从最终 route 文件读取每辆车的路径
   - 统计某个 OD 对的车辆经过每个 gate 所属 `SUMO from_edge` 的比例
   - 该 `H` 仅作为仿真诊断输出，不参与贝叶斯更新

### 4.6 `prior.py` — 先验分布

**职责**: 构建贝叶斯更新的先验分布 (μ₀, V₀)。

- 采用**均匀先验**: 假设每对 OD 的流量均值相同 = 总观测流量 / OD 对数
- 先验协方差为对角阵，方差 = β × 均值（β 默认 100，表示先验很不确定）
- β 越大，先验越弱，观测数据主导后验

### 4.7 `bayes.py` — 贝叶斯推断核心

**职责**: 执行贝叶斯线性高斯模型的后验更新。

**数学模型**: 假设观测模型为 g = Hx + ε，其中 ε ~ N(0, Σ)，先验 x ~ N(μ₀, V₀)

**三种更新模式**:

| 模式 | 方法 | 特点 |
|------|------|------|
| `batch` | 矩阵一次求解 | 数学上最优，需要求解大矩阵 M = Σ + HV₀Hᵀ |
| `sequential` | 逐个观测迭代更新 | 内存友好，每步只处理一个标量观测 |
| `error_free` | 假设 Σ = 0 的逐个更新 | 极端情况，完全信任观测数据 |

**Batch 更新公式**（最常用）:
```
M = Σ + H V₀ Hᵀ
μ₁ = μ₀ + V₀ Hᵀ M⁻¹ (g - H μ₀)
V₁ = V₀ - V₀ Hᵀ M⁻¹ H V₀
```

**后处理**:
- 负值截断为 0（流量不能为负）
- 计算 95% 置信区间
- 将一维向量 reshape 回 23×23 的 OD 矩阵
- 计算变异系数 (CV) 衡量估计不确定性

### 4.8 `pipeline.py` — 流水线编排

**职责**: 将上述所有模块串联成完整的流水线。

提供两种运行方式:
- `run_vehicle_pipeline(vehicle_type)`: 运行单一车型的完整流程，并自动追加 posterior SUMO 仿真
- `run_all()`: 依次运行所有 4 种车型（共享 analytic H，只有观测不同），并自动追加 posterior SUMO 仿真

同时负责结果的落盘:
- `save_od_matrix()`: 将 OD 矩阵保存为 CSV（行列标题为区域 ID）
- `save_turning_summary()`: 将转向比例保存为 JSON
- `simulate_posterior_sumo()`: 将后验 OD 送入 SUMO，保存仿真产物

### 4.9 `cli.py` — 命令行接口

**职责**: 提供命令行入口，支持分步执行或一键运行。

```bash
# 加载并检查路网
python -m sumo_part prepare-network

# 查看某车型的观测数据统计
python -m sumo_part prepare-observations --vehicle car

# 准备 SUMO-native 资产（posterior_sumo 会复用）
python -m sumo_part prepare-sumo-assets --vehicle car

# 构建 analytic 分配矩阵
python -m sumo_part build-h --K 3 --theta 0.1

# 估计单一车型的 OD 矩阵
python -m sumo_part estimate-od --vehicle car --mode batch

# 一键运行所有车型，并在每个车型完成后自动跑 posterior SUMO
python -m sumo_part run-all --K 3 --theta 0.1 --beta 100 --mode batch
```

---

## 五、关键概念解释

### 5.1 什么是 Gate？

Gate 是摄像头上定义的一个虚拟检测线。一个摄像头路口可以有多个 gate，每个 gate 对应一个方向的车流。例如一个十字路口的摄像头可能有 4 个 gate，分别检测东→西、西→东、南→北、北→南的车流。

### 5.2 什么是分配矩阵 H？

H 矩阵是连接 "未知 OD 流量" 和 "可观测 gate 流量" 的桥梁：

- H 的每一行对应一个 gate（观测点）
- H 的每一列对应一对 OD（如 区域3→区域7）
- H[i,j] 的值 = 从 OD 对 j 出发的车辆经过 gate i 的概率

这个概率通过 "K 最短路径 + Logit 选择模型" 计算得到。

### 5.3 为什么用贝叶斯方法？

因为这是一个**不适定问题**（underdetermined）：
- 观测点（gate 数量）远少于未知数（506 个 OD 对）
- 直接求解 g = Hx 有无穷多解
- 贝叶斯方法通过先验分布正则化，给出唯一且合理的后验估计，并量化不确定性

### 5.4 θ (theta) 参数的意义

θ 控制 Logit 模型中驾驶员对最短路径的敏感度：
- **θ → 0**: 所有路径等概率（驾驶员完全随机选路）
- **θ → ∞**: 只走最短路径（驾驶员完全理性）
- **θ = 0.1**（默认）: 较弱的偏好，路径之间概率差异不大

### 5.5 β (beta) 参数的意义

β 控制先验的松紧度：
- **β 很大**（如 100）: 先验很弱，后验主要由观测数据决定
- **β 很小**（如 1）: 先验较强，后验受先验影响较大

---

## 六、输出文件说明

### OD 矩阵 CSV (`od_{vehicle}.csv`)
- 23×23 矩阵，行 = 起点区域，列 = 终点区域
- 值 = 该时段估计的平均车流量
- 对角线为 0（不统计区内出行）
- `_lower.csv` / `_upper.csv` 分别为 95% 置信区间的下界和上界

### 转向统计 JSON (`turning_{vehicle}.json`)
- 按摄像头汇总各方向的流量和比例
- `observed_counts`: 各方向绝对流量
- `observed_proportions`: 各方向流量占比（和为 1）
- `edge_keys`: 对应的路网边 (u, v) 对

### posterior SUMO 仿真产物 (`output/sumo/<period>/<vehicle>/`)
- `posterior_od_demand__{vehicle}.xml/.json`: 后验 OD 写成的 SUMO demand 及其 metadata
- `routes_final__posterior_sumo__{vehicle}.rou.xml`: 仿真最终 route
- `assignment_matrix__backend-posterior_sumo__...npz/.json`: 从 route 重建的仿真产物 H
- `turning_{vehicle}__posterior_sumo.json`: 仿真转向结果（包含 movement 明细和 camera 摘要）
- `posterior_sumo_artifacts__{vehicle}.json`: SUMO 工具、文件、gap 和诊断信息

### 分配矩阵缓存 (`.npz` + `.json`)
- `.npz`: scipy 稀疏矩阵格式，H 矩阵本体
- `.json`: 元数据（K、θ、gate 签名、OD 对列表），用于缓存校验
- 文件名包含 K 值、θ 值和配置哈希，参数变化自动生成新缓存

---

## 七、典型执行流程

```python
from sumo_part import run_all

# 一行代码跑完全部
results = run_all(K=3, theta=0.1, beta=100.0, mode="batch")

# results["car"].od_matrix  →  23×23 numpy 数组
# results["car"].confidence_intervals_95  →  506×2 数组 (下界, 上界)
# results["car"].info["posterior_total_flow"]  →  后验总流量
# results["car"].info["mean_cv"]  →  平均变异系数 (越小越确定)
# 同时会在 output/sumo/... 下自动生成 posterior SUMO 仿真产物
```

内部执行顺序:
1. `prepare_network()` → 加载路网 G + 区域映射 zone_node_map
2. `prepare_gates()` → 加载摄像头 gate 并映射到路网边
3. `build_assignment()` → 构建分配矩阵 H（或从缓存读取）
4. 对每种车型:
   - `prepare_observations()` → 解析该车型的观测数据
   - `estimate_posterior_od()` → 贝叶斯更新得到后验 OD
   - `save_od_matrix()` → 保存 CSV
   - `save_turning_summary()` → 保存转向 JSON
   - `simulate_posterior_sumo()` → 将后验 OD 写入 SUMO demand 并输出仿真 H / turning
