# Windows11 + CARLA：VLN/UGV 初步开发工程规格（给 Codex）

> 目标：在 Windows11 上用 CARLA 跑通一个“语言条件导航（Vision-Language Navigation / VLN）”最小闭环：  
> 指令（language）→ 感知/地图（perception+map）→ 规划（planning）→ 控制（control）→ 评测（metrics）→ 数据记录（logging）  
> 先做 **Urban/道路场景**（CARLA擅长），后续可扩展到“非结构化道路/越野”（需要其他仿真器如 AirSim/IsaacLab）。

---

## 0. 约束与原则（必须遵守）

- 纯仿真（Simulator-based），不涉及实体机器人。
- 本地调试：单卡/低并发；服务器冲刺：多卡短时训练（如果有）。
- 避免从零预训练大模型：优先 **预提特征** / **LoRA/Adapter** / **离线缓存**。
- “可通行性/风险代价”是一等公民：不仅到达目标，还要衡量安全与代价。

---

## 1. 项目里程碑（按阶段交付）

### Phase 0（1-2 周）：CARLA 闭环跑通
- [ ] 启动 CARLA Server（Windows11）+ Python Client 连接
- [ ] 能 spawn 车辆与传感器（RGB/Depth/Seg 可选）
- [ ] 实现 reset/step，能执行动作（throttle/steer/brake）并拿到观测
- [ ] 基础碰撞检测、回合结束条件、轨迹记录（位置/速度/碰撞/红灯等）

### Phase 1（2-4 周）：最小任务 + 强基线（不学习也能跑）
- [ ] 任务：给定语言指令（先模板化）+ 目标点/目标区域
- [ ] 基线A：使用 CARLA 原生 Route Planner（或 map waypoint）跟踪到目标
- [ ] 基线B：自研“代价图 + A*”输出 waypoints，低层 PID 跟踪
- [ ] 输出统一评测：SR / SPL + 安全/违规指标

### Phase 2（可选）：离线伪标注 → 蒸馏轻量可通行性（偏离线）
- [ ] 离线跑重模型（例如 open-vocab 分割）产出地形/可行驶 mask（或直接用 CARLA GT 语义）
- [ ] 训练轻量分割/可通行性网络（在线用）
- [ ] 验证在线速度（例如每 N 帧更新一次）

### Phase 3（关键）：语言条件风险代价图
- [ ] 设计语言模板（如：avoid mud / stay on road / avoid pedestrian / minimize lane invasion / prefer right lane）
- [ ] 把语言映射为代价权重（rule-based 或 小网络/LoRA）
- [ ] 规划器使用“risk-aware cost”生成轨迹/waypoints

### Phase 4：端到端整合 + 对比实验
- [ ] 对比：无风险代价 / 有语义无风险 / 有语义+风险（主线）
- [ ] 指标：SR、SPL、Risk-weighted SPL、碰撞率、卡死率、违规率（红灯/越线/逆行等）

### Phase 5（短冲刺可选）：少量 RL/IL 精调
- [ ] 只微调：高层子目标选择器 or 价值头（PPO/DD-PPO/BC）
- [ ] 严格控制训练成本：短 horizon + 小并发 + checkpoint 可中断

---

## 2. 任务定义（CARLA Urban VLN v0）

### 2.1 Episode 输入
- `instruction: str`  
- `start: {x,y,z,yaw}`（可选，若不固定则随机）
- `goal: {x,y,z,radius}`（目标区域半径用于成功判定）
- `preferences: dict`（可从 instruction 解析得到，例如 `{"avoid_lane_invasion": 1.0, "avoid_collision": 5.0}`）

### 2.2 Episode 终止条件
- Success：到达 goal 区域（distance <= radius），且（可选）无严重违规
- Fail：
  - 超时：step >= max_steps
  - 严重碰撞：collision_intensity >= threshold
  - 卡死：速度长期过低且不在 goal 附近（stuck）
  - 严重违规（可选）：逆行/连续闯红灯/越界过多等

---

## 3. 观测与动作空间（统一 Env API）

### 3.1 Observation（字典）
最小实现（先做这些）：
- `rgb: np.uint8[H,W,3]`
- `speed: float`
- `ego_pose: {x,y,z,yaw}`（v0 建议用 GT pose）
- `route_hint: optional`（来自 CARLA map/route planner 的 next waypoint / lane centerline）
- `instruction: str`（原始指令，或 tokenized）

可扩展：
- `depth`
- `semantic: int[H,W]`（CARLA 语义标签）
- `bev_costmap: float[C,Hm,Wm]`（在线地图输出）

### 3.2 Action（两种模式二选一，建议都支持）

**Mode A：低层控制（最易在 CARLA 跑通）**
- `action = {throttle: float[0,1], steer: float[-1,1], brake: float[0,1]}`

**Mode B：中层 waypoint（推荐长期架构）**
- `action = {target_waypoint: {x,y,z}, target_speed: float}`  
由低层控制器将 waypoint 转成 throttle/steer/brake

---

## 4. 系统结构（建议实现的模块边界）

> 核心思想：分层结构  
> 高层：地图/代价图上选子目标（waypoint）  
> 低层：车辆动力学约束执行（PID/MPC 或轻量策略）

### 4.1 `carla_env/`
- `CarlaEnv.reset(seed, episode_spec) -> obs`
- `CarlaEnv.step(action) -> (obs, reward, terminated, truncated, info)`
- 负责：连接、同步 tick、spawn、传感器回调、world settings、清理 actors

### 4.2 `instruction/`
- `parse_instruction(text) -> InstructionSpec`
  - 输出结构化偏好（代价权重）、目标描述（可先不用）、子任务（可选）
- v0 推荐：模板指令 + 规则解析（不要一开始做 LLM）

### 4.3 `perception/`
- `DrivableEstimator.predict(rgb|semantic) -> drivable_mask`
- `RiskEstimator.predict(rgb|semantic, instruction_spec) -> risk_layers`
  - risk_layers 示例：lane_invasion_risk、collision_risk_proxy、red_light_risk_proxy、offroad_risk 等
- v0 推荐：直接用 CARLA 语义 GT 构建“可行驶区域/车道/人车”mask，先别做重分割

### 4.4 `mapping/`
- `Costmap.update(ego_pose, drivable_mask, risk_layers) -> bev_costmap`
- 表示：在线栅格地图（Hm x Wm），通道包含：
  - drivable_prob
  - risk_cost (language-conditioned)
  - (可选) semantic_class distribution
- v0：先做“局部代价图”（以车为中心的局部 BEV），不做全局 SLAM

### 4.5 `planner/`
- `WaypointPlanner.plan(ego_pose, goal, bev_costmap, instruction_spec) -> List[waypoints]`
- v0 两条路线：
  1) **Baseline**：CARLA GlobalRoutePlanner 产出路线点
  2) **Risk-aware**：A* / D* 在 costmap 上搜一段局部轨迹（rolling horizon）
- 输出 waypoints 给 controller

### 4.6 `controller/`
- `PIDController.track(ego_state, waypoint, target_speed) -> low_level_action`
- （可选）`MPCController`：更稳但实现成本高

### 4.7 `agent/`
- `BaseAgent.act(obs) -> action`
- 提供三种 agent：
  1) `RouteFollowerAgent`：纯路线跟随
  2) `CostmapAStarAgent`：costmap+A*
  3) `LearningAgent`（占位）：BC/PPO 只学高层选择或价值头

### 4.8 `logging/`
- 每回合落盘：
  - episode_spec.json
  - step_log.parquet（每 step 的 pose/speed/action/violations/collision 等）
  - video.mp4（可选）
  - costmap_debug.png（可选）

---

## 5. 评测指标（必须实现）

基础：
- `SR`（成功率）
- `SPL`（成功加权路径效率）

安全与代价（CARLA 版本）：
- `collision_rate`（碰撞次数/回合 或 是否碰撞）
- `stuck_rate`（卡死回合比例）
- `lane_invasion_count`（越线次数）
- `red_light_violations`（闯红灯次数）
- `offroad_ratio`（轮胎在非道路区域时间占比，可用语义/车道判断）

风险加权（核心）：
- `risk_cost = Σ_t ( w_collision*I_collision + w_lane*I_lane + w_red*I_red + w_offroad*I_offroad + ... )`
- `Risk-weighted SPL`：把 SPL 用 `(1 + risk_cost)` 或 `exp(-risk_cost)` 进行惩罚（具体函数写成可配置）

---

## 6. 训练/学习（可选但要留好接口）

### 6.1 数据生成（BC）
- 用 Baseline planner/controller 生成 expert 轨迹（waypoints 或 low-level actions）
- 训练 `LearningAgent` 模仿（优先学高层 waypoint 或 cost 权重映射）

### 6.2 少量 RL 精调（短冲刺）
- 只更新小模块（value head / waypoint selector / language-to-cost mapper）
- 奖励建议：
  - 到达奖励 + 路径效率
  - 大惩罚：碰撞、闯红灯、越线、offroad、卡死
  - 软惩罚：risk_cost

---

## 7. 推荐仓库结构（Codex 按此生成代码）

vln_carla/
README.md
SPEC.md
pyproject.toml
configs/
env.yaml
agent_baseline.yaml
agent_risk_aware.yaml
carla_env/
env.py
sensors.py
utils.py
instruction/
parser.py
templates.py
perception/
drivable.py
risk.py
mapping/
costmap.py
bev.py
planner/
route_planner.py
astar.py
controller/
pid.py
agent/
base.py
route_follower.py
costmap_astar.py
learning_agent.py
eval/
metrics.py
evaluator.py
logging/
recorder.py
scripts/
run_episode.py
collect_dataset.py
train_bc.py
train_rl.py


---

## 8. 止损条件（防止陷入无效迭代）

- 如果在线地图/定位导致失败率 >30%，v0 直接使用 GT pose 或简化里程计，不要先做 SLAM。
- 如果单回合吞吐 < 1 FPS：必须离线化重计算（分割/评估器）、降低分辨率、降低地图更新频率。
- 如果加入 risk-aware 后在典型场景上 Risk-weighted SPL 提升 < 5% 且碰撞/违规无明显下降：回退并检查代价定义与解析逻辑。

---

## 9. Codex 生成代码的优先级（照这个顺序实现）

1) `carla_env`：连接/同步/传感器/step-reset/碰撞与越线事件
2) `eval.metrics`：SR/SPL + collision/lane/red/offroad/stuck
3) `controller.pid` + `planner.route_planner`：baseline 跑通
4) `mapping.costmap` + `planner.astar`：risk-aware 路线
5) `instruction.parser`：语言→代价权重（先规则）
6) （可选）数据采集与 BC 训练脚本
