# Episode 功能与测试说明

本文说明当前 `Episode` 相关实现的作用、运行方式与结果解读，覆盖：

1. 第 2 层：CARLA 联调冒烟（`demo/smoke_carla_episode_detour.py`）
2. 第 3 层：自动化集成测试（`tests/integration/test_carla_episode_integration.py`）

## 1. Episode 的目标与作用

当前实现把一次驾驶任务抽象为 `EpisodeSpec`，通过 `RunEpisodeUseCase` 驱动环境 `reset/step` 并产出 `EpisodeResult`。

主要价值：

1. 明确起点/终点（`start/goal`）与终止语义（`termination_reason`）
2. 统一收集关键安全事件（碰撞、压线、违规）
3. 输出可比较的统计结果（步数、事件计数、终止原因等）

## 2. 第 2 层：冒烟脚本怎么跑

脚本：`demo/smoke_carla_episode_detour.py`

示例命令：

```powershell
# --provoke none 没有任何干扰
conda activate vln_carla_ue5_py312
python demo/smoke_carla_episode_detour.py --host 127.0.0.1 --port 2000 --steps 80 --provoke none

# --provoke lane
# 在后半程强行左右打方向，让车更容易压线
python demo/smoke_carla_episode_detour.py --host 127.0.0.1 --port 2000 --steps 80 --provoke lane

# --provoke collision
# 按 scenario json 里定义的坐标 spawn 障碍物，制造必撞场景
python demo/smoke_carla_episode_detour.py --host 127.0.0.1 --port 2000 --steps 80 --provoke collision
```

主要行为：

1. 读取 `tests/fixtures/scenarios/construction_detour_001.json` 作为金标 episode
2. `reset(spec)` 后校验车辆位姿是否接近 start
3. 运行 50~200 step（默认恒油门 `SimpleAgent`）
4. 输出每步关键统计，观察计数与终止状态
5. 视角采用“高空竖直向下跟随”（pitch=-90，按 ego XY 跟随）

## 3. 第 3 层：集成测试怎么跑

测试：`tests/integration/test_carla_episode_integration.py`

示例命令：

```powershell
conda activate vln_carla_ue5_py312
$env:CARLA_SERVER_HOST='127.0.0.1'
$env:CARLA_SERVER_PORT='2000'
python -m pytest -m integration -q
```

测试启动条件：

1. 必须设置 `CARLA_SERVER_HOST` 和 `CARLA_SERVER_PORT`
2. 必须能 `import carla`
3. 必须能连接 CARLA server

否则测试会 `skip`（这是预期设计，避免在无仿真环境时误报失败）。

## 4. Step 输出行含义

脚本每步输出格式：

```text
[step] 001 0 0 0 ONGOING 1.958 16.001
```

列含义（按顺序）：

1. `step_index`：当前 step 编号（从 1 开始）
2. `collision_count`：累计碰撞次数
3. `lane_invasion_count`：累计压线次数
4. `violation_count`：累计违规次数（当前口径：`lane_invasion_count + red_light_violation_count`）
5. `termination_reason`：当前主终止原因
6. `speed_mps`：当前速度（米/秒）
7. `distance_to_goal_m`：到目标点距离（米）

## 5. TerminationReason 枚举与含义

当前可见值（`src/usecases/episode_types.py`）：

1. `ONGOING`：仍在进行，未触发终止
2. `SUCCESS`：达到目标区域
3. `TIMEOUT`：超过最大步数
4. `COLLISION`：发生碰撞
5. `VIOLATION`：发生违规（如压线/红灯）
6. `STUCK`：长时间低速卡滞
7. `ERROR`：运行异常或外部中断

## 6. 碰撞/压线/违规如何检测

实现位置：`src/infrastructure/carla/carla_env_adapter.py`

1. 碰撞：`sensor.other.collision` 回调累计 `collision_count`
2. 压线：`sensor.other.lane_invasion` 回调累计 `lane_invasion_count`
3. 红灯：当前为 stub（`_count_red_light_violations_stub`），默认返回 0
4. 违规计数：`violation_count = lane_invasion_count + red_light_violation_count`

说明：

1. 当前版本的“违规”不包含碰撞，碰撞单独计数
2. 终止主原因会按优先级选取，且保留 `termination_reasons` 列表用于分析

## 7. `demo` 与 `integration` 的区别（为什么是不同层级）

`demo/smoke_carla_episode_detour.py` 和 `tests/integration/test_carla_episode_integration.py` 都会连 CARLA、跑 episode，看起来相似，但定位不同，因此属于不同测试层级。

### 7.1 角色定位不同

1. `demo/smoke_carla_episode_detour.py`：
   - 目标是“联调冒烟/人工观察”。
   - 强调可视化与交互可控：支持 `--provoke none/lane/collision`，打印每步详细 telemetry。
   - 适合开发阶段快速排查：起点对齐、事件回调是否触发、视角是否正常。
2. `tests/integration/test_carla_episode_integration.py`：
   - 目标是“自动化回归保护”。
   - 强调稳定可重复、可在 CI 或批量脚本里执行。
   - 断言聚焦接口契约：能跑完短 episode，返回 `termination_reason`，结果结构完整。

### 7.2 通过标准不同

1. `demo`（第 2 层）通过标准：
   - reset 位姿接近金标 start。
   - 跑够指定步数或按预期提前终止。
   - 在指定 provoke 模式下，相关计数确实变化（例如 `lane_invasion_count` 增长）。
2. `integration`（第 3 层）通过标准：
   - 用固定场景可以稳定运行并返回 `EpisodeResult`。
   - `termination_reason` 非空且有效。
   - `reset_info`、`step_log`、`metrics` 等核心字段满足契约。

### 7.3 为什么同样“跑了一次车”，却不算同一层

1. 第 2 层回答的是“系统现在能不能动起来、能不能看见和验证关键行为”。
2. 第 3 层回答的是“以后改代码会不会把关键契约改坏”。
3. 前者偏联调诊断，后者偏回归防线；两者互补，不可互相替代。

### 7.4 结果解读建议

1. `demo` 失败但 `integration` 通过：
   - 常见于 provoke 策略不稳定、场景偶发扰动、手工联调条件变化。
   - 先看运行日志与可视化行为。
2. `integration` 失败：
   - 优先视为真实回归风险，先修复契约破坏。
3. `integration skipped`：
   - 通常是未设置 `CARLA_SERVER_HOST/CARLA_SERVER_PORT`，或 CARLA 不可连接，不代表功能错误。
