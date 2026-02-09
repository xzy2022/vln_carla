# Demo 脚本说明

本文档用于简要记录 `demo/` 目录脚本的用途与常用参数。

## fixed_spectator_axes_spawn.py

作用（UE5）：
- 固定 `spectator` 到 `(0, 0, 10)` 俯视视角。
- 在世界原点绘制 UE 坐标轴（X/Y/Z）。
- 在指定坐标生成一个测试静态物体（默认 `static.prop.barrel`，自动尝试无碰撞 `z`）。

基本运行：

```powershell
python demo/fixed_spectator_axes_spawn.py
```

常用参数：
- `--spawn-x` / `--spawn-y`：指定生成点平面坐标。
- `--prop-blueprint`：指定要生成的物体蓝图（如 `static.prop.trafficcone01`）。
- `--spawn-min-z` / `--spawn-max-z`：限制生成高度搜索范围（`spawn-max-z < 10`）。
- `--destroy-on-exit`：退出脚本时销毁本次生成物体。

## spectator_coordinate_navigator.py

作用（UE5）：
- 连接到正在运行的 CARLA（UE5）并将 `spectator` 固定为俯视朝下。
- 支持指定目标地图（`Mine_01` / `Town10HD_Opt`），若与当前地图不一致会自动切换并在新地图继续执行。
- 使用键盘实时移动观察点：方向键控制 XY 平面，`U/O` 控制 Z 高度。
- 持续在世界原点绘制 UE 坐标轴，并在移动时输出当前坐标。

基本运行：

```powershell
python demo/spectator_coordinate_navigator.py
```

常用参数：
- `--map`：指定地图，UE5 仅支持 `Mine_01` 与 `Town10HD_Opt`（默认 `Town10HD_Opt`）。
- `--speed`：各方向移动速度（m/s）。
- `--tick-hz`：控制循环频率。
- `--axis-length` / `--axis-z-offset`：坐标轴显示长度与抬升高度。
- `--redraw-interval` / `--life-time`：坐标轴重绘周期与生命周期。

## ue5_spawn_vehicle_speed_profile_log.py

作用（UE5）：
- 在 `Town10HD_Opt` 指定点（默认 `(0.038, 15.320, 0.15, yaw=180)`）生成一辆车。
- 按速度分段驱动车辆：`0-2s` 为 `1.0m/s`，`2-5s` 为 `1.5m/s`，随后停止。
- 支持切换多种控制方式用于效果对比：`constant_velocity`、`target_velocity`、`vehicle_control`、`ackermann`。
- 启用俯视跟车观察视角（`spectator` 跟随车辆 XY，固定顶视角高度）。
- 将车辆真值速度（`get_velocity()` 计算标量速度）追加写入 `tmp/` 日志文件。

基本运行：

```powershell
python demo/ue5_spawn_vehicle_speed_profile_log.py --control-mode constant_velocity
```

常用参数：
- `--control-mode`：控制方式，支持 `constant_velocity` / `target_velocity` / `vehicle_control` / `ackermann`。
- `--spawn-x` / `--spawn-y` / `--spawn-z` / `--spawn-yaw`：车辆生成坐标与朝向。
- `--spectator-z` / `--spectator-yaw`：俯视跟车相机高度与朝向。
- `--tick-hz`：控制与采样循环频率。
- `--log-path`：日志输出路径（默认 `tmp/town10hd_opt_speed_truth.log`，追加写入）。
- `--destroy-on-exit`：退出脚本时销毁本次生成车辆。

当前的测试结果表面，选用`target_velocity`最好，因为它不仅在速度变化时追踪迅速，停止命令也相应最快。不过全部方法的真实速度都和命令
速度有不同。