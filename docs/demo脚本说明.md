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
