from __future__ import annotations

import os
import random
import subprocess
import time

import carla  # type: ignore

def start_server(
    carla_exe: str,
    port: int = 2000,
    offscreen: bool = False,
) -> subprocess.Popen:
    """启动 CARLA 服务器"""
    if not os.path.isfile(carla_exe):
        raise FileNotFoundError(f"未找到 CARLA 可执行文件: {carla_exe}")

    args = [carla_exe, f"-carla-rpc-port={port}"]
    if offscreen:
        args.append("-RenderOffScreen")

    print(f"正在启动服务器: {carla_exe}")
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def get_initialized_client(host: str = "localhost", port: int = 2000, timeout: float = 20.0) -> carla.Client:
    """初始化并返回一个配置好的 Client
    timeout: 等待服务器启动的超时时间
    """
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    return client

def setup_world(client: carla.Client, map_name: str) -> carla.World:
    """根据指定名称加载地图，支持模糊匹配（如 Town04 匹配到 /Game/Carla/Maps/Town04）"""
    available_maps = client.get_available_maps()
    
    # 查找匹配的完整路径
    full_map_path = None
    for m in available_maps:
        if m.endswith(map_name) or m.split('/')[-1] == map_name:
            full_map_path = m
            break
    
    if not full_map_path:
        short_names = [m.split('/')[-1] for m in available_maps]
        raise ValueError(f"地图 '{map_name}' 不在可用范围内。可选: {short_names}")

    world = client.get_world()
    # 比较时忽略前缀，只看地图名
    current_map_name = world.get_map().name.split('/')[-1]
    
    if current_map_name != map_name:
        print(f"正在加载地图: {full_map_path} ...")
        world = client.load_world(full_map_path)
    else:
        print(f"当前已在地图: {map_name}，跳过加载。")
    
    return world

def spawn_vehicle(
    world: carla.World,
    blueprint_filter: str = "vehicle.tesla.model3",
) -> carla.Vehicle:
    """在随机生成点生成一辆车"""
    blueprint_library = world.get_blueprint_library()
    candidates = blueprint_library.filter(blueprint_filter)
    
    if not candidates:
        print(f"警告: 未找到车型 '{blueprint_filter}'，随机选择车辆。")
        candidates = blueprint_library.filter("vehicle.*")
        
    bp = random.choice(candidates)
    spawn_points = world.get_map().get_spawn_points()
    
    if not spawn_points:
        raise RuntimeError("地图上没有可用的生成点")

    spawn_point = random.choice(spawn_points)
    vehicle = world.try_spawn_actor(bp, spawn_point)
    
    if vehicle is None:
        raise RuntimeError("车辆生成失败（可能生成点已被占用）")
        
    return vehicle

def move_spectator_to_vehicle(world: carla.World, vehicle: carla.Vehicle):
    """将上帝视角移动到车辆后方"""
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    
    backward_vector = transform.get_forward_vector() * -10.0
    up_vector = carla.Location(z=5.0)
    spectator_pos = transform.location + backward_vector + up_vector
    
    spectator_rot = transform.rotation
    spectator_rot.pitch = -30.0
    
    spectator.set_transform(carla.Transform(spectator_pos, spectator_rot))

def main() -> None:
    # --- 配置参数 ---
    CARLA_EXE_PATH = r"D:\Workspace\02_Playground\CARLA_Latest\CarlaUE4.exe"
    TARGET_MAP = "Town04"  # 尝试加载 Town04
    
    server_process = None
    try:
        # 1. 启动服务器并留出更长的加载时间
        server_process = start_server(CARLA_EXE_PATH)
        time.sleep(10) 

        client = get_initialized_client()
        print("可用地图列表:", client.get_available_maps())
        
        # 2. 增强后的地图加载逻辑
        world = setup_world(client, TARGET_MAP)

        # 3. 后续操作
        vehicle = spawn_vehicle(world)
        move_spectator_to_vehicle(world, vehicle)

        print("开始控制车辆...")
        control = carla.VehicleControl(throttle=0.4, steer=0.0)
        vehicle.apply_control(control)
        
        time.sleep(5.0) # 增加观察时间
        
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        print("控制结束。")
        vehicle.destroy()

    except Exception as e:
        print(f"运行过程中出错: {e}")
    finally:
        # 如果需要自动关闭服务器，取消下面注释
        # if server_process: server_process.terminate()
        pass

if __name__ == "__main__":
    main()