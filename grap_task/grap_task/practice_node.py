#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 这个代码用于实现抓取水瓶完成放置功能的节点
import rclpy
import threading
import time
from rclpy.executors import MultiThreadedExecutor

from grap_task.server import ChassisTCPServer
from grap_task.nav2_navigator import Nav2Navigator
from grap_task.robot_control import RobotController
from grap_task.goal_manager import GoalManager   # ✅ 只用这个

import logging
from pathlib import Path

def setup_logger(log_path="app.log", level=logging.INFO):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("myapp")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:  # 防止重复添加，避免同一条写多次
        return logger

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s [%(filename)s:%(lineno)d]: %(message)s"
    )
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

logger = setup_logger("logs/app.log")  # 建议放项目 logs 目录


def main(args=None):
    # 1. 先启动 TCP 服务器在线程里
    tcpserver = ChassisTCPServer(host="0.0.0.0", port=8080)
    server_thread = threading.Thread(
        target=tcpserver.start,
        daemon=True
    )
    server_thread.start()
    logger.info("TCP 服务器已启动，监听 0.0.0.0:8080，等待客户端命令...")

    # 2. 初始化 ROS2
    rclpy.init(args=args)

    # 负责底盘开环控制
    robotcontroller = RobotController()
    # 负责 Nav2 导航
    navigator = Nav2Navigator()
    # 负责从 JSON 加载导航点（纯模块，不是节点）
    goals = GoalManager("/home/rpp/wzh/nav_ws/src/grap_task/goals/goals.json")

    # 3. Executor 同时管理两个 ROS 节点
    executor = MultiThreadedExecutor()
    executor.add_node(robotcontroller)
    executor.add_node(navigator)

    try:
        while rclpy.ok():
            # 处理 ROS 回调（Nav2 action feedback、TF 等）
            executor.spin_once(timeout_sec=0.1)

            # 轮询 TCP 最新命令
            cmd = tcpserver.cmd
            if cmd is None:
                time.sleep(0.1)
                continue

            print(f"[MAIN] 收到 TCP 命令：{cmd}")
            tcpserver.cmd = None

            # 3.1 先让机器人后退一下（可按需注释）
            robotcontroller.get_logger().info(f"收到命令 {cmd}，先后退 0.2 m")
            robotcontroller.move_straight(distance=-0.3, speed=0.2)

            # 3.2 根据命令名称构造 Nav2 目标点 Pose
            stamp = navigator.get_clock().now().to_msg()
            pose = goals.build_pose_from_name(cmd, stamp)
            if pose is None:
                navigator.get_logger().warn(f"未找到名称为 '{cmd}' 的导航点，跳过本次命令")
                continue

            navigator.get_logger().info(
                f"开始发送导航目标点 '{cmd}'："
                f"x={pose.pose.position.x:.3f}, "
                f"y={pose.pose.position.y:.3f}"
            )

            # 3.3 同步等待导航结果
            status = navigator.send_goal_and_wait(pose)

            # 3.4 导航成功后通过 TCP 发送 "ok"
            if status == 4:  # 一般 Nav2 SUCCEEDED 是 4
                robotcontroller.move_straight(distance=0.35, speed=0.15)
                navigator.get_logger().info("导航成功，准备给客户端发送 'ok'")
                if tcpserver.parser is not None:
                    response = {"command": "ok"}
                    tcpserver.parser.send_response(resp_dict=response)
                else:
                    navigator.get_logger().warn("无法发送响应：server.parser 为 None")
            else:
                navigator.get_logger().warn(f"导航失败或被取消，status={status}")

    except KeyboardInterrupt:
        print("收到 Ctrl+C，准备退出...")
    finally:
        executor.shutdown()
        navigator.destroy_node()
        robotcontroller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
