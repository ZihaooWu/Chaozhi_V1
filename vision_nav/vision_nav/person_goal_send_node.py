#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PersonGoalPublisher 节点

功能：
 - 使用 PersonTrackerReID 获取当前单个行人的 3D 位置（相机坐标系）
 - 将该位置封装为 PoseStamped 消息
 - 发布到 /person_goal 话题，供 Nav2 或上层逻辑使用
"""
import rclpy.time
import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from rclpy.executors import MultiThreadedExecutor
# 使用你的 ReID 跟踪模块
# from vision_nav.person_tracker_reid import PersonTrackerReID
from vision_nav.person_tracker_multi_reid import PersonTrackerReID
from vision_nav.robot_control import RobotController

class PersonGoalPublisher(Node):
    def __init__(self):
        super().__init__('person_goal_publisher')

        # ---------------- ROS 参数 ----------------
        self.declare_parameter('publish_rate', 1.0)         # Hz
        self.declare_parameter('frame_id', 'base_link')     # PoseStamped.header.frame_id
        self.declare_parameter('show_window', True)         # 是否显示检测窗口
        self.declare_parameter('yolo_weights', 'weights/yolo11n.pt')
        self.declare_parameter('tracker_cfg', 'botsort.yaml')
        self.declare_parameter('sim_threshold', 0.7)
        self.declare_parameter('conf', 0.7)
        self.declare_parameter('iou', 0.7)
        self.declare_parameter('device', 'cuda')
        
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        publish_rate   = float(self.get_parameter('publish_rate').value)
        self.frame_id  = self.get_parameter('frame_id').value
        show_window    = bool(self.get_parameter('show_window').value)
        yolo_weights   = self.get_parameter('yolo_weights').value
        tracker_cfg    = self.get_parameter('tracker_cfg').value
        sim_threshold  = float(self.get_parameter('sim_threshold').value)
        conf           = float(self.get_parameter('conf').value)
        iou            = float(self.get_parameter('iou').value)
        device         = self.get_parameter('device').value

        # ---------------- 创建 ReID 跟踪器 ----------------
        # 说明：PersonTrackerReID 是一个独立的后台线程模块，
        #       负责 RealSense 采集 + YOLO + BoT-SORT + ReID。
        self.get_logger().info(
            f"初始化 PersonTrackerReID: weights={yolo_weights}, "
            f"tracker_cfg={tracker_cfg}, conf={conf}, iou={iou}, "
            f"sim_th={sim_threshold}, device={device}, show_window={show_window}"
        )

        self.tracker = PersonTrackerReID(
            weights=yolo_weights,
            tracker_cfg=tracker_cfg,
            conf=conf,
            iou=iou,
            sim_threshold=sim_threshold,
            show_window=show_window,
            device=device,
        )
        self.tracker.start()
        # ---------------- ROS 发布器 & 定时器 ----------------
        self.person_goal_pub = self.create_publisher(
            PoseStamped,
            '/person_goal',
            10
        )

        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f"PersonGoalPublisher 启动，发布话题 /person_goal，"
            f"frame_id={self.frame_id}, rate={publish_rate}Hz"
        )
    def find_person(self, angular_speed=0.5):
        if angular_speed <= 0.0:
            return
        twist = Twist()
        twist.angular.z = -abs(angular_speed)
        self.cmd_pub.publish(twist)

    # -------------------------------------------------------
    # 定时回调：发布当前目标行人的 PoseStamped
    # -------------------------------------------------------
    def timer_callback(self):
        """
        周期性回调：读取 PersonTrackerReID 状态并发布 PoseStamped
        """
        # 未进行行人注册
        if not self.tracker.has_enrolled:    
            return

        # 1) 已有行人，但丢失则旋转
        if not self.tracker.person_flag:
            self.find_person()
        # 多旋转几次
        for i in range(0, 5):
            i=i+1
            self.find_person()
        # 2) 相机坐标系下的 (X, Y, Z)
        pos = self.tracker.get_person_position()
        if pos is None:
            return

        X, Y, Z = pos  # X: 向右；Y: 向下；Z: 向前（相机坐标系）

        # 3) 简单转换到 base_link（按你的约定：px 前、py 左）
        px = Z        # 前
        py = -X       # 左

        # 机器人目标位姿：站在人后方 1m 处（x = px - 1.0）
        goal_x = px - 1.0
        goal_y = py

        # 4) 填充 PoseStamped 消息
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.pose.position.x = float(goal_x)
        msg.pose.position.y = float(goal_y)
        msg.pose.position.z = 0.0

        # 5) 朝向行人方向：yaw = atan2(py, px)
        yaw = math.atan2(py, px)
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)

        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        # 6) 发布
        self.person_goal_pub.publish(msg)

        self.get_logger().info(
            f"发布 /person_goal: frame={self.frame_id}, "
            f"X={msg.pose.position.x:.2f}, Y={msg.pose.position.y:.2f}, "
            f"yaw={yaw:.2f} rad"
        )

    # -------------------------------------------------------
    # 生命周期管理：节点销毁时停止跟踪线程
    # -------------------------------------------------------
    def destroy_node(self):
        """
        重载 destroy_node，在节点销毁时顺便停掉 tracker 线程
        """
        try:
            if hasattr(self, 'tracker') and self.tracker is not None:
                self.get_logger().info("停止 PersonTrackerReID 线程...")
                self.tracker.stop()
        except Exception as e:
            self.get_logger().warn(f"停止 PersonTrackerReID 时出错: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PersonGoalPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C, 退出 PersonGoalPublisher")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
