#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 机器人的基础控制节点模块

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Quaternion
import rclpy.time

def normalize_angle(angle):
    """将角度归一化到 [-pi, pi]。"""
    a = math.fmod(angle + math.pi, 2.0 * math.pi)
    if a < 0.0:
        a += 2.0 * math.pi
    return a - math.pi

def quaternion_to_yaw(q):
    """从四元数计算 yaw 角（绕 z 轴）."""
    # q: geometry_msgs.msg.Quaternion
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class RobotController(Node):
    def __init__(self,
                 cmd_vel_topic: str = 'cmd_vel',
                 map_frame: str = 'map',
                 base_frame: str = 'base_link'):
        super().__init__('robot_controller')

        self.map_frame = map_frame
        self.base_frame = base_frame

        # 发布速度控制
        self.cmd_pub = self.create_publisher(Twist, cmd_vel_topic, 10)

        # TF 缓存与监听
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f'RobotController 初始化完成，使用 cmd_vel: {cmd_vel_topic}, '
            f'坐标系: {map_frame} -> {base_frame}'
        )

    # ------------------- 基础功能：获取位姿 -------------------

    def get_pose_in_map(self, timeout_sec: float = 2.0):
        """
        获取机器人在 map 坐标系下的位姿 (x, y, yaw).
        :param timeout_sec: TF 查询超时时间
        :return: (x, y, yaw) 或 None（失败）
        """
        # 先等一会儿，直到 TF 可用
        try:
            can = self.tf_buffer.can_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),              # 最新时间
                timeout=Duration(seconds=timeout_sec)
            )
        except Exception as ex:
            self.get_logger().warn(
                f'can_transform 异常: {self.map_frame} -> {self.base_frame}: {ex}')
            return None

        if not can:
            self.get_logger().warn(
                f'在 {timeout_sec}s 内无法获得 TF: {self.map_frame} -> {self.base_frame}')
            return None

        # 真正取 transform
        try:
            trans = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time()
            )
        except Exception as ex:
            self.get_logger().warn(
                f'lookup_transform 失败: {self.map_frame} -> {self.base_frame}: {ex}')
            return None

        t = trans.transform.translation
        r = trans.transform.rotation

        x = t.x
        y = t.y
        yaw = quaternion_to_yaw(r)

        return x, y, yaw

    # ------------------- 行为：停止机器人 -------------------
    def stop_robot(self):
        """立即发布 0 速度，停止机器人。"""
        twist = Twist()
        self.cmd_pub.publish(twist)

    # ------------------- 行为：直线运动 -------------------
    def move_straight(self,
                      distance: float,
                      speed: float = 0.2):
        """
        开环直线运动：只根据时间控制，不读取位姿。
        distance > 0：前进；distance < 0：后退。
        :param distance: 目标距离（m）
        :param speed:   线速度标称值（m/s，取正数）
        """
        if abs(distance) < 1e-6:
            self.get_logger().info('目标距离为 0，直接返回。')
            return

        if speed <= 0.0:
            self.get_logger().error('speed 必须为正数。')
            return

        # 方向
        sign = 1.0 if distance > 0.0 else -1.0
        v = sign * speed

        # 需要持续的时间
        duration = abs(distance) / speed

        self.get_logger().info(
            f'[开环] 直线运动：distance={distance:.3f} m, '
            f'speed={v:.3f} m/s, duration={duration:.3f} s'
        )

        twist = Twist()
        twist.linear.x = v
        twist.angular.z = 0.0

        start_time = time.time()
        try:
            while rclpy.ok() and (time.time() - start_time) < duration:
                self.cmd_pub.publish(twist)
                # 如果你不需要在运动过程中处理其他回调，这里 spin_once 可以省略
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(0.02)  # 50 Hz 发布
        finally:
            self.stop_robot()
            self.get_logger().info('[开环] 直线运动完成，已停止。')

    # ------------------- 行为：旋转运动 -------------------
    def rotate(self,
               angle_rad: float,
               angular_speed: float = 0.5):
        """
        开环旋转：只根据时间控制，不读取位姿。
        angle_rad > 0：逆时针；angle_rad < 0：顺时针。
        :param angle_rad:     目标旋转角（弧度）
        :param angular_speed: 角速度标称值（rad/s，取正数）
        """
        if abs(angle_rad) < 1e-6:
            self.get_logger().info('目标旋转角度为 0，直接返回。')
            return

        if angular_speed <= 0.0:
            self.get_logger().error('angular_speed 必须为正数。')
            return

        sign = 1.0 if angle_rad > 0.0 else -1.0
        w = sign * angular_speed

        duration = abs(angle_rad) / angular_speed

        self.get_logger().info(
            f'[开环] 旋转：angle={angle_rad:.3f} rad '
            f'({math.degrees(angle_rad):.1f} deg), '
            f'angular_speed={w:.3f} rad/s, duration={duration:.3f} s'
        )
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = w
        start_time = time.time()
        try:
            while rclpy.ok() and (time.time() - start_time) < duration:
                self.cmd_pub.publish(twist)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(0.02)  # 50 Hz 发布
        finally:
            self.stop_robot()
            self.get_logger().info('[开环] 旋转完成，已停止。')

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    try:
        # 如果你只是想控制，不关心位姿，可以直接调用：
        node.move_straight(1.0, speed=0.2)
        node.rotate(math.pi / 2.0, angular_speed=0.5)

        # 如果你偶尔需要看一下当前位姿，再主动调用一次 get_pose_in_map：
        pose = None
        # 注意：为了让 TF listener 工作，这里要至少 spin 一下
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.1)
            pose = node.get_pose_in_map(timeout_sec=0.5)
            if pose is not None:
                break

        if pose is not None:
            x, y, yaw = pose
            node.get_logger().info(
                f'当前位姿：x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} rad '
                f'({math.degrees(yaw):.1f} deg)'
            )
        else:
            node.get_logger().warn('仍未获取到位姿。')

    finally:
        node.destroy_node()
        rclpy.shutdown()