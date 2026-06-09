#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs      # 仅用于注册 PoseStamped type support
from tf2_ros import TransformException

from vision_nav.nav2_navigator import Nav2Navigator


class PersonGoalNavNode(Nav2Navigator):
    """
    订阅 /person_goal，将其从源坐标系转换到 target_frame（默认 map），
    然后调用 Nav2Navigator.send_goal() 发送给 Nav2。

    行为逻辑：
    - 若当前未在导航：直接发送新目标；
    - 若当前正在导航：
        * 新目标与上次目标距离 < 阈值：忽略；
        * 新目标与上次目标距离 ≥ 阈值：取消当前导航，并发送新目标。
    """

    def __init__(self):
        super().__init__(node_name="person_goal_nav")

        # 目标坐标系（一般是 "map"，你也可以先改成 "odom" 测）
        self.declare_parameter('target_frame', 'map')
        self.target_frame: str = self.get_parameter('target_frame').value

        # 触发重新规划的距离阈值（单位: m）
        self.declare_parameter('replan_distance_threshold', 0.5)
        self.replan_distance_threshold: float = self.get_parameter(
            'replan_distance_threshold'
        ).value

        # 记录上一次发送给 Nav2 的目标（target_frame 下的 PoseStamped）
        self.last_goal_in_target: PoseStamped | None = None

        # TF Buffer + Listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 订阅 /person_goal
        self.person_goal_sub = self.create_subscription(
            PoseStamped,
            '/person_goal',
            self.person_goal_callback,
            1
        )

        self.get_logger().info(
            f"PersonGoalNavNode 启动，订阅 /person_goal，转换到 {self.target_frame} 后发送给 Nav2。"
        )

    def person_goal_callback(self, msg: PoseStamped):
        """
        收到 /person_goal 后：
        1）将 PoseStamped 从 src_frame 转到 target_frame；
        2）按照阈值和导航状态决定是否更新 Nav2 目标。
        """
        src_frame = msg.header.frame_id

        # 使用 latest TF，避免 future extrapolation
        msg.header.stamp = Time().to_msg()

        try:
            # 直接用 transform：PoseStamped -> PoseStamped（target_frame）
            goal_in_target: PoseStamped = self.tf_buffer.transform(
                msg,
                self.target_frame,
                timeout=Duration(seconds=0.5)
            )
        except TransformException as e:
            self.get_logger().warn(
                f"TF 转换失败： {src_frame} -> {self.target_frame}, 错误：{e}"
            )
            return

        # 更新时间戳，表示“当前目标”
        goal_in_target.header.stamp = self.get_clock().now().to_msg()

        new_x = goal_in_target.pose.position.x
        new_y = goal_in_target.pose.position.y

        # 1. 首次目标：直接发送
        if self.last_goal_in_target is None:
            self.get_logger().info(
                f"[首次] /person_goal ({src_frame}) → {self.target_frame}: "
                f"x={new_x:.2f}, y={new_y:.2f}，发送导航目标"
            )
            self.last_goal_in_target = goal_in_target
            self.send_goal(goal_in_target)
            return

        # 2. 与上次目标的距离
        old_x = self.last_goal_in_target.pose.position.x
        old_y = self.last_goal_in_target.pose.position.y
        dx = new_x - old_x
        dy = new_y - old_y
        dist = (dx * dx + dy * dy) ** 0.5

        self.get_logger().info(
            f"/person_goal 更新：与上次目标距离 = {dist:.2f} m"
        )

        # 3. 正在导航中
        if self._is_navigating:
            if dist < self.replan_distance_threshold:
                self.get_logger().debug(
                    f"行人移动({dist:.2f} m) < 阈值({self.replan_distance_threshold:.2f} m)，忽略本次更新"
                )
                return
            else:
                self.get_logger().warn(
                    f"行人移动较大({dist:.2f} m ≥ {self.replan_distance_threshold:.2f} m)，"
                    f"取消当前导航并发送新目标"
                )
                self.cancel_current_goal()
                self.last_goal_in_target = goal_in_target
                self.send_goal(goal_in_target)
                return

        # 4. 当前未在导航：直接发新目标
        self.get_logger().info(
            f"当前未在导航，发送新的行人目标: x={new_x:.2f}, y={new_y:.2f}"
        )
        self.last_goal_in_target = goal_in_target
        self.send_goal(goal_in_target)


def main(args=None):
    rclpy.init(args=args)
    node = PersonGoalNavNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C, 退出 PersonGoalNavNode")
    finally:
        node.destroy_node()
        rclpy.shutdown()
