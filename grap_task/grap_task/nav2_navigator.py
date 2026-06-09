#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from grap_task.server import ChassisMessageParser, ChassisTCPServer
from grap_task.client import ChassisTCPClient

from geometry_msgs.msg import Twist

class Nav2Navigator(Node):
    """
    基于 Nav2 NavigateToPose 的导航封装类：
    - 发送目标点
    - 接收反馈
    - 获取导航结果
    """

    def __init__(self, node_name: str = "nav2_navigator"):
        super().__init__(node_name)

        # 创建 ActionClient，话题名为 Nav2 默认的 `navigate_to_pose`
        self._action_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        # 状态变量
        self._is_navigating = False
        self._current_goal = None
        self._current_goal_handle = None
        self._last_feedback = None
        self._last_result = None
        self.status = None

    # ========== 对外接口 ==========
    def is_navigating(self) -> bool:
        """是否当前有正在执行的导航任务"""
        return self._is_navigating

    def send_goal(self, pose: PoseStamped):
        """
        异步发送一个导航目标，立即返回。
        实际的反馈和结果通过回调处理。
        """
        if not self._action_client.wait_for_server():
            self.get_logger().error("Nav2 NavigateToPose action server 不可用！")
            return None

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        self._current_goal = pose
        self._is_navigating = True
        self._last_feedback = None
        self._last_result = None
        self.get_logger().info(
            f"发送导航目标: frame={pose.header.frame_id}, "
            f"x={pose.pose.position.x:.2f}, y={pose.pose.position.y:.2f}"
        )

        send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self._feedback_callback
        )
        send_goal_future.add_done_callback(self._goal_response_callback)    
        return send_goal_future

    def send_goal_and_wait(self, pose: PoseStamped):
        """
        方便调试/脚本使用：发送目标并同步等待结果。

        注意：
        - 需要在 rclpy.spin 或者 executor 运行的上下文中使用
        - timeout 为 None 表示一直等
        """
        future = self.send_goal(pose)
        if future is None:
            return None

        # 等待 goal 是否被接受
        rclpy.spin_until_future_complete(self, future)
        if not future.done():
            self.get_logger().warn("等待导航目标响应超时")
            return None

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("导航目标被 Nav2 拒绝")
            self._is_navigating = False
            return None

        self.get_logger().info("导航目标已被接受，开始执行")

        # 等待结果
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        if not result_future.done():
            self.get_logger().warn("等待导航结果超时")
            return None
        result = result_future.result().result
        status = result_future.result().status
        self.status = status
        self._last_result = result
        self._is_navigating = False
        self.get_logger().info(f"导航任务完成，状态: {status}")
        return status

    def cancel_current_goal(self):
        """取消当前导航"""
        if self._current_goal_handle is not None:
            cancel_future = self._current_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self._cancel_done_callback)

    # ========== 内部回调 ==========
    def _goal_response_callback(self, future):
        """Nav2 对 goal 请求的响应（是否接受）"""
        goal_handle = future.result()
        self._current_goal_handle = goal_handle

        if not goal_handle.accepted:
            self.get_logger().warn("导航目标被 Nav2 拒绝")
            self._is_navigating = False
            return

        self.get_logger().info("导航目标已被接受")

        # 等待结果（异步方式）
        self.get_logger().info('✅ 目标已接受，等待导航完成...')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        """导航任务结束回调"""
        result = future.result().result
        status = future.result().status

        self._last_result = result
        self.status = status
        self._is_navigating = False

        self.get_logger().info(f"导航任务结束，status={status}")
        # 这里可以根据 status 做额外逻辑，比如重试等

    def _feedback_callback(self, feedback_msg):
        """导航过程中的反馈"""
        feedback = feedback_msg.feedback
        self._last_feedback = feedback

        # 根据 Nav2 的 feedback 定义，一般会包含当前位姿、剩余距离等字段
        # 这里简单打印一下剩余距离（不同版本字段名可能略有不同）
        try:
            dist = feedback.distance_remaining
            self.get_logger().info(f"导航中，剩余距离: {dist:.2f} m")
        except AttributeError:
            self.get_logger().info("收到导航反馈（字段名与当前 Nav2 版本不完全匹配）")

    def _cancel_done_callback(self, future):
        self.get_logger().info("取消导航请求已发送")
        self._is_navigating = False

def main():
    rclpy.init()
    nav = Nav2Navigator()

    # ========== 在主函数中直接写死“回到原点” ==========
    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.stamp = nav.get_clock().now().to_msg()

    goal.pose.position.x = 0.0
    goal.pose.position.y = 0.0
    goal.pose.position.z = 0.0

    # yaw = 0 -> quaternion = (0, 0, sin(0/2), cos(0/2)) = (0, 0, 0, 1)
    goal.pose.orientation.x = 0.0
    goal.pose.orientation.y = 0.0
    goal.pose.orientation.z = 0.0
    goal.pose.orientation.w = 1.0
    # ===============================================

    nav.get_logger().info("开始测试：发送“回到原点”目标 (map: x=0,y=0,yaw=0)")
    status = nav.send_goal_and_wait(goal)
    nav.get_logger().info(f"回到原点测试结束，Nav2 status={status}")

    nav.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
