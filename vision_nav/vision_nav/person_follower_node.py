#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PersonFollowerNode：基于 ROS2 的人跟随节点
-----------------------------------------
功能：
1. 调用 PersonTracker（YOLOv11n + BoT-SORT + RealSense）模块实现行人检测和深度估计
2. 从 PersonTracker 获取目标在相机坐标系下的三维位置
3. 通过 TF 转换到 map 坐标系
4. 发布 /person/point_map（PointStamped）
5. （可选）自动下发 Nav2 导航目标，实现“自动跟随人”功能
初始版本 v 1.0
"""

import math
import time
from typing import Optional, Tuple
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, PointStamped, Quaternion, Twist
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer, TransformListener
try:
    from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point
except Exception:
    from tf2_geometry_msgs import do_transform_point  # type: ignore

# 导入视觉跟踪模块
from vision_nav.person_tracking_realsense import PersonTracker


class PersonFollowerNode(Node):
    """ROS2 节点：基于视觉行人检测+TF+Nav2 的自动人跟随系统"""

    def __init__(self):
        super().__init__('person_follower_node')

        # ---------------- 参数声明 ----------------
        self.declare_parameters('', [
            ('camera_is_base', True),
            ('camera_frame', 'camera_link'),
            ('map_frame', 'map'),
            ('base_frame', 'base_link'),
            ('publish_topic', '/person/point_map'),
            ('auto_navigate', True),
            ('stop_distance', 0.8),
            ('goal_replan_delta', 1.0),
            ('send_goal_period', 0.5),
            ('face_target', True),
            ('show_debug_window', True),
        ])

        self.cam_is_base = self._param_b('camera_is_base')
        self.camera_frame = self._param_s('camera_frame')
        self.map_frame = self._param_s('map_frame')
        self.base_frame = self._param_s('base_frame')
        pub_topic = self._param_s('publish_topic')
        self.auto_nav = self._param_b('auto_navigate')
        self.stop_distance = self._param_f('stop_distance')
        self.goal_replan_delta = self._param_f('goal_replan_delta')
        self.send_goal_period = self._param_f('send_goal_period')
        self.face_target = self._param_b('face_target')

        # ---------------- ROS 通信组件 ----------------
        self.pub_target_map = self.create_publisher(PointStamped, pub_topic, 10)
        self.pub_cmdvel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_ac = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        if self.auto_nav:
            self.get_logger().info('等待 Nav2 navigate_to_pose 服务器...')
            self.nav_ac.wait_for_server(timeout_sec=10.0)

        # ---------------- 初始化视觉模块 ----------------
        self.tracker = PersonTracker(show_window=False)
        self.tracker.start()

        # ---------------- 状态变量 ----------------
        self.last_goal_xy: Optional[Tuple[float, float]] = None
        self.last_goal_time = 0.0
        self.lost_frames = 0
        self.LOST_RESET = 30
        self.prev_ts = time.time()
        self.fps_est = 0.0

        # 定时器回调（主循环）
        self.timer = self.create_timer(0.05, self._on_timer)
        msg = 'person_follower_node 已启动（camera_is_base=True，无需发布相机静态TF）' if self.cam_is_base \
              else 'person_follower_node 已启动（camera_is_base=False，需要发布相机静态TF）'
        self.get_logger().info(msg)

    # ----------------- 参数读取工具 -----------------
    def _param_s(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value
    def _param_f(self, name: str) -> float:
        pv = self.get_parameter(name).get_parameter_value()
        return float(pv.double_value if pv.type == pv.double_value else pv.integer_value)
    def _param_b(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    # ----------------- 常用几何工具 -----------------
    @staticmethod
    def yaw_to_quaternion(yaw: float) -> Quaternion:
        return Quaternion(x=0.0, y=0.0, z=math.sin(yaw/2.0), w=math.cos(yaw/2.0))
    @staticmethod
    def quaternion_to_yaw(q: Quaternion) -> float:
        return math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z * q.z))
    
    # 获取机器人在map中的位姿
    def get_robot_pose_in_map(self) -> Optional[Tuple[float, float, float]]:
        try:
            # 获取两个坐标系之间的位置关系
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, rclpy.time.Time(), timeout=Duration(seconds=0.2))
            tx = tf.transform.translation.x
            ty = tf.transform.translation.y
            yaw = self.quaternion_to_yaw(tf.transform.rotation)
            return tx, ty, yaw
        except Exception as e:
            self.get_logger().warn(f'获取 {self.map_frame}->{self.base_frame} 失败: {e}')
            return None
    
    # 将相机坐标系下的点转换到map坐标系下
    def cam_point_to_map(self, ps_cam: PointStamped) -> Optional[PointStamped]:
        try:
            # 获取map与相机坐标系的tf关系
            tf = self.tf_buffer.lookup_transform(self.map_frame, ps_cam.header.frame_id, rclpy.time.Time(), timeout=Duration(seconds=0.2))
            return do_transform_point(ps_cam, tf)    # 计算
        except Exception as e:
            self.get_logger().warn(f'变换 {ps_cam.header.frame_id}->{self.map_frame} 失败: {e}')
            return None
    
    def publish_cmd_vel(self, vx: float, vy: float, wz: float):
        """用于发送底盘控制的函数"""
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.angular.z = wz
        self.pub_cmdvel.publish(msg)

    # ----------------- Nav2 相关 -----------------
    def maybe_send_goal(self, tgt_xy_map: Tuple[float, float]):
        if not self.auto_nav:
            return
        now_t = time.time()
        if self.last_goal_xy is not None:
            dx = tgt_xy_map[0] - self.last_goal_xy[0]
            dy = tgt_xy_map[1] - self.last_goal_xy[1]
            moved = math.hypot(dx, dy)
            if moved < self.goal_replan_delta and (now_t - self.last_goal_time) < self.send_goal_period:   # 要移动较大才重新发布新的目标点
                return
        rb = self.get_robot_pose_in_map()  
        if rb is None:
            return
        rx, ry, _ = rb
        tx, ty = tgt_xy_map
        vx, vy = tx - rx, ty - ry
        dist = math.hypot(vx, vy)
        goal_x, goal_y = tx, ty
        if dist > self.stop_distance:
            scale = (dist - self.stop_distance) / max(dist, 1e-6)
            goal_x = rx + vx * scale
            goal_y = ry + vy * scale
        yaw = math.atan2(ty - ry, tx - rx) if self.face_target else 0.0
        self.send_goal_to_nav2(goal_x, goal_y, yaw)     # 发送目标点的功能
        self.last_goal_xy = (tx, ty)                    # 更新上一个目标点
        self.last_goal_time = now_t                     # 更新上一个目标时间

    # 将目标点（以map为参考坐标系）发送到nav2中
    def send_goal_to_nav2(self, x: float, y: float, yaw: float):
        if not self.nav_ac.server_is_ready():
            self.get_logger().warn('Nav2 动作服务器未就绪，丢弃此次 goal')
            return
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation = self.yaw_to_quaternion(yaw)
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self.get_logger().info(f'发送导航目标: x={x:.2f}, y={y:.2f}, yaw={yaw:.2f}')
        self.nav_ac.send_goal_async(goal)

    def _on_goal_response(self, fut):
        goal_handle = fut.result()
        if not goal_handle or not goal_handle.accepted:
            self.get_logger().warn('导航目标被拒绝')
            return
        self.get_logger().info('导航目标已接受')
        goal_handle.get_result_async().add_done_callback(self._on_result)

    def _on_feedback(self, feedback_msg):
        pass

    def _on_result(self, result_future):
        try:
            result = result_future.result().result
            self.get_logger().info(f'导航完成，结果码: {result.result}')
        except Exception as e:
            self.get_logger().warn(f'导航结果异常: {e}')

    # ----------------- 视觉主循环 -----------------
    def _on_timer(self):
        try:
            if not self.tracker.person_flag:
                self.lost_frames += 1
                if self.lost_frames > 10:
                    self.publish_cmd_vel(0.0, 0.0, 0.3)
                return

            pos_cam = self.tracker.get_person_position()     # 获取行人在相机坐标系的位置
            if pos_cam is None:
                self.publish_cmd_vel(0.0, 0.0, 0.3)
                return

            Y, _, X = pos_cam
            ps = PointStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = self.base_frame if self.cam_is_base else self.camera_frame
            ps.point.x, ps.point.y, ps.point.z = X, -Y, 0.0

            ps_map = self.cam_point_to_map(ps)
            if ps_map is not None:
                self.pub_target_map.publish(ps_map)
                tx, ty = ps_map.point.x, ps_map.point.y
                self.maybe_send_goal((tx, ty))
                self.get_logger().info(f"目标点(map): ({tx:.2f}, {ty:.2f})")
        except Exception as e:
            self.get_logger().warn(f'处理帧异常: {e}')
    
    def destroy_node(self):
        try:
            self.tracker.stop()
        except Exception:
            pass
        super().destroy_node()

def main():
    rclpy.init()
    node = PersonFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
