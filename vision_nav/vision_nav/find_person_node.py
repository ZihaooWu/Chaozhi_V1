#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_nav.person_detection_realsense import PersonDetector
from geometry_msgs.msg import Twist

class FindPersonNode(Node):
    def __init__(self):
        super().__init__('person_detection_node')
        self.get_logger().info("创建了一个行人检测的节点")
         # 发布cmd_vel的发布者
        self.pub_cmdvel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.persondetector = PersonDetector(show_window=True)
        self.persondetector.start_detection()
        # 创建一个定时器，周期性检查检测结果
        self.timer = self.create_timer(0.5, self.timer_cb)

    def timer_cb(self):
        """周期性调用该函数用于判断是否检测到行人，并控制机器人"""        
        if self.persondetector.person_flag:
            vx, vy, wz = 0.0, 0.0, 0.0
        else:
            vx, vy, wz = 0.0, 0.0, 0.5
        self.publish_cmd_vel(vx, vy, wz)

    
    def publish_cmd_vel(self, vx, vy, wz):
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.angular.z = float(wz)
        self.pub_cmdvel.publish(msg)
    
    def destroy_node(self):
        """退出时清理资源"""
        try:
            self.persondetector.stop_detection()
        except Exception:
            pass
        super().destroy_node()
            
def main(args = None):
    rclpy.init(args=args)
    node = FindPersonNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()     
