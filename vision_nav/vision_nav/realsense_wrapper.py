# 启动realsense的功能模块
import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import os
from datetime import datetime

class RealSenseRGBD:
    def __init__(self, width=640, height=480, fps=30,
                 color_format=rs.format.bgr8, depth_format=rs.format.z16,
                 align_to_color=True):
        self.width, self.height, self.fps = width, height, fps
        self.color_format, self.depth_format = color_format, depth_format
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, depth_format, fps)
        self.config.enable_stream(rs.stream.color, width, height, color_format, fps)
        self.profile = None
        self.align = rs.align(rs.stream.color) if align_to_color else None
        self._started = False
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("RobotRealsense")

    def start(self):
        if not self._started:
            self.profile = self.pipeline.start(self.config)
            self._started = True
            self.logger.info("相机已经启动......")

    def stop(self):
        if self._started:
            try:
                self.pipeline.stop()
                self.logger.info("相机已停止......")
            finally:
                self._started = False

    def wait_for_frames(self, timeout_ms):
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if self.align is not None:
            frames = self.align.process(frames)
        return frames

    def get_rgb_depth(self, timeout_ms):
        """返回对齐后的 (color_np, depth_np, color_frame, depth_frame)"""
        frames = self.wait_for_frames(timeout_ms)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, None, None
        color_np = np.asanyarray(color_frame.get_data())
        depth_np = np.asanyarray(depth_frame.get_data())
        return color_np, depth_np, color_frame, depth_frame

    def save_one_png(self, out_path="/tmp/d455_color.png", timeout_ms=5000, warmup_frames=10):
        """
        保存一张彩色 PNG。
        - warmup_frames: 丢弃前几帧，避免自动曝光/白平衡未稳定导致第一帧偏暗/偏色
        """
        self.start()

        # 确保输出目录存在
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 预热：丢弃若干帧
        for _ in range(max(0, warmup_frames)):
            _ = self.wait_for_frames(timeout_ms)

        color, _, _, _ = self.get_rgb_depth(timeout_ms)
        if color is None:
            raise RuntimeError("未获取到彩色/深度帧，请检查相机连接与流配置。")

        ok = cv2.imwrite(out_path, color)
        if not ok:
            raise RuntimeError(f"保存失败：cv2.imwrite 返回 False，路径={out_path}")

        self.logger.info(f"已保存彩色 PNG：{out_path}")
        return out_path

    def show_stream(self):
        """实时显示彩色图像"""
        self.start()
        self.logger.info("按 Q 退出显示。")
        try:
            while True:
                color, _, _, _ = self.get_rgb_depth(5000)
                if color is None:
                    continue
                cv2.imshow("RealSense RGB", color)
                key = cv2.waitKey(1)
                if key & 0xFF in (ord('q'), ord('Q'), 27):
                    break
        finally:
            self.stop()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    cam = RealSenseRGBD(width=1280, height=720, fps=30)
    try:
        # 用时间戳命名，避免覆盖
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        cam.save_one_png(out_path=f"/home/rpp/wzh/nav_ws/src/vision_nav/resource/1.png")
    finally:
        cam.stop()
