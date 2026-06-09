#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
from ultralytics import YOLO
import numpy as np
import time
import threading
import logging
from vision_nav.realsense_wrapper import RealSenseRGBD
import torch


class PersonDetector:
    """使用 YOLOv11n 检测 person 类别的 RealSense 摄像头检测类"""

    def __init__(
        self,
        conf=0.6,
        iou=0.5,
        show_window=True,
    ):
        # --- 配置参数 ---
        self.conf = conf
        self.iou = iou
        self.show_window = show_window

        # 设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_path = "weights/yolo11n.pt"

        # 加载模型
        self.model = YOLO(self.model_path)
        # 如果 ultralytics 版本不支持 .to，可以注释掉下面这一行，改为在 predict 时指定 device
        self.model.to(self.device)

        # RealSense 相机
        self.realsense = RealSenseRGBD()  # 创建一个 realsense 对象

        # --- 状态变量 ---
        self.last_time = time.time()
        self.fps_est = 0.0

        self.persons_count = 0          # 当前帧检测到的行人数
        self.person_flag = False        # 是否检测到行人
        self.nearest_distance = None    # 最近行人距离（米）
        self.nearest_center = None      # 最近行人图像坐标 (cx, cy)

        # --- 线程控制 ---
        self._stop_event = threading.Event()
        self._thread = None

        # 日志配置
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        self.logger = logging.getLogger("PersonDetector")

    def process_frame(self, frame):
        """运行 YOLO 检测，只检测 person 类（class 0）"""
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            classes=[0],  # 只检测 person 类
            verbose=False,
            device=self.device,
        )
        return results[0]

    def _update_fps(self):
        """简单的 FPS 估计"""
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps_est = 0.9 * self.fps_est + 0.1 * (1.0 / dt)

    def _detection_loop(self):
        """后台检测循环（在线程中运行）"""
        self.logger.info("后台检测线程已启动（按 'q' 关闭窗口 / 停止）")

        # 启动相机
        try:
            self.realsense.start()
        except Exception as e:
            self.logger.error(f"RealSense 启动失败: {e}")
            return

        try:
            while not self._stop_event.is_set():
                # 获取彩色 + 深度
                try:
                    color_np, depth_np, color_frame, depth_frame = self.realsense.get_rgb_depth(
                        timeout_ms=5000
                    )
                except Exception as e:
                    self.logger.warning(f"获取 RealSense 帧失败: {e}")
                    continue

                if color_np is None or depth_frame is None:
                    # 有时候相机会返回空帧，直接跳过
                    continue

                # YOLO 检测
                result = self.process_frame(color_np)
                annotated = result.plot()

                # 更新 FPS
                self._update_fps()

                # 初始化当前帧状态
                self.persons_count = 0
                self.person_flag = False
                self.nearest_distance = None
                self.nearest_center = None

                # 检查是否检测到 person
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    self.persons_count = len(boxes)
                    self.person_flag = True

                    H, W, _ = color_np.shape
                    center_x, center_y = W // 2, H // 2
                    margin_x = W * 0.1  # 宽度方向 ±10%
                    margin_y = H * 0.1  # 高度方向 ±10%

                    nearest_d = None
                    nearest_c = None

                    for box in boxes:
                        # xyxy 是 tensor，取出并转为 int
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                        # 边界保护
                        cx = max(0, min(W - 1, cx))
                        cy = max(0, min(H - 1, cy))

                        # 深度测量可能会失败，需 try 一下
                        try:
                            depth_val = depth_frame.get_distance(cx, cy)
                        except Exception:
                            depth_val = 0.0

                        # RealSense 返回 0 一般意味着无效深度
                        if depth_val is None or depth_val <= 0:
                            self.logger.debug(
                                f"无效深度: ({cx}, {cy})，跳过该检测框"
                            )
                            continue

                        # 更新最近行人
                        if (nearest_d is None) or (depth_val < nearest_d):
                            nearest_d = depth_val
                            nearest_c = (cx, cy)

                        # 判断是否在图像中央
                        in_center = (
                            abs(cx - center_x) < margin_x
                            and abs(cy - center_y) < margin_y
                        )

                        if in_center:
                            print(f"✅ 行人在中央，距离相机 {depth_val:.2f} 米")
                        else:
                            dx = cx - center_x
                            direction = "左边" if dx < 0 else "右边"
                            print(
                                f"⚠ 行人偏在{direction}，水平偏移 {abs(dx)} 像素，距离 {depth_val:.2f} 米"
                            )

                    # 把最近行人信息写入成员变量
                    self.nearest_distance = nearest_d
                    self.nearest_center = nearest_c

                    self.logger.info(
                        f"检测到 {self.persons_count} 个行人，FPS≈{self.fps_est:.1f}"
                    )
                else:
                    self.person_flag = False
                    self.logger.debug("本帧未检测到行人")

                # 显示窗口
                if self.show_window:
                    # 在图像中央画一个参考框，帮助观察“中央区域”
                    H, W, _ = color_np.shape
                    cx, cy = W // 2, H // 2
                    margin_x = int(W * 0.1)
                    margin_y = int(H * 0.1)
                    cv2.rectangle(
                        annotated,
                        (cx - margin_x, cy - margin_y),
                        (cx + margin_x, cy + margin_y),
                        (0, 255, 0),
                        2,
                    )

                    cv2.imshow("YOLOv11n Person Detection", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.logger.info("收到按键 'q'，准备退出检测循环")
                        break

        finally:
            self.cleanup()
            self.logger.info("检测线程已退出")

    def start_detection(self):
        """启动后台检测线程"""
        if self._thread and self._thread.is_alive():
            self.logger.info("检测线程已在运行，无需重复启动")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._detection_loop, daemon=True
        )
        self._thread.start()
        self.logger.info("检测线程启动成功")

    def stop_detection(self):
        """停止后台检测线程"""
        self.logger.info("正在停止检测线程...")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.cleanup()

    def cleanup(self):
        """释放资源"""
        # 相机释放
        try:
            if hasattr(self, "realsense") and self.realsense is not None:
                self.realsense.stop()
        except Exception as e:
            self.logger.debug(f"停止 RealSense 时出错: {e}")

        # 窗口释放
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        self.logger.info("资源已释放，检测已退出")


# ====================== main 函数 ======================

def main(show_window=True):
    """程序入口：创建检测器并循环打印状态"""
    detector = PersonDetector(show_window=show_window)
    detector.start_detection()

    try:
        while True:
            print(
                f"检测标志：{detector.person_flag}，"
                f"人数：{detector.persons_count}，"
                f"最近距离：{detector.nearest_distance}"
            )
            time.sleep(3)
    except KeyboardInterrupt:
        detector.stop_detection()
        print("程序结束。")


if __name__ == "__main__":
    # 可以在这里控制是否弹出窗口，比如 main(show_window=False)
    main(show_window=True)
