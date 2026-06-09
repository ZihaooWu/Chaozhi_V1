#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 实现无差别跟踪，无引入特征识别
"""
PersonTracker：基于 YOLOv11n + BoT-SORT 的行人检测与跟踪类
 - 依赖 ultralytics, opencv-python, numpy
 - 相机部分通过 RealSenseRGBD 封装模块调用
"""
import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
from vision_nav.realsense_wrapper import RealSenseRGBD

class PersonTracker:
    def __init__(self,
                 weights="weights/yolo11n.pt",
                 tracker_cfg="botsort.yaml",
                 conf=0.6,
                 iou=0.5,
                 show_window=True):
        # --- 参数 ---
        self.weights = weights
        self.tracker_cfg = tracker_cfg
        self.conf = conf
        self.iou = iou
        self.show_window = show_window
        
        # --- 状态 ---
        self.person_flag = False
        self.person_distance = None
        self.person_offset = 0.0
        self.target_id = None
        self.lost_frames = 0
        self.LOST_RESET = 60
        self._stop_event = threading.Event()
        
        # --- 初始化 YOLO ---
        self.model = YOLO(self.weights)
        self.model.fuse()
        print(f"✅ 加载YOLO成功：{self.weights}")
        
        # --- 初始化 RealSense ---
        self.realsense = RealSenseRGBD(width=640, height=480, fps=30)
        self.realsense.start()
        
        # 获取相机内参（仅一次）
        color_img, depth_img, color_frame, depth_frame = self.realsense.get_rgb_depth(timeout_ms=5000)
        assert color_frame is not None, "未拿到彩色首帧"
        self.depth_scale = self.realsense.profile.get_device().first_depth_sensor().get_depth_scale()
        self.color_intr = color_frame.profile.as_video_stream_profile().get_intrinsics()
        
        # --- 线程 ---
        self._thread = None

    def _get_center_depth(self, depth_img, bbox, ksize=5):
        """取检测框中心区域的中值深度"""
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        h, w = depth_img.shape
        k = ksize // 2
        xL, xR = max(0, cx - k), min(w, cx + k + 1)
        yT, yB = max(0, cy - k), min(h, cy + k + 1)
        patch = depth_img[yT:yB, xL:xR]
        patch = patch[(patch > 0) & (patch < 10000)]
        if patch.size == 0:
            return cx, cy, None
        depth_m = float(np.median(patch) * self.depth_scale)
        return cx, cy, depth_m

    def get_person_position(self):
        """
        获取当前检测到的行人在相机坐标系下的位置 (X, Y, Z)，单位：米。
        若未检测到有效行人，返回 None。
        X:向右； Y:向下； Z:向前
        """
        if not self.person_flag or self.person_distance is None:
            return None

        # 获取相机内参
        fx = self.color_intr.fx
        fy = self.color_intr.fy
        cx = self.color_intr.ppx
        cy = self.color_intr.ppy

        # 获取最近一次检测框中心点
        x1, y1, x2, y2 = map(int, self.last_bbox)
        u, v = (x1 + x2)//2, (y1 + y2)//2
        Z = self.person_distance
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.array([X, Y, Z])

    def _tracking_loop(self):
        print("🟢 行人追踪线程启动。按 Ctrl+C 退出。")
        try:
            while not self._stop_event.is_set():
                # 从 realsense 模块读取数据
                color_img, depth_img, color_frame, depth_frame = self.realsense.get_rgb_depth(timeout_ms=5000)
                if color_img is None:
                    continue

                # YOLO 检测+跟踪
                result = self.model.track(
                    source=color_img,
                    persist=True,
                    classes=[0],
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False,
                    tracker=self.tracker_cfg
                )
                r = result[0]
                annotated = color_img.copy()
                has_target_this_frame = False
                if r.boxes is not None and len(r.boxes) > 0:
                    if self.target_id is None:
                        ids = [int(b.id.item()) for b in r.boxes if b.id is not None]
                        if ids:
                            self.target_id = min(ids)

                    if self.target_id is not None:
                        for b in r.boxes:
                            if b.id is None:
                                continue
                            tid = int(b.id.item())
                            if tid != self.target_id:
                                continue
                            xyxy = b.xyxy[0].cpu().numpy()
                            cx, cy, depth_m = self._get_center_depth(depth_img, xyxy, ksize=5)
                            conf = float(b.conf.item()) if b.conf is not None else 0.0
                            # 保存当前检测框
                            self.last_bbox = xyxy
                            if depth_m is not None:
                                self.person_distance = depth_m
                                self.person_flag = True
                                self.person_offset = cx - (color_img.shape[1] // 2)
                            else:
                                self.person_flag = False

                            # 绘图
                            if self.show_window:
                                x1, y1, x2, y2 = map(int, xyxy)
                                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                if depth_m is not None:
                                    depth_str = f"{depth_m:.2f}m"
                                else:
                                    depth_str = "N/A"
                                txt = f"ID:{tid} {conf:.2f} {depth_str}"
                                cv2.putText(annotated, txt, (x1, max(0, y1-10)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                            has_target_this_frame = True
                            break

                # 若丢失目标
                if self.target_id is not None and not has_target_this_frame:
                    self.lost_frames += 1
                    if self.lost_frames > self.LOST_RESET:
                        self.target_id = None
                        self.person_flag = False
                        self.person_distance = None
                        self.lost_frames = 0
                else:
                    self.lost_frames = 0

                if self.show_window:
                    info = f"offset={self.person_offset:.0f}px dist={self.person_distance if self.person_distance else 0:.2f}m"
                    cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                    cv2.imshow("PersonTracker", annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.cleanup()
    


    def start(self):
        """启动后台追踪线程"""
        if self._thread and self._thread.is_alive():
            print("⚠️ 追踪线程已在运行")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止后台线程"""
        print("🛑 停止追踪线程...")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.cleanup()

    def cleanup(self):
        """释放资源"""
        try:
            self.realsense.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("✅ 已释放 RealSense 资源。")


# ----------------- 示例 -----------------
if __name__ == "__main__":
    tracker = PersonTracker(show_window=True)
    tracker.start()
    try:
        while True:
            if tracker.person_flag:
                pos = tracker.get_person_position()
                if pos is not None:
                    print(f"行人位置：X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m")
                else:
                    print("检测到行人，但未获取到有效深度。")
            else:
                print("未检测到行人")
            time.sleep(0.5)
    except KeyboardInterrupt:
        tracker.stop()
