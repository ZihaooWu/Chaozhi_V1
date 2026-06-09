#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 在person_tracking_realsense的基础上，捕获单个特征并对齐识别
"""
PersonTrackerReID：基于 YOLOv11n + BoT-SORT + ReID + RealSense 的行人检测与跟踪类

功能：
- RealSense 采集 RGBD
- YOLOv11n + BoT-SORT 检测+多目标跟踪（track_id）
- ResNet50 提取行人外观特征，作为 ReID 向量
- 支持“注册某一行人，只跟踪这个人”
- 行人重现时自动重新锁定
- 对目标行人做深度估计 + 相机坐标系 3D 位置计算

接口：
- start()/stop()：启动/停止后台跟踪线程
- person_flag：是否当前帧成功锁定目标人（且有有效深度）
- get_person_position() -> np.array([X,Y,Z]) 或 None
- last_bbox：目标人的像素坐标 bbox (xyxy)
"""

import cv2
import numpy as np
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from ultralytics import YOLO
from vision_nav.realsense_wrapper import RealSenseRGBD


# ====================== ReID 模块 ===========================
class PersonReID:
    """
    使用 ResNet50(ImageNet预训练) 作为行人特征提取器。
    实际工程中可替换为专门的行人ReID模型（如OSNet等），保持 extract 接口不变即可。
    """
    def __init__(self, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        # 加载 ResNet50 并去掉最后的分类层
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Identity()  # 输出2048维特征
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # 行人ReID常用尺寸(高,宽)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    @torch.no_grad()
    def extract(self, img_bgr: np.ndarray) -> np.ndarray | None:
        """
        输入：BGR 行人裁剪图 (H, W, 3)
        输出：L2 归一化后的特征向量 (D,) numpy.float32
        """
        if img_bgr is None or img_bgr.size == 0:
            return None

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 图像会通过 transform 转换成一个固定大小的 tensor，大小为 (256, 128)
        tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)  # (1,3,H,W)
        feat = self.model(tensor)  # (1,2048)
        feat = F.normalize(feat, p=2, dim=1)  # L2 归一化，便于后续相似度计算
        feat_np = feat.cpu().numpy().reshape(-1)  # (2048,)
        return feat_np.astype(np.float32)        # 返回 2048 维的特征向量，这个特征向量是该行人图像的独特表示。


# ====================== 主跟踪类 ============================
class PersonTrackerReID:
    def __init__(self,
                 weights: str = "weights/yolo11n.pt",
                 tracker_cfg: str = "botsort.yaml",
                 conf: float = 0.6,
                 iou: float = 0.5,
                 sim_threshold: float = 0.5,
                 show_window: bool = True,
                 device: str = "cuda"):
        """
        :param weights: YOLO 权重
        :param tracker_cfg: BoT-SORT 配置
        :param conf: YOLO 置信度阈值
        :param iou: YOLO IOU 阈值
        :param sim_threshold: ReID 余弦相似度阈值
        :param show_window: 是否显示窗口
        :param device: "cuda" or "cpu"
        """

        # --- 参数 ---
        self.weights = weights
        self.tracker_cfg = tracker_cfg
        self.conf = conf
        self.iou = iou
        self.sim_threshold = sim_threshold
        self.show_window = show_window

        # --- 跟踪状态 ---
        self.person_flag = False           # 当前帧是否锁定目标人（且有有效深度）
        self.person_distance = None        # 目标人与相机距离（米）
        self.person_offset = 0.0           # 目标人在图像中的水平偏移（像素）
        self.last_bbox = None              # 目标人bbox (xyxy)
        self.target_id = None              # BoT-SORT track_id，可选
        self.lost_frames = 0
        self.LOST_RESET = 60               # 连续丢失多少帧认为暂时丢失

        # --- ReID 状态 ---
        self.reid = PersonReID(device=device)
        self.has_enrolled = False          # 是否已经注册目标人
        self.enrolled_feature = None       # 注册目标人的特征
        self.enrolled_valid = False        # 当前是否有有效注册特征

        # --- 线程控制 ---
        self._stop_event = threading.Event()
        self._thread = None

        # --- YOLO 初始化 ---
        self.model = YOLO(self.weights)
        if device == "cuda" and torch.cuda.is_available():
            self.model.to("cuda")
        else:
            self.model.to("cpu")
        self.model.fuse()
        print(f"✅ 加载 YOLO 权重成功：{self.weights}")

        # --- RealSense 初始化 ---
        self.realsense = RealSenseRGBD(width=640, height=480, fps=30)
        self.realsense.start()

        color_img, depth_img, color_frame, depth_frame = self.realsense.get_rgb_depth(timeout_ms=5000)
        assert color_frame is not None, "未拿到彩色首帧"
        self.depth_scale = self.realsense.profile.get_device().first_depth_sensor().get_depth_scale()
        self.color_intr = color_frame.profile.as_video_stream_profile().get_intrinsics()

        print("✅ RealSense 初始化完成")

    # -------------------------------------------------------
    # 对外接口
    # -------------------------------------------------------
    def start(self):
        """启动后台跟踪线程"""
        if self._thread and self._thread.is_alive():
            print("⚠️ PersonTrackerReID 线程已在运行")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._tracking_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止后台线程"""
        print("🛑 停止 PersonTrackerReID 线程...")
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
        print("✅ 已释放 RealSense 资源")

    def get_person_position(self) -> np.ndarray | None:
        """
        获取当前检测到的“目标行人”在相机坐标系下的位置 (X, Y, Z)，单位：米。
        若未检测到有效行人，返回 None。
        X:向右；Y:向下；Z:向前
        """
        if not self.person_flag or self.person_distance is None or self.last_bbox is None:
            return None

        fx = self.color_intr.fx
        fy = self.color_intr.fy
        cx = self.color_intr.ppx
        cy = self.color_intr.ppy

        x1, y1, x2, y2 = map(int, self.last_bbox)
        u, v = (x1 + x2) // 2, (y1 + y2) // 2
        Z = self.person_distance
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        return np.array([X, Y, Z], dtype=np.float32)

    # -------------------------------------------------------
    # 内部工具函数
    # -------------------------------------------------------
    def _get_center_depth(self, depth_img, bbox, ksize=5):
        """取检测框中心区域的中值深度（米）"""
        x1, y1, x2, y2 = map(int, bbox)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
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

    def _enroll_from_box(self, frame_bgr, xyxy) -> bool:
        """从给定框注册目标人（更新 enrolled_feature）"""
        x1, y1, x2, y2 = map(int, xyxy)
        crop = frame_bgr[y1:y2, x1:x2]
        feat = self.reid.extract(crop)
        if feat is None:
            print("❌ 注册失败：裁剪图为空或特征提取失败")
            return False
        self.enrolled_feature = feat
        self.has_enrolled = True
        self.enrolled_valid = True
        print("🎯 已成功注册目标行人")
        return True

    def _draw_unenrolled(self, img, boxes):
        """未注册模式下，仅画出检测框并提示"""
        if boxes is not None:
            for idx, b in enumerate(boxes):
                xyxy = b.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(img, str(idx + 1), (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(img, "Press [1-9] to enroll, or [E] to enroll largest",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # -------------------------------------------------------
    # 主跟踪线程
    # -------------------------------------------------------
    def _tracking_loop(self):
        print("🟢 PersonTrackerReID 跟踪线程启动（按 Q 退出窗口）")
        try:
            while not self._stop_event.is_set():
                color_img, depth_img, color_frame, depth_frame = self.realsense.get_rgb_depth(timeout_ms=5000)
                if color_img is None:
                    continue
                # YOLO + BoT-SORT 检测+跟踪
                results = self.model.track(
                    source=color_img,
                    persist=True,
                    classes=[0],
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False,
                    tracker=self.tracker_cfg
                )
                r = results[0]
                boxes = r.boxes
                annotated = color_img.copy()
                
                # ========== 1. 未注册状态：只展示检测框，等用户注册 ==========
                if not self.has_enrolled or not self.enrolled_valid:
                    if boxes is not None and len(boxes) > 0:
                        self._draw_unenrolled(annotated, boxes)
                    else:
                        cv2.putText(annotated, "No person detected. Waiting...",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    if self.show_window:
                        cv2.imshow("PersonTrackerReID", annotated)
                        key = cv2.waitKey(1) & 0xFF
                    else:
                        key = 0
                    # 键盘控制注册逻辑
                    if boxes is not None and len(boxes) > 0:
                        if key in [ord(str(d)) for d in range(1, 10)]:
                            idx = int(chr(key)) - 1
                            if 0 <= idx < len(boxes):
                                xyxy = boxes[idx].xyxy[0].cpu().numpy()
                                self._enroll_from_box(color_img, xyxy)
                        elif key in (ord('e'), ord('E')):
                            # 注册面积最大的人
                            areas = []
                            for b in boxes:
                                xyxy = b.xyxy[0].cpu().numpy()
                                x1, y1, x2, y2 = map(int, xyxy)
                                area = max(0, x2 - x1) * max(0, y2 - y1)
                                areas.append((xyxy, area))
                            if areas:
                                areas.sort(key=lambda x: -x[1])
                                self._enroll_from_box(color_img, areas[0][0])
                    # Q 退出
                    if key in (ord('q'), ord('Q')):
                        self._stop_event.set()
                        break
                    # 未注册模式下不更新 person_flag，只继续循环
                    self.person_flag = False
                    self.person_distance = None
                    self.last_bbox = None
                    continue
                # ========== 2. 已注册状态：进行 ReID 匹配 ==========
                best_sim = -1.0
                best_box = None
                best_tid = None
                if boxes is not None and len(boxes) > 0:
                    for b in boxes:
                        xyxy = b.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        crop = color_img[y1:y2, x1:x2]
                        feat = self.reid.extract(crop)
                        if feat is None or self.enrolled_feature is None:
                            continue
                        sim = float(np.dot(self.enrolled_feature, feat))
                        if sim > best_sim:
                            best_sim = sim
                            best_box = xyxy
                            best_tid = int(b.id.item()) if b.id is not None else None
                if best_box is not None and best_sim >= self.sim_threshold:
                    # 找到目标人
                    self.last_bbox = best_box
                    self.target_id = best_tid
                    cx, cy, depth_m = self._get_center_depth(depth_img, best_box)
                    if depth_m is not None and depth_m > 0:
                        self.person_distance = depth_m
                        self.person_offset = cx - (color_img.shape[1] // 2)
                        self.person_flag = True
                        self.lost_frames = 0
                    else:
                        # 深度无效，可以认为暂时未获取到稳定距离
                        self.person_distance = None
                        self.person_flag = False
                    # 绘制目标框
                    x1, y1, x2, y2 = map(int, best_box)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(annotated, f"TARGET sim={best_sim:.2f}",
                                (x1, max(0, y1 - 15)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # 本帧未匹配到目标人
                    self.person_flag = False
                    self.person_distance = None
                    self.lost_frames += 1
                    if boxes is not None and len(boxes) > 0:
                        # 画出所有检测框（黄色）
                        for b in boxes:
                            xyxy = b.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(annotated, "Target lost, searching...",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    # 若长时间丢失，可以选择清空注册信息（这里仅重置 lost_frames，不清除注册）
                    if self.lost_frames > self.LOST_RESET:
                        self.lost_frames = 0
                        # 若你希望丢失后需要重新注册，可放开下面两行：
                        # self.enrolled_valid = False
                        # self.has_enrolled = False
                # 显示位移和距离信息
                info = f"flag={int(self.person_flag)} dist={self.person_distance if self.person_distance else 0:.2f}m"
                cv2.putText(annotated, info, (10, annotated.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if self.show_window:
                    cv2.imshow("PersonTrackerReID", annotated)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = 0
                if key in (ord('q'), ord('Q')):
                    self._stop_event.set()
                    break
        finally:
            self.cleanup()


# ======================= 简单测试 ===========================
if __name__ == "__main__":
    tracker = PersonTrackerReID(show_window=True)
    tracker.start()
    try:
        while True:
            if tracker.person_flag:
                pos = tracker.get_person_position()
                if pos is not None:
                    print(f"目标行人相机坐标: X={pos[0]:.2f}m, Y={pos[1]:.2f}m, Z={pos[2]:.2f}m")
            time.sleep(0.5)
    except KeyboardInterrupt:
        tracker.stop()
