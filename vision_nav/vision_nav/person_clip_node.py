#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS2 节点：RealSense + YOLO(person) + CLIP(衣服颜色分类，阈值筛选，多目标)
发布：
  - /person/clip_debug_image (sensor_msgs/Image)  带框与概率的可视化
  - /person/clip_detections  (vision_msgs/Detection2DArray)  结构化结果
参数（可运行时修改）：
  - color_width:int, color_height:int, fps:int
  - yolo_weights:str, conf_thr:float, iou_thr:float
  - threshold:float, temperature:float 或 -1 表示不用温度缩放
  - prompts:str[]      默认 ["a person wearing black clothes", "a person wearing white clothes", "a person wearing green clothes"]
  - device:str         "cuda" / "cpu" / "auto"
  - window:bool        是否弹本地窗口（服务器上建议关）
依赖：
  - ROS2: rclpy, sensor_msgs, vision_msgs, cv_bridge
  - PyPI: pyrealsense2, ultralytics, torch, openai-clip, opencv-python, pillow, numpy
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import numpy as np
import cv2
from cv_bridge import CvBridge
import pyrealsense2 as rs
from ultralytics import YOLO
import torch
import clip
from PIL import Image
from sensor_msgs.msg import Image as ImageMsg
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D

# ---------------- 参数配置数据类 ----------------
@dataclass
class Cfg:
    color_width: int = 640
    color_height: int = 480
    fps: int = 30
    yolo_weights: str = "weights/yolo11n.pt"
    conf_thr: float = 0.45
    iou_thr: float = 0.50
    threshold: float = 0.80
    temperature: Optional[float] = 0.01  # None 表示不用温度缩放；若参数设为负数则视为 None
    prompts: Optional[List[str]] = None
    device: str = "auto"  # "cuda" | "cpu" | "auto"
    window: bool = False  # 是否用 OpenCV 弹窗调试

    def finalize(self):
        if self.prompts is None or len(self.prompts) == 0:
            self.prompts = [
                "a person wearing black clothes",
                "a person wearing white clothes",
                "a person wearing green clothes",
            ]

class PersonColorClipNode(Node):
    def __init__(self):
        super().__init__("person_color_clip_node")
        self.bridge = CvBridge()
        # --------- ROS 参数读取 ----------
        self.declare_parameter("color_width", 640)
        self.declare_parameter("color_height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("yolo_weights", "weights/yolo11n.pt")
        self.declare_parameter("conf_thr", 0.45)
        self.declare_parameter("iou_thr", 0.50)
        self.declare_parameter("threshold", 0.80)
        self.declare_parameter("temperature", 0.01)  # 设为负数表示禁用温度缩放
        self.declare_parameter("prompts", [
            "a person wearing black clothes",
            "a person wearing white clothes",
            "a person wearing green clothes",
        ])

        self.declare_parameter("device", "auto")  # auto/cuda/cpu
        self.declare_parameter("window", True)
        self.cfg = Cfg(
            color_width=int(self.get_parameter("color_width").value),
            color_height=int(self.get_parameter("color_height").value),
            fps=int(self.get_parameter("fps").value),
            yolo_weights=str(self.get_parameter("yolo_weights").value),
            conf_thr=float(self.get_parameter("conf_thr").value),
            iou_thr=float(self.get_parameter("iou_thr").value),
            threshold=float(self.get_parameter("threshold").value),
            temperature=float(self.get_parameter("temperature").value),
            prompts=list(self.get_parameter("prompts").value),
            device=str(self.get_parameter("device").value),
            window=bool(self.get_parameter("window").value),
        )
        if self.cfg.temperature is not None and self.cfg.temperature < 0:
            self.cfg.temperature = None
        self.cfg.finalize()

        # 支持运行时动态改参
        from rcl_interfaces.msg import SetParametersResult
        def _on_param_update(params):
            for p in params:
                n, v = p.name, p.value
                try:
                    if n in ("color_width", "color_height", "fps"):
                        setattr(self.cfg, n if n != "color_width" else "color_width", int(v))
                        setattr(self.cfg, n if n != "color_height" else "color_height", int(v))
                    elif n in ("conf_thr", "iou_thr", "threshold"):
                        setattr(self.cfg, n, float(v))
                    elif n == "temperature":
                        self.cfg.temperature = None if float(v) < 0 else float(v)
                    elif n == "yolo_weights":
                        self.cfg.yolo_weights = str(v)
                    elif n == "prompts":
                        self.cfg.prompts = list(v)
                    elif n == "device":
                        self.cfg.device = str(v)
                    elif n == "window":
                        self.cfg.window = bool(v)
                except Exception as e:
                    self.get_logger().warn(f"参数更新失败 {n}: {e}")
            return SetParametersResult(successful=True)
        self.add_on_set_parameters_callback(_on_param_update)

        # --------- 初始化 RealSense ----------
        self.pipeline = rs.pipeline()
        self.rs_cfg = rs.config()
        self.rs_cfg.enable_stream(rs.stream.color, self.cfg.color_width, self.cfg.color_height, rs.format.bgr8, self.cfg.fps)
        self.profile = self.pipeline.start(self.rs_cfg)
        # 预热几帧，自动曝光稳定
        for _ in range(15):
            self.pipeline.wait_for_frames(1000)
        self.get_logger().info("RealSense pipeline started.")
        # --------- 初始化 YOLO ----------
        self.model = YOLO(self.cfg.yolo_weights)
        self.names: Dict[int, str] = self.model.model.names
        self.get_logger().info(f"YOLO loaded: {self.cfg.yolo_weights}")
        # --------- 初始化 CLIP ----------
        if self.cfg.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.cfg.device
        self.device = device
        self.get_logger().info(f"Using device: {self.device}")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        with torch.no_grad():
            text_tokens = clip.tokenize(self.cfg.prompts).to(self.device)   # [T, L]
            text_emb = self.clip_model.encode_text(text_tokens)             # [T, D]
            self.text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)  # 归一化
        self.get_logger().info(f"CLIP text embeddings ready: {len(self.cfg.prompts)} prompts.")
        # --------- 发布者 ----------
        self.pub_img = self.create_publisher(ImageMsg, "/person/clip_debug_image", 10)
        self.pub_det = self.create_publisher(Detection2DArray, "/person/clip_detections", 10)
        # OpenCV 窗口可选
        if self.cfg.window:
            cv2.namedWindow("D435 + YOLO + CLIP (person color, threshold)", cv2.WINDOW_NORMAL)

        # --------- 定时器主循环 ----------
        self.prev = time.time()
        self.timer = self.create_timer(0.0 if self.cfg.fps <= 0 else 1.0 / self.cfg.fps, self.on_timer)
    # ---------------- 计时器回调：读取帧 -> YOLO -> CLIP -> 发布 ----------------
    def on_timer(self):
        try:
            frames = self.pipeline.wait_for_frames(1000)
            color = frames.get_color_frame()
            if not color:
                return
            frame = np.asanyarray(color.get_data())

            # YOLO 检测（仅 person）
            yolo_res = self.model(frame, conf=self.cfg.conf_thr, iou=self.cfg.iou_thr, verbose=False)[0]

            # 收集 person ROI
            rois_pil, rois_xyxy = self._collect_person_rois(frame, yolo_res)

            selected = []
            if rois_pil:
                # CLIP 颜色概率（在 prompts 子集上计算 softmax）
                probs_labels = self._clip_probs(rois_pil)
                # 阈值筛选
                for (x1, y1, x2, y2), (prob, label_idx) in zip(rois_xyxy, probs_labels):
                    if prob >= self.cfg.threshold:
                        selected.append((prob, label_idx, (x1, y1, x2, y2)))

            # 可视化 + 发布
            now = time.time()
            fps = 1.0 / max(1e-6, (now - self.prev))
            self.prev = now

            vis = frame.copy()
            self._draw(vis, selected, fps)
            # 发布图像
            msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_color_optical_frame"
            self.pub_img.publish(msg)

            # 发布结构化检测
            detarr = Detection2DArray()
            detarr.header = msg.header
            for prob, label_idx, (x1, y1, x2, y2) in selected:
                det = Detection2D()
                det.header = msg.header
                det.results.append(self._hypothesis(label_idx, prob))
                det.bbox = self._bbox2d(x1, y1, x2, y2)
                detarr.detections.append(det)
            self.pub_det.publish(detarr)

            # OpenCV 窗口
            if self.cfg.window:
                cv2.imshow("D435 + YOLO + CLIP (person color, threshold)", vis)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().warn(f"处理帧异常: {e}")

    # ---------------- 工具：ROI 收集、CLIP 概率、可视化、消息构建 ----------------
    def _collect_person_rois(self, frame_bgr, yolo_res):
        rois_pil: List[Image.Image] = []
        rois_xyxy: List[Tuple[int, int, int, int]] = []
        boxes = yolo_res.boxes
        if boxes is None or len(boxes) == 0:
            return rois_pil, rois_xyxy

        H, W = frame_bgr.shape[:2]
        for b in boxes:
            cls = int(b.cls[0].item()) if b.cls is not None else -1
            cls_name = self.names.get(cls, "")
            if cls_name != "person":
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            x1 = max(0, min(x1, W - 1)); x2 = max(0, min(x2, W - 1))
            y1 = max(0, min(y1, H - 1)); y2 = max(0, min(y2, H - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            roi = frame_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            rois_pil.append(pil)
            rois_xyxy.append((x1, y1, x2, y2))
        return rois_pil, rois_xyxy

    def _clip_probs(self, rois_pil: List[Image.Image]):
        if not rois_pil:
            return []
        with torch.no_grad():
            imgs = torch.stack([self.clip_preprocess(p) for p in rois_pil]).to(self.device)  # [N,3,224,224]
            img_emb = self.clip_model.encode_image(imgs)                                      # [N,D]
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sims = img_emb @ self.text_emb.T                                                 # [N,T]
        results: List[Tuple[float, int]] = []
        for i in range(sims.shape[0]):
            logits = sims[i]
            if self.cfg.temperature is not None:
                logits = logits / self.cfg.temperature
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            j = int(probs.argmax())
            results.append((float(probs[j]), j))
        return results

    def _draw(self, frame: np.ndarray, selected, fps: float):
        for prob, label_idx, (x1, y1, x2, y2) in selected:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = self.cfg.prompts[label_idx]
            cv2.putText(frame, f"{label} ({prob*100:.1f}%)", (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 230, 30), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 0), 2)

    @staticmethod
    def _hypothesis(label_idx: int, prob: float) -> ObjectHypothesisWithPose:
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = str(label_idx)  # 文本标签在 prompts 中的索引
        hyp.hypothesis.score = prob               # 概率
        return hyp

    @staticmethod
    def _bbox2d(x1, y1, x2, y2) -> BoundingBox2D:
        bbox = BoundingBox2D()
        bbox.center.x = (x1 + x2) / 2.0
        bbox.center.y = (y1 + y2) / 2.0
        bbox.size_x = max(0.0, float(x2 - x1))
        bbox.size_y = max(0.0, float(y2 - y1))
        return bbox

    def destroy_node(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        if self.cfg.window:
            cv2.destroyAllWindows()
        super().destroy_node()

def main():
    rclpy.init()
    node = PersonColorClipNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
