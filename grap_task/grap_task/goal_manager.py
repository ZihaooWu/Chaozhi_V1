#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, Any, Optional

from geometry_msgs.msg import PoseStamped


class GoalManager:
    """
    负责：
    - 从 JSON 文件加载多个导航点
    - 根据名称构造 PoseStamped（需要外部提供时间戳）
    """

    def __init__(self, json_path: str):
        self._goals: Dict[str, Any] = {}
        self._json_path = json_path
        self.load_goals_from_json(json_path)

    def load_goals_from_json(self, json_path: str) -> None:
        """
        从 JSON 文件加载导航点到 self._goals
        JSON 结构参考：
        {
          "name1": {
            "frame_id": "map",
            "position": {"x":..., "y":..., "z":...},
            "orientation": {"x":..., "y":..., "z":..., "w":...}
          },
          ...
        }
        """
        path = Path(json_path)
        if not path.is_file():
            raise FileNotFoundError(f"目标点配置文件不存在: {json_path}")

        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("goals.json 格式错误：顶层必须是字典")

        self._goals = data

    def has_goal(self, name: str) -> bool:
        return name in self._goals

    def build_pose_from_name(
        self,
        name: str,
        stamp,
        default_frame: str = "map"
    ) -> Optional[PoseStamped]:
        """
        根据导航点名称构造 PoseStamped。
        :param name: 目标点名称（JSON 的 key）
        :param stamp: 外部传入的时间戳，一般用 node.get_clock().now().to_msg()
        :param default_frame: 若 JSON 未写 frame_id，则使用该默认值
        """
        if not self._goals:
            return None

        goal_cfg = self._goals.get(name)
        if goal_cfg is None:
            return None

        try:
            frame_id = goal_cfg.get("frame_id", default_frame)
            pos = goal_cfg["position"]
            ori = goal_cfg["orientation"]
        except KeyError:
            return None

        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = stamp

        pose.pose.position.x = float(pos.get("x", 0.0))
        pose.pose.position.y = float(pos.get("y", 0.0))
        pose.pose.position.z = float(pos.get("z", 0.0))

        pose.pose.orientation.x = float(ori.get("x", 0.0))
        pose.pose.orientation.y = float(ori.get("y", 0.0))
        pose.pose.orientation.z = float(ori.get("z", 0.0))
        pose.pose.orientation.w = float(ori.get("w", 1.0))

        return pose
