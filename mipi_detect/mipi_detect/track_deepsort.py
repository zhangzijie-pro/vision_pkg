#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2024，Zhangzijie.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
订阅:
  /yolo_detections (identify/msg/YoloDetections)
  /image_raw       (sensor_msgs/Image，可选；用于外观特征提取)
发布:
  /yolo_tracks         (std_msgs/String, JSON)
  /tracking_markers    (visualization_msgs/MarkerArray)
  /uav/track_cmd       (geometry_msgs/Twist)

参数(ros2 run vision_pkg track_deepsort --ros-args -p key:=val ...):
  target_class: str = "person"     # 只锁定此类的第一个目标
  score_threshold: float = 0.25
  max_age: int = 30
  n_init: int = 3
  max_cosine_distance: float = 0.2
  use_embedder: bool = True        # 有图像帧时启用外观嵌入；无图像自动退化
  embedder_half: bool = True
  image_topic: str = "/image_raw"  # 可按你的相机topic调整
  yolo_msg: str = "yolo_detections"
  debug_log: bool = False
"""

import threading
from typing import Tuple, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from deep_sort_realtime.deepsort_tracker import DeepSort

try:
    from identify.msg import YoloDetections, DpSTracker
except Exception:
    YoloDetections = None
    DpSTracker = None

try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None


def xyxy_from_det(det) -> Tuple[float, float, float, float]:
    if all(hasattr(det, k) for k in ("x_min", "y_min", "x_max", "y_max")):
        return float(det.x_min), float(det.y_min), float(det.x_max), float(det.y_max)
    # fallback
    iw = float(getattr(det, "image_width", 0.0))
    ih = float(getattr(det, "image_height", 0.0))
    w = max(4.0, iw * 0.02)
    h = max(4.0, ih * 0.02)
    cx = float(getattr(det, "cx", iw / 2.0))
    cy = float(getattr(det, "cy", ih / 2.0))
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


class DeepSortUAVNode(Node):
    def __init__(self):
        super().__init__("track_deepsort")

        self.declare_parameter("target_class", "person")
        self.declare_parameter("score_threshold", 0.25)
        self.declare_parameter("max_age", 30)
        self.declare_parameter("n_init", 3)
        self.declare_parameter("max_cosine_distance", 0.2)
        self.declare_parameter("use_embedder", True)
        self.declare_parameter("embedder_half", True)
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("yolo_msg", "/yolo_detections")
        self.declare_parameter("debug_log", False)

        self.target_class: str = str(self.get_parameter("target_class").value)
        self.score_thr: float = float(self.get_parameter("score_threshold").value)
        self.max_age: int = int(self.get_parameter("max_age").value)
        self.n_init: int = int(self.get_parameter("n_init").value)
        self.max_cos_dist: float = float(self.get_parameter("max_cosine_distance").value)
        self.use_embedder: bool = bool(self.get_parameter("use_embedder").value)
        self.embedder_half: bool = bool(self.get_parameter("embedder_half").value)
        self.image_topic: str = str(self.get_parameter("image_topic").value)
        self.yolo_msg: str = str(self.get_parameter("yolo_msg").value)
        self.debug_log: bool = bool(self.get_parameter("debug_log").value)

        # ===== DeepSORT =====
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.n_init,
            nms_max_overlap=1.0,
            max_cosine_distance=self.max_cos_dist,
            embedder=("mobilenet" if self.use_embedder else None),
            half=self.embedder_half,
            bgr=True,
        )

        # ===== QoS =====
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        if YoloDetections is not None:
            self.sub_det = self.create_subscription(YoloDetections, self.yolo_msg, self.on_dets, qos)
        else:
            self.get_logger().error("identify/YoloDetections 消息类型缺失，无法订阅检测结果！")
            self.sub_det = None

        self.pub_tracks = self.create_publisher(DpSTracker, "/yolo_locked_track", 10)

        self.bridge = CvBridge() if CvBridge is not None else None
        self.latest_frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None

        if self.use_embedder and self.bridge is not None:
            self.sub_img = self.create_subscription(Image, self.image_topic, self.on_image, qos)
            self.get_logger().info(f"[track] image subscribe on {self.image_topic}")
        else:
            self.sub_img = None

        # ===== 锁定策略 =====
        self.locked_id: Optional[int] = None
        self.locked_class: Optional[str] = None
        self.lost_counter: int = 0
        self.max_lost_frames: int = 10  # 防抖动，连续丢失 N 帧后才认为目标丢失

        self.get_logger().info(
            f"DeepSORT UAV tracker started. target_class='{self.target_class}', score_thr={self.score_thr}, use_embedder={self.use_embedder}"
        )

    def on_image(self, msg: Image):
        if self.bridge is None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        with self.latest_frame_lock:
            self.latest_frame = frame

    def on_dets(self, msg: YoloDetections):
        dets_input = []  # [(xyxy), score, class_name]
        img_w, img_h = 0, 0

        for det in msg.detections:
            cls_name = str(getattr(det, "target_name", ""))
            score = float(getattr(det, "confidence", 0.0))
            if score < self.score_thr:
                continue
            if self.target_class and cls_name != self.target_class:
                continue
            x1, y1, x2, y2 = xyxy_from_det(det)
            dets_input.append(((x1, y1, x2, y2), score, cls_name))
            img_w = int(getattr(det, "image_width", img_w))
            img_h = int(getattr(det, "image_height", img_h))

        if not dets_input:
            self._handle_no_detections()
            return

        # 取最新帧用于外观嵌入
        frame = None
        if self.use_embedder and self.bridge is not None:
            with self.latest_frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

        try:
            tracks = self.tracker.update_tracks(dets_input, frame=frame)
        except Exception as e:
            self.get_logger().warn(f"tracker.update exception: {repr(e)}")
            tracks = []

        locked_found = False
        chosen_info = None

        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = int(t.track_id)
            x1, y1, x2, y2 = t.to_ltrb()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            label = t.get_det_class() or ""

            # 锁定逻辑：如果未锁定，遇到第一个目标就锁
            if self.locked_id is None and label == self.target_class:
                self.locked_id = tid
                self.locked_class = label
                self.lost_counter = 0
                if self.debug_log:
                    self.get_logger().info(f"[LOCK] id={self.locked_id}, class={self.locked_class}")

            # 如果已锁定，仅输出锁定目标
            if self.locked_id is not None and tid == self.locked_id:
                locked_found = True
                chosen_info = {
                    "id": tid,
                    "class": label,
                    "score": float(t.get_det_conf() or 0.0),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "center": [float(cx), float(cy)],
                }

        if self.locked_id is not None:
            if locked_found and chosen_info:
                self.lost_counter = 0
                self._publish_track(chosen_info)
            else:
                self.lost_counter += 1
                if self.lost_counter >= self.max_lost_frames:
                    if self.debug_log:
                        self.get_logger().warn(f"[LOST] id={self.locked_id}, releasing lock")
                    self.locked_id = None
                    self.locked_class = None

    def _publish_track(self, track_info: dict):
        msg = DpSTracker()
        msg.id = track_info["id"]
        msg.label = track_info["class"]
        msg.confidence = track_info["score"]
        msg.cx, msg.cy = tuple(track_info["center"])
        msg.x_min, msg.y_min, msg.x_max, msg.y_max = tuple(track_info["bbox_xyxy"])
        msg.area_ratio = abs((msg.x_max-msg.x_min) * (msg.y_max-msg.y_min)) / (1920.0*1080.0)

        self.pub_tracks.publish(msg)

    def _handle_no_detections(self):
        if self.locked_id is not None:
            self.lost_counter += 1
            if self.lost_counter >= self.max_lost_frames:
                if self.debug_log:
                    self.get_logger().warn(f"[LOST] id={self.locked_id}, releasing lock")
                self.locked_id = None
                self.locked_class = None


def main(args=None):
    rclpy.init(args=args)
    node = DeepSortUAVNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
