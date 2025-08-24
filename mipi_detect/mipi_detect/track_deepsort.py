#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
  k_yaw: float = 0.003             # 像素误差->角速度
  k_z: float = 0.6                 # 面积误差->前后速度
  target_area_ref: float = 0.05    # 期望bbox面积/画面面积
  relock_on_lost: bool = True      # 丢失后允许重锁新目标
  cmd_rate_hz: float = 30.0        # 控制输出频率上限
  marker_text_size: float = 20.0
  debug_log: bool = False
"""

import json
import math
import threading
from collections import deque
from typing import List, Tuple, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from geometry_msgs.msg import Twist
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image

from deep_sort_realtime.deepsort_tracker import DeepSort

# 你的自定义消息
try:
    from identify.msg import YoloDetections  # 格式由你的包定义
except Exception:
    YoloDetections = None

# CV Bridge(可选，只有订阅图像时需要)
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

        # ===== param =====
        self.declare_parameter("target_class", "person")
        self.declare_parameter("score_threshold", 0.25)
        self.declare_parameter("max_age", 30)
        self.declare_parameter("n_init", 3)
        self.declare_parameter("max_cosine_distance", 0.2)
        self.declare_parameter("use_embedder", True)
        self.declare_parameter("embedder_half", True)
        self.declare_parameter("image_topic", "/image_raw")
        self.declare_parameter("k_yaw", 0.003)
        self.declare_parameter("k_z", 0.6)
        self.declare_parameter("target_area_ref", 0.05)
        self.declare_parameter("relock_on_lost", True)
        self.declare_parameter("cmd_rate_hz", 30.0)
        self.declare_parameter("marker_text_size", 20.0)
        self.declare_parameter("debug_log", False)

        self.target_class: str = str(self.get_parameter("target_class").value)
        self.score_thr: float = float(self.get_parameter("score_threshold").value)
        self.max_age: int = int(self.get_parameter("max_age").value)
        self.n_init: int = int(self.get_parameter("n_init").value)
        self.max_cos_dist: float = float(self.get_parameter("max_cosine_distance").value)
        self.use_embedder: bool = bool(self.get_parameter("use_embedder").value)
        self.embedder_half: bool = bool(self.get_parameter("embedder_half").value)
        self.image_topic: str = str(self.get_parameter("image_topic").value)
        self.k_yaw: float = float(self.get_parameter("k_yaw").value)
        self.k_z: float = float(self.get_parameter("k_z").value)
        self.area_ref: float = float(self.get_parameter("target_area_ref").value)
        self.relock_on_lost: bool = bool(self.get_parameter("relock_on_lost").value)
        self.cmd_rate_hz: float = float(self.get_parameter("cmd_rate_hz").value)
        self.marker_text_size: float = float(self.get_parameter("marker_text_size").value)
        self.debug_log: bool = bool(self.get_parameter("debug_log").value)

        # ===== DeepSORT =====
        # 使用 mobilenet 嵌入器（轻量、RDK X5 友好）；无图像时，DeepSORT自动退化到IoU/卡尔曼
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

        # ===== 话题 =====
        self.sub_det = self.create_subscription(YoloDetections, "/yolo_detections", self.on_dets, qos)

        self.pub_tracks = self.create_publisher(String, "/yolo_tracks", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/tracking_markers", 10)
        self.pub_cmd = self.create_publisher(Twist, "/uav/track_cmd", 10)

        # 可选的图像订阅（提升ReID稳健性 & 精度）
        self.bridge = CvBridge() if CvBridge is not None else None
        self.latest_frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.img_size = (0, 0)  # (w, h)

        if self.use_embedder and self.bridge is not None:
            self.sub_img = self.create_subscription(Image, self.image_topic, self.on_image, qos)
            self.get_logger().info(f"[track] image subscribe on {self.image_topic}")
        else:
            self.sub_img = None
            if self.use_embedder and self.bridge is None:
                self.get_logger().warn("cv_bridge missing; embedder will be disabled at runtime.")

        # ===== 锁定策略 =====
        self.locked_id: Optional[int] = None
        self.locked_class: Optional[str] = None

        # 控制输出限频
        self.timer_cmd = self.create_timer(1.0 / max(1.0, self.cmd_rate_hz), self._on_cmd_timer)
        self.pending_cmd: Optional[Twist] = None

        self.get_logger().info(
            f"DeepSORT UAV tracker started. target_class='{self.target_class}', score_thr={self.score_thr}, use_embedder={self.use_embedder}"
        )

    # ========= 图像回调 =========
    def on_image(self, msg: Image):
        if self.bridge is None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        with self.latest_frame_lock:
            self.latest_frame = frame
            h, w = frame.shape[:2]
            self.img_size = (w, h)

    # ========= 检测回调 =========
    def on_dets(self, msg: YoloDetections):
        dets_input = []  # [(xyxy), score, class_name]
        img_w = 0
        img_h = 0

        for det in msg.detections:
            cls_name = str(getattr(det, "target_name", ""))
            score = float(getattr(det, "confidence", 0.0))
            if score < self.score_thr:
                continue
            # 仅保留我们关心的类别（性能优先）
            if self.target_class and cls_name != self.target_class:
                continue
            x1, y1, x2, y2 = xyxy_from_det(det)
            dets_input.append(((x1, y1, x2, y2), score, cls_name))
            img_w = int(getattr(det, "image_width", img_w))
            img_h = int(getattr(det, "image_height", img_h))

        # 没有候选，清空可视化&停止控制
        if not dets_input:
            self._publish_tracks([])
            self._publish_markers([])
            self._queue_stop_cmd()
            return

        # 取最新帧用于外观嵌入（可选）
        frame = None
        if self.use_embedder and self.bridge is not None:
            with self.latest_frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()

        # DeepSORT 更新
        try:
            tracks = self.tracker.update_tracks(dets_input, frame=frame)
        except Exception as e:
            self.get_logger().warn(f"tracker.update exception: {repr(e)}")
            tracks = []

        img_area = max(1, img_w * img_h if img_w and img_h else (self.img_size[0] * self.img_size[1] or 1))

        # 收集可视化与输出
        track_list = []
        locked_found = False
        chosen = None  # (cx, cy, area_ratio, label, track_id, bbox)

        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = int(t.track_id)
            x1, y1, x2, y2 = t.to_ltrb()
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            area_ratio = (w * h) / img_area
            label = t.get_det_class() or ""

            track_list.append(
                {
                    "id": tid,
                    "class": label,
                    "score": float(t.get_det_conf() or 0.0),
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "center": [float(cx), float(cy)],
                    "area_ratio": float(area_ratio),
                }
            )

            # 锁定逻辑：如果未锁定，遇到第一个符合类别的就锁
            if self.locked_id is None and label == self.target_class:
                self.locked_id = tid
                self.locked_class = label
                if self.debug_log:
                    self.get_logger().info(f"[LOCK] id={self.locked_id}, class={self.locked_class}")

            # 如果已锁定，仅取锁定的
            if self.locked_id is not None and tid == self.locked_id:
                locked_found = True
                chosen = (cx, cy, area_ratio, label, tid, (x1, y1, x2, y2))

        # 锁定目标丢失 -> 允许重锁则清空等待下一次
        if self.locked_id is not None and not locked_found:
            if self.relock_on_lost:
                if self.debug_log:
                    self.get_logger().warn(f"[LOST] id={self.locked_id}, relock waiting...")
                self.locked_id = None
                self.locked_class = None
            else:
                # 不重锁时，直接停止控制
                self._queue_stop_cmd()

        # 发布轨迹与可视化
        self._publish_tracks(track_list)
        self._publish_markers(track_list)

        # 生成控制（仅在有 chosen 且锁定存在时）
        if chosen and self.locked_id is not None:
            cx, cy, area_ratio, label, tid, bbox = chosen
            img_w_eff = img_w if img_w else self.img_size[0]
            img_h_eff = img_h if img_h else self.img_size[1]
            if img_w_eff > 0 and img_h_eff > 0:
                ex = (cx - img_w_eff / 2.0)  # +右 -左
                ez = (self.area_ref - area_ratio)  # 面积误差：小->向前

                cmd = Twist()
                # 注意：根据你的机体/相机坐标，可能需要调整符号
                cmd.angular.z = -self.k_yaw * ex
                cmd.linear.x = self.k_z * ez
                cmd.linear.y = 0.0
                cmd.linear.z = 0.0

                self._queue_cmd(cmd)
            else:
                self._queue_stop_cmd()
        else:
            self._queue_stop_cmd()

    # ========= 发布：轨迹（JSON） =========
    def _publish_tracks(self, track_list: List[dict]):
        msg = String()
        msg.data = json.dumps({"locked_id": self.locked_id, "tracks": track_list}, ensure_ascii=False)
        self.pub_tracks.publish(msg)

    # ========= 发布：RViz 标注 =========
    def _publish_markers(self, track_list: List[dict]):
        ma = MarkerArray()
        now = self.get_clock().now().to_msg()
        for tr in track_list:
            tid = int(tr["id"])
            cls = str(tr["class"])
            x1, y1, x2, y2 = tr["bbox_xyxy"]
            cx, cy = tr["center"]
            # 文本标记
            m = Marker()
            m.header.frame_id = "camera"  # 像素平面
            m.header.stamp = now
            m.ns = "tracks"
            m.id = tid
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.pose.position.x = float(cx)
            m.pose.position.y = float(cy)
            m.pose.position.z = 0.0
            m.scale.z = self.marker_text_size
            locked_tag = " [LOCK]" if (self.locked_id is not None and tid == self.locked_id) else ""
            m.text = f"{cls}#{tid}{locked_tag}"
            ma.markers.append(m)
        self.pub_markers.publish(ma)

    # ========= 控制输出 限频 =========
    def _queue_cmd(self, cmd: Twist):
        self.pending_cmd = cmd

    def _queue_stop_cmd(self):
        # 只在已存在非零指令时发送停止，避免频繁 publish
        if self.pending_cmd is None:
            self.pending_cmd = Twist()

    def _on_cmd_timer(self):
        if self.pending_cmd is not None:
            self.pub_cmd.publish(self.pending_cmd)
            self.pending_cmd = None

def main(args=None):
    rclpy.init(args=args)
    node = DeepSortUAVNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()