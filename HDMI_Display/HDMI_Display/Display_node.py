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

import random
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from hobot_vio import libsrcampy
from identify.msg import YoloDetections


def random_color(seed=None):
    if seed is not None:
        random.seed(seed)
    return tuple(random.randint(0, 255) for _ in range(3))


def draw_detection(img, bbox, score, label, color=None) -> None:
    x1, y1, x2, y2 = bbox
    color = color or random_color(hash(label) % 256)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    label_text = f"{label}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y), color, cv2.FILLED)
    cv2.putText(img, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


class DisplayNode(Node):
    def __init__(self):
        super().__init__("yolo_display_node")

        self.declare_parameter('image_rendering', 1)
        self.declare_parameter('subscription_yolo_topic_name', 'yolo_detections')
        self.declare_parameter('subscription_img_topic_name', 'image_raw')

        self.rendering_flag = self.get_parameter('image_rendering').value
        self.yolo_topic = self.get_parameter('subscription_yolo_topic_name').value
        self.image_topic = self.get_parameter('subscription_img_topic_name').value

        self.bridge = CvBridge()
        self.display = libsrcampy.Display()
        self.display.display(0, 1920, 1080, 0, 1)

        self.latest_detections = []
        self.current_frame = None

        # 订阅图像
        if '/' in self.image_topic:
            self.sub_image = self.create_subscription(CompressedImage, self.image_topic, self.compress_image_callback, 10)
        else:
            self.sub_image = self.create_subscription(Image, self.image_topic, self.image_callback, 10)

        # 订阅目标检测
        if self.rendering_flag:
            self.sub_yolo = self.create_subscription(YoloDetections, self.yolo_topic, self.yolo_callback, 10)

    def yolo_callback(self, msg):
        self.latest_detections = msg.detections
        self.get_logger().debug(f"Received {len(self.latest_detections)} detections")

    def handle_frame(self, frame):
        if frame is None:
            return

        if self.rendering_flag and self.latest_detections:
            for det in self.latest_detections:
                bbox = int(det.x_min), int(det.y_min), int(det.x_max), int(det.y_max)
                draw_detection(frame, bbox, det.confidence, det.target_name)

        self.display.set_img(frame)

    def compress_image_callback(self, msg):
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        self.handle_frame(frame)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.handle_frame(frame)
        except Exception as e:
            self.get_logger().error(f"CV bridge failed: {e}")

    def destroy_node(self):
        self.display.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = DisplayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down display node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
