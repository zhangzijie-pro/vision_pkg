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


def draw_detection(img, bbox, score, label) -> None:
    """
    Draws a detection bounding box and label on the image.

    Parameters:
        img (np.array): The input image.
        bbox (tuple[int, int, int, int]): A tuple containing the bounding box coordinates (x1, y1, x2, y2).
        score (float): The detection score of the object.
        class_id (int): The class ID of the detected object.
    """
    x1, y1, x2, y2 = bbox
    color = (random.randint(0,255) for _ in range(3))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{label}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


class DisplayNode(Node):
    def __init__(self):
        super().__init__("yolo_display_node")

        self.declare_parameter('subscription_yolo_topic_name', 'image_raw')
        self.declare_parameter('subscription_img_topic_name', 'yolo_detections')

        self.yolo_node = self.get_parameter('subscription_yolo_topic_name').value
        self.image_node = self.get_parameter('subscription_img_topic_name').value

        self.bridge = CvBridge()
        self.display = libsrcampy.Display()
        self.display.open()

        if len(self.image_node.split("/")) > 1:
            self.sub_image = self.create_subscription(CompressedImage, self.yolo_node, self.compress_image_callback, 10)
        else:
            self.sub_image = self.create_subscription(Image, self.yolo_node, self.image_callback, 10)
            
        self.sub_yolo = self.create_subscription(YoloDetections, self.image_raw, self.yolo_callback, 10)

        self.latest_detections = []
        self.current_frame = None

    def yolo_callback(self, msg):
        self.latest_detections = msg.detections

    def compress_image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        for det in self.latest_detections:
            bbox = int(det.x_min), int(det.y_min), int(det.x_max), int(det.y_max)
            label = det.target_name
            score = det.confidence

            draw_detection(frame, bbox, score, label)

        self.display.show(frame)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        for det in self.latest_detections:
            bbox = int(det.x_min), int(det.y_min), int(det.x_max), int(det.y_max)
            label = det.target_name
            score = det.confidence

            draw_detection(frame, bbox, score, label)

        self.display.show(frame)

    def destroy_node(self):
        self.display.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = DisplayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
