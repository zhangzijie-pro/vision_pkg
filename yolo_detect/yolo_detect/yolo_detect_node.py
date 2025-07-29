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

import json
from typing import List, Union
import numpy as np

import cv2
import rclpy
from rclpy.node import Node
# from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import os


from .lib import YOLOv8_Detect
from identify.msg import YoloDetection, YoloDetections

names = ["drone"]


class Detect(Node):
    def __init__(self, name="yolo_detect_node"):
        super().__init__(name)

        # Declare parameters
        self.declare_parameter('yolo_detect_config_file', 'config.json')
        self.declare_parameter('feed_type', 1)
        self.declare_parameter('image', 'test.jpg')
        self.declare_parameter('pushlisher_node_name', 'Point_msg')
        self.declare_parameter('subscription_node_name', 'Camera_Image')

        # Get parameters
        feed_type = self.get_parameter('feed_type').value
        config_path = self.get_parameter('yolo_detect_config_file').value
        image_path = self.get_parameter('image').value
        self.publisher_topic = self.get_parameter('pushlisher_node_name').value
        self.subscription_topic = self.get_parameter('subscription_node_name').value

        self.get_logger().info(f"Feed Type: {feed_type}, Config Path: {config_path}, Image Path: {image_path}")

        # Read config and initialize
        self.model_config, self.cv_image, self.bridge = self._read_config(feed_type, config_path, image_path)
        self.model = YOLOv8_Detect(*self.model_config)
        self.model_path = self.model_config[0]
        self.score_thres = self.model_config[3]

        # Create publisher
        self.publisher = self.create_publisher(
            YoloDetections,
            self.publisher_topic,
            10
        )
        # self.array = Int16MultiArray()

        # If using static image
        if self.cv_image is not None:
            self.publish_msg(self.cv_image)

        self.get_logger().info(f"Detect Model initialized with model: {self.model_path}")

    def img_callback(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if cv_image is None:
            self.get_logger().error("Received empty image")
            return
        self.publish_msg(cv_image)

    def compress_img_callback(self, msg: CompressedImage):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.publish_msg(cv_image)

        
    def publish_msg(self, cv_image):
        input_tensor = self.model.preprocess_yuv420sp(cv_image)
        if input_tensor is None:
            self.get_logger().error("Failed to preprocess image")
            return

        outputs = self.model.c2numpy(self.model.forward(input_tensor))
        results = self.model.postProcess(outputs)

        # 构造新的检测消息
        msg = YoloDetections()
        msg.stamp = self.get_clock().now().to_msg()
        msg.detections = []

        for class_id, score, x1, y1, x2, y2 in results:
            # self.get_logger().info(f"Detected: class_id={class_id}, score={score}")
            if score < self.score_thres:
                continue

            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

            det = YoloDetection()
            det.class_id = class_id
            det.target_name = names[class_id] if class_id < len(names) else f"class_{class_id}"
            det.confidence = score
            det.cx, det.cy = mid_x, mid_y
            det.image_height, det.image_width = cv_image.shape[:2]
            det.x_min, det.y_min, det.x_max, det.y_max = x1, y1, x2, y2

            msg.detections.append(det)

            # self.array.data.extend([int(class_id), int(score * 100), int(mid_x), int(mid_y)])

        # self.publisher.publish(self.array)
        self.publisher.publish(msg)
        # self.publish_msg_data.detections.clear()

    # def _read_config(self, feed_type: int, file_path: str, image_path: str) -> List[Union[list, cv2.Mat, CvBridge]]:
    #     with open(file_path, 'r') as f:
    #         cfg = json.load(f)
    #         model_path = cfg['model_path']
    #         class_num = cfg['class_num']
    #         nms_threshold = cfg['nms_threshold']
    #         score_thres = cfg['score_thres']
    #         reg_max = cfg['reg_max']

    #     self.get_logger().info(
    #         f"Model Path: {model_path}, Class Num: {class_num}, "
    #         f"NMS Threshold: {nms_threshold}, Score Threshold: {score_thres}, Reg Max: {reg_max}"
    #     )

    #     config = [model_path, class_num, nms_threshold, score_thres, reg_max]

    #     if feed_type:
    #         bridge = CvBridge()
    #         if len(self.subscription_topic.spilt('/')) > 1:
    #             self.subscription = self.create_subscription(
    #                 CompressedImage,
    #                 self.subscription_topic,
    #                 self.compress_img_callback,
    #                 10
    #             )
    #         else:
    #             self.subscription = self.create_subscription(
    #                 Image,
    #                 self.subscription_topic,
    #                 self.img_callback,
    #                 10
    #             )
    #         self.get_logger().info("Subscribed to YOLO/Camera_Image topic")
    #         return [config, None, bridge]
    #     else:
    #         cv_image = cv2.imread(image_path)
    #         return [config, cv_image, None]
    def _read_config(self, feed_type: int, file_name: str, image_path: str) -> List[Union[list, cv2.Mat, CvBridge]]:
        # 获取安装路径
        try:
            pkg_path = get_package_share_directory('yolo_detect')  # ← 你实际的包名
        except Exception as e:
            self.get_logger().error(f"Could not find package path: {e}")
            raise e

        config_path = os.path.join(pkg_path, 'config', file_name)

        if not os.path.exists(config_path):
            self.get_logger().error(f"Config file not found at: {config_path}")
            raise FileNotFoundError(f"No such config: {config_path}")

        with open(config_path, 'r') as f:
            cfg = json.load(f)
            model_path = cfg['model_path']
            class_num = cfg['class_num']
            nms_threshold = cfg['nms_threshold']
            score_thres = cfg['score_threshold']
            reg_max = cfg['reg_max']

        self.get_logger().info(
            f"Model Path: {model_path}, Class Num: {class_num}, "
            f"NMS Threshold: {nms_threshold}, Score Threshold: {score_thres}, Reg Max: {reg_max}"
        )

        config = [model_path, class_num, nms_threshold, score_thres, reg_max]

        if feed_type:
            bridge = CvBridge()
            if len(self.subscription_topic.split('/')) > 1:
                self.subscription = self.create_subscription(
                    CompressedImage,
                    self.subscription_topic,
                    self.compress_img_callback,
                    10
                )
            else:
                self.subscription = self.create_subscription(
                    Image,
                    self.subscription_topic,
                    self.img_callback,
                    10
                )
            self.get_logger().info(f"Subscribed to {self.subscription_topic}")
            return [config, None, bridge]
        else:
            cv_image = cv2.imread(image_path)
            return [config, cv_image, None]


def main(args=None):
    rclpy.init(args=args)
    node = Detect()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
