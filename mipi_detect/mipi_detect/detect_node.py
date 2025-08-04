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
from ament_index_python.packages import get_package_share_directory
import os
from hobot_vio import libsrcampy


from .lib import YOLOv8_Detect, draw_detection
from identify.msg import YoloDetection, YoloDetections

sensor_width = 1920
sensor_height = 1080
names = ["drone"]

h, w = 1080, 1920

def get_display_res():
    disp_w_small=1920
    disp_h_small=1080
    disp = libsrcampy.Display()
    resolution_list = disp.get_display_res()
    if (sensor_width, sensor_height) in resolution_list:
        print(f"Resolution {sensor_width}x{sensor_height} exists in the list.")
        return int(sensor_width), int(sensor_height)
    else:
        print(f"Resolution {sensor_width}x{sensor_height} does not exist in the list.")
        for res in resolution_list:
            # Exclude 0 resolution first.
            if res[0] == 0 and res[1] == 0:
                break
            else:
                disp_w_small=res[0]
                disp_h_small=res[1]

            # If the disp_w、disp_h is not set or not in the list, default to iterating to the smallest resolution for use.
            if res[0] <= sensor_width and res[1] <= sensor_height:
                print(f"Resolution {res[0]}x{res[1]}.")
                return int(res[0]), int(res[1])

    disp.close()
    return disp_w_small, disp_h_small

disp_w, disp_h = get_display_res()

class Detect(Node):
    def __init__(self, name="mipi_detect_node"):
        super().__init__(name)

        # Declare parameters
        self.declare_parameter('yolo_detect_config_file', 'config.json')
        self.declare_parameter('feed_type', 1)
        self.declare_parameter('image', 'test.jpg')
        self.declare_parameter('pushlisher_node_name', 'ai_msg')

        # Get parameters
        feed_type = self.get_parameter('feed_type').value
        config_path = self.get_parameter('yolo_detect_config_file').value
        image_path = self.get_parameter('image').value
        self.publisher_topic = self.get_parameter('pushlisher_node_name').value

        self.camera = libsrcampy.Camera()
        self.model = YOLOv8_Detect(*self.model_config)
        self.model_path = self.model_config[0]
        self.score_thres = self.model_config[3]

        # Get model input size
        self.h, self.w = self.model.input_height, self.model.input_width
        self.sensor_h, self.sensor_w = 1080, 1920  # Modify based on actual sensor config
        self.camera.open_cam(0, -1, -1, [self.w, disp_w], [self.h, disp_h], self.sensor_h, self.sensor_w)

        self.disp = libsrcampy.Display()
        self.disp.display(0, disp_w, disp_h)
        libsrcampy.bind(self.camera, self.disp)
        self.disp.display(3, disp_w, disp_h)
        
        self.timer = self.create_timer(1.0 / 30.0, self.time_callback)
        
        self.get_logger().info(f"Feed Type: {feed_type}, Config Path: {config_path}, Image Path: {image_path}")
        
        self.get_logger().info(f"Detect Model initialized with model: {self.model_path}")
        
    def time_callback(self):
        nv12_img = self.camera.get_img(2, self.w, self.h)
        if nv12_img is None:
            self.get_logger().warn("Can't read camera image. it's None")
            return

        nv12_img = np.frombuffer(nv12_img, dtype=np.uint8)
        bgr_img = cv2.cvtColor(nv12_img.reshape((int(self.h * 1.5), self.w)), cv2.COLOR_YUV2BGR_NV12)

        # Display with bounding boxes (drawn later in publish_msg)
        self.publish_msg(bgr_img)
        
        
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
            if score < self.score_thres:
                continue

            #bbox = (x1, y1, x2, y2)
            # draw_detection(cv_image, bbox, score, class_id)
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

            det = YoloDetection()
            det.class_id = class_id
            det.target_name = names[class_id] if class_id < len(names) else f"class_{class_id}"
            det.confidence = score
            det.cx, det.cy = mid_x, mid_y
            det.image_height, det.image_width = cv_image.shape[:2]
            det.x_min, det.y_min, det.x_max, det.y_max = x1, y1, x2, y2
            msg.detections.append(det)

            # Display overlay via hardware
            self.disp.set_graph_rect(x1, y1, x2, y2, 3, 1, (0, 255, 0))
            label = f"{det.target_name} {score:.2f}"
            self.disp.set_graph_word(x1, y1 - 2, label, 3, 1 (0, 255, 0))

        self.publisher.publish(msg)

    def destroy_node(self):
        self.camera.close_cam()
        self.disp.close()
        super().destroy_node()
        self.get_logger().info("Camera detect Destory.")
        

def main(args=None):
    rclpy.init(args=args)
    node = Detect()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()