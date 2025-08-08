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

import threading
import queue
import time

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
        # self.declare_parameter('feed_type', 1)
        # self.declare_parameter('image', 'test.jpg')
        self.declare_parameter('pushlisher_node_name', '/yolo_detections')

        # Get parameters
        # feed_type = self.get_parameter('feed_type').value
        # image_path = self.get_parameter('image').value
        config_path = self.get_parameter('yolo_detect_config_file').value
        self.publisher_topic = self.get_parameter('pushlisher_node_name').value
        self.publisher = self.create_publisher(
            YoloDetections,
            "/yolo_detections",
            10
        )
        
        self.camera = libsrcampy.Camera()
        self.model_config= self._read_config(config_path)
        self.model = YOLOv8_Detect(*self.model_config)
        self.model_path = self.model_config[0]
        self.score_thres = self.model_config[3]

        # Get model input size
        self.h, self.w = self.model.input_H, self.model.input_W
        # self.h, self.w = 1080, 1920
        self.sensor_h, self.sensor_w = 1080, 1920  # Modify based on actual sensor config
        # self.camera.open_cam(0, -1, -1, [self.w, disp_w], [self.h, disp_h], self.sensor_h, self.sensor_w)
        self.camera.open_cam(0, -1, 30, [self.w, disp_w], [self.h, disp_h], self.sensor_h, self.sensor_w)
        # self.camera.open_cam(0, -1, 30, self.w, self.h, self.sensor_h, self.sensor_w)


        self.disp = libsrcampy.Display()
        self.disp.display(0, disp_w, disp_h)
        libsrcampy.bind(self.camera, self.disp)
        self.disp.display(3, disp_w, disp_h)

        self.draw_queue = queue.Queue(maxsize=3)
        self._stop_event = threading.Event()
        self._disp_lock = threading.Lock()

        # 启动绘制线程
        self._draw_thread = threading.Thread(target=self._draw_worker, daemon=True)
        self._draw_thread.start()


        self.timer = self.create_timer(1.0 / 30.0, self.time_callback)


        self.data = []
        self.get_logger().info(f"publisher Name: {self.publisher_topic}")

        self.get_logger().info(f"Config Path: {config_path}")
        self.get_logger().info(f"display width: {disp_w}, display height: {disp_h}")
        
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
        draw_batch = []
        msg = YoloDetections()
        msg.stamp = self.get_clock().now().to_msg()
        # msg.detections = []

        for class_id, score, x1, y1, x2, y2 in results:
            if score < self.score_thres or class_id != 0:
                continue

            bbox = (x1, y1, x2, y2)
            (x1, y1, x2, y2) = self.scale_mask(bbox)
            # draw_detection(cv_image, bbox, score, class_id)
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

            det = YoloDetection()
            # det.class_id = class_id
            det.target_name = names[class_id] if class_id < len(names) else f"class_{class_id}"
            det.confidence = float(score)
            det.cx, det.cy = mid_x, mid_y
            det.image_height, det.image_width = cv_image.shape[:2]
            det.x_min, det.y_min, det.x_max, det.y_max = x1, y1, x2, y2
            msg.detections.append(det)

            self.get_logger().info(f"name: {det.target_name}")
            self.get_logger().info(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
            

            # self.data.append([bbox,det.target_name])
            # self.draw_hardware_rect()

            rect_item = {
                "type": "rect",
                "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                "thickness": 3, "mode": 1, "color": 0xffff00ff
            }
            text_item = {
                "type": "text",
                "x": int(x1), "y": int(max(0, y1 - 2)),
                "text": f"{det.target_name} {score:.2f}",
                "mode": 1, "color": 0xffff00ff
            }
            draw_batch.append(rect_item)
            draw_batch.append(text_item)

            # Display overlay via hardware

            # self.disp.set_graph_rect(x1, y1, x2, y2, 3, 1, 0xffff00ff)
            # label = f"{det.target_name} {score:.2f}"
            # label = label.encode('gb2312')
            # self.disp.set_graph_word(x1, y1 - 2, label, 3, 1, 0xffff00ff)
        
        if len(msg.detections) >=1:
            if len(draw_batch) > 0:
                try:
                    self.draw_queue.put_nowait(draw_batch)
                except queue.Full:
                    self.get_logger().warning("draw_queue full, dropping this frame's draw batch")
            self.publisher.publish(msg)
            # self.get_logger().info("Send successfully")
        # else:
        #     det = YoloDetection()
        #     det.target_name="none"
        #     det.confidence = float(0.7)
        #     det.cx, det.cy = 544, 684
        #     det.image_height, det.image_width = 1920,1080
        #     det.x_min, det.y_min, det.x_max, det.y_max = 544, 648,457,129
        #     msg.detections.append(det)
        #     self.publisher.publish(msg)

        self.get_logger().debug(f"msg: {msg}")   
    
    def _draw_worker(self):
        while not self._stop_event.is_set():
            try:
                draw_batch = self.draw_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                with self._disp_lock:
                    for item in draw_batch:
                        if item["type"] == "rect":
                            # set_graph_rect(x1,y1,x2,y2, thickness, mode, color)
                            try:
                                self.disp.set_graph_rect(
                                    item["x1"], item["y1"], item["x2"], item["y2"],
                                    item.get("thickness", 3), item.get("mode", 1), item.get("color", 0xffff00ff)
                                )
                            except Exception as e:
                                self.get_logger().warning(f"disp.set_graph_rect failed: {e}")
                        elif item["type"] == "text":
                            try:
                                text_bytes = item["text"].encode('gb2312', errors='ignore')
                                self.disp.set_graph_word(
                                    item["x"], item["y"], text_bytes,
                                    item.get("mode", 1), 1, item.get("color", 0xffff00ff)
                                )
                            except Exception as e:
                                self.get_logger().warning(f"disp.set_graph_word failed: {e}")
            finally:
                try:
                    self.draw_queue.task_done()
                except Exception:
                    pass


    def draw_hardware_rect(self):
        if self.data is None:
            return
        
        for index, result in enumerate(self.data):
            bbox = result[0]
            label = result[1]

            label = label.encode('gb2312')
            box_color_ARGB = 0xffff00ff

            if index == 0:
                self.disp.set_graph_rect(
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    3, 0,box_color_ARGB)
                self.disp.set_graph_word(
                    bbox[0], bbox[3], label,
                    3, 0, box_color_ARGB)
            else:
                self.disp.set_graph_rect(
                    bbox[0], bbox[1], bbox[2], bbox[3],
                    3, 0, box_color_ARGB)
                self.disp.set_graph_word(
                    bbox[0], bbox[3], label,
                    3, 0, box_color_ARGB)
                

    def _read_config(self, file_name: str) -> List[Union[list]]:
        try:
            pkg_path = get_package_share_directory('mipi_detect')
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

        # cv_image = cv2.imread(image_path)
        return config

    def scale_mask(self, bbox):
        x_scale = 1.0 * (disp_w / self.w)
        y_sclae = 1.0 * (disp_h / self.h)

        x1,x2 = bbox[0]*x_scale, bbox[2]*x_scale
        y1,y2 = bbox[1]*y_sclae, bbox[3]*y_sclae

        return (int(x1), int(y1), int(x2), int(y2))


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