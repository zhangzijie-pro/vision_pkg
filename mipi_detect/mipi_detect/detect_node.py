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
发布:
  /yolo_detections (identify/msg/YoloDetections)
  /image_raw       (sensor_msgs/Image)

参数(ros2 run mipi_detect mipi_detect_node --ros-args -p key:=val ...):
  yolo_detect_config_file: str = "config.json"      # 模型配置文件地址
  publisher_node_name: str = "/yolo_detections"
  publisher_img_topic_name: str = "/image_raw"
"""

import json
from typing import List, Union
import numpy as np

import cv2
import rclpy
from rclpy.node import Node
# from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import os
from hobot_vio import libsrcampy


from identify.msg import YoloDetection, YoloDetections

# from .utils import *
from .lib import YOLOv8_Detect, draw_detection
# from .tracker.byte_tracker import BYTETracker

sensor_width = 1920
sensor_height = 1080


log_file = "$HOME/detect_log.log"

names_class1 = ["drone"]
names_class80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

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
        global names 

        # Declare parameters
        self.declare_parameter('yolo_detect_config_file', 'config.json')
        self.declare_parameter('publisher_img_topic_name', '/image_raw')
        self.declare_parameter('pushlisher_node_name', '/yolo_detections')

        # Get parameters
        config_path = self.get_parameter('yolo_detect_config_file').value
        self.img_topic_name = self.get_parameter('publisher_img_topic_name').value
        self.publisher_topic = self.get_parameter('pushlisher_node_name').value

        self.publisher = self.create_publisher(
            YoloDetections,
            self.publisher_topic,
            10
        )
        self.pub_raw = self.create_publisher(
            Image, 
            self.img_topic_name, 
            10
        )

        self.bridge = CvBridge()
        self.camera = libsrcampy.Camera()
        self.model_config= self._read_config(config_path)
        names = names_class1 if self.model_config[1]==1 else names_class80
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

        # self.byte_tracker = BYTETracker(self._byte_tracker_args(), frame_rate=30)
        # self.trajectories = {}
        # self.max_traj_len = 10

        self.disp = libsrcampy.Display()
        self.disp.display(0, disp_w, disp_h)
        libsrcampy.bind(self.camera, self.disp)
        self.disp.display(3, disp_w, disp_h)


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
        
        # self.publish_tracker_msg(bgr_img)
        
        
    def publish_msg(self, cv_image):
        """
        pushlish origin yolo result
        Returns:
            msg:
                YoloDetections()

                YoloDetection()
        """
        input_tensor = self.model.preprocess_yuv420sp(cv_image)
        if input_tensor is None:
            self.get_logger().error("Failed to preprocess image")
            return

        outputs = self.model.c2numpy(self.model.forward(input_tensor))
        results = self.model.postProcess(outputs)

        msg = YoloDetections()
        msg.stamp = self.get_clock().now().to_msg()
        # msg.detections = []

        for class_id, score, x1, y1, x2, y2 in results:
            if score < self.score_thres:
                continue

            bbox = (x1, y1, x2, y2)
            (x1, y1, x2, y2) = self.scale_mask(bbox)
            # draw_detection(cv_image, bbox, score, class_id)
            mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2

            det = YoloDetection()
            if class_id == 0:
            # det.class_id = class_id
                msg.detect_flag = True
                det.target_name = names[class_id] if class_id < len(names) else f"class_{class_id}"
                det.confidence = float(score)
                det.cx, det.cy = mid_x, mid_y
                det.image_height, det.image_width = cv_image.shape[:2]
                det.x_min, det.y_min, det.x_max, det.y_max = x1, y1, x2, y2
                msg.detections.append(det)

            # self.get_logger().debug(f"name: {det.target_name}")
            # self.get_logger().debug(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
            

            # self.data.append([bbox,det.target_name])
            # self.draw_hardware_rect()

            # Display overlay via hardware
            self.disp.set_graph_rect(det.x_min, det.y_min, det.x_max, det.y_max, 3, 1, 0xffff00ff)
            # label = f"{det.target_name} {score:.2f}"
            # label = label.encode('gb2312')

            # self.disp.set_graph_word(x1, y1 - 2, label, 3, 1, 0xffff00ff)
        
        if len(msg.detections) >=1:
            self.publisher.publish(msg)
            self.record_time(msg)
        else:
            msg.detect_flag = False
            self.publisher.publish(msg)
            self.record_time(msg)
        
        img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.pub_raw.publish(img_msg)

        # self.get_logger().debug(f"msg: {msg}")   


    def record_time(self, msg):
        path = os.path.expandvars(log_file)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode="a", encoding="utf-8") as f:
            f.write(f"{msg.stamp.sec}.{str(msg.stamp.nanosec).zfill(9)}\n")
        
        
    # def publish_tracker_msg(self, cv_image):
    #     """
    #     publish byte_tracker msg
        
    #     Returns:
    #         msg:
    #             YoloDetections()

    #             YoloDetection()
    #     """
    #     input_tensor = self.model.preprocess_yuv420sp(cv_image)
    #     if input_tensor is None:
    #         self.get_logger().error("Failed to preprocess image")
    #         return

    #     outputs = self.model.c2numpy(self.model.forward(input_tensor))
    #     results = self.model.postProcess(outputs)

    #     msg = YoloDetections()
    #     msg.stamp = self.get_clock().now().to_msg()
    #     msg.detections = []

    #     detections_for_tracker = []
    #     class_ids = []

    #     for class_id, score, x1, y1, x2, y2 in results:
    #         if score < self.score_thres:
    #             continue

    #         bbox = (x1, y1, x2, y2)
    #         (x1, y1, x2, y2) = self.scale_mask(bbox)

    #         # target_name = names[class_id] if class_id < len(names) else f"class_{class_id}"

    #         # if target_name == 'person':
    #         detections_for_tracker.append([x1, y1, x2, y2, score])
    #         class_ids.append(class_id)
            
    #     if detections_for_tracker:
    #         detections_for_tracker = np.array(detections_for_tracker, dtype=np.float32)
    #     else:
    #         detections_for_tracker = np.empty((0, 5), dtype=np.float32)

    #     # ======================  ! ! !  =========================
    #     tracks = self.byte_tracker.update(
    #         detections_for_tracker, 
    #         img_info=[self.sensor_h, self.sensor_w], 
    #         img_size=[self.sensor_h, self.sensor_w]
    #     )

    #     for track in tracks:
    #         box = track.tlbr
    #         track_id = track.track_id
    #         center = (int((box[0]+box[2]) / 2), int((box[1]+box[3]) / 2))

    #         if track_id not in self.trajectories:
    #             self.trajectories[track_id] = []
    #         self.trajectories[track_id].append(center)

    #         if len(self.trajectories[track_id]) > self.max_traj_len:
    #             self.trajectories[track_id] = self.trajectories[track_id][-self.max_traj_len:]
            
    #         if len(detections_for_tracker)>0:
    #             ious = self.iou(box, detections_for_tracker[:,:4])
    #             max_index = np.argmax(ious)
    #             class_id = class_ids[max_index]
    #         else:
    #             class_id = -1
            

    #         if class_id == 62:
    #             det = YoloDetection()
    #             msg.detect_flag = True
    #             det.class_id = class_id
    #             det.target_name = names[class_id] if class_id < len(names) else f"class_{class_id}"
    #             det.confidence = float(detections_for_tracker[max_index][4]) if len(detections_for_tracker) > 0 else 0
    #             det.cx, det.cy = center
    #             det.image_height, det.image_width = cv_image.shape[:2]
    #             det.x_min, det.y_min, det.x_max, det.y_max = map(int, box)
    #             msg.detections.append(det)


    #         self.disp.set_graph_rect(int(box[0]), int(box[1]), int(box[2]), int(box[3]), 3, 1, 0xffff00ff)
    #         # label = f"{det.target_name}#{track_id} {det.confidence:.2f}"
    #         # self.disp.set_graph_word(int(box[0]), int(box[1]) - 2, label.encode('gb2312'), 3, 1, 0xffff00ff)

    #         # draw point line to show trajectory
    #         # traj_points = self.trajectories[track_id]
    #         # for i in range(1, len(traj_points)):
    #         #     self.disp.set_graph_line(traj_points[i-1][0], traj_points[i-1][1],
    #         #                         traj_points[i][0], traj_points[i][1],
    #         #                         3, 1, 0x00ff00ff)
            
    #     if len(msg.detections) >=1:
    #         self.publisher.publish(msg)
    #     else:
    #         msg.detect_flag = False
    #         self.publisher.publish(msg)

    #     img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
    #     img_msg.header.stamp = self.get_clock().now().to_msg()
    #     self.pub_raw.publish(img_msg)

    #     # self.get_logger().debug(f"msg: {msg}")   


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
        y_scale = 1.0 * (disp_h / self.h)

        x1,x2 = bbox[0]*x_scale, bbox[2]*x_scale
        y1,y2 = bbox[1]*y_scale, bbox[3]*y_scale

        return (int(x1), int(y1), int(x2), int(y2))

    # def _byte_tracker_args(self):
    #     class Args:
    #         track_thresh = 0.25
    #         track_buffer = 30
    #         match_thresh = 0.8
    #         aspect_ratio_thresh = 3.0
    #         min_box_area = 1.0
    #         mot20 = False
    #     return Args()

    # def iou(self, box, boxes):
    #     xy_max = np.minimum(boxes[:, 2:], box[2:])
    #     xy_min = np.maximum(boxes[:, :2], box[:2])
    #     inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
    #     inter = inter[:, 0] * inter[:, 1]
    #     area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    #     area_box = (box[2] - box[0]) * (box[3] - box[1])
    #     return inter / (area_box + area_boxes - inter)


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