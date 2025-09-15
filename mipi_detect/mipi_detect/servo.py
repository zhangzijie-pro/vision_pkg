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

参数(ros2 run vision_pkg track_deepsort --ros-args -p key:=val ...):
  queue_size: int = 2
  img_height: int = 1080
  subscription_topic: str = /yolo_detections
"""

import Hobot.GPIO as GPIO
import rclpy
from rclpy.node import Node
from identify.msg import YoloDetection, YoloDetections
import queue

output_pin = 32
GPIO.setwarnings(False)

class Servo_node(Node):
    def __init__(self, node_name="servo_node"):
        super().__init__(node_name)
        self.declare_parameter('queue_size', 2)
        self.declare_parameter('img_height', 1080)
        self.declare_parameter('subscription_topic', "/yolo_detections")

        self.queue_size = self.get_parameter('queue_size').value
        self.img_height = self.get_parameter('img_height').value
        self.subscription_topic = self.get_parameter('subscription_topic').value

        self.buffer = queue.Queue(self.queue_size)
        self._servo_init()
        self.servo_listen = self.create_subscription(
            YoloDetections,
            self.subscription_topic,
            self.listening,
            10
        )

    def listening(self,msg):
        if msg.detect_flag:
            self.p.stop()
            if self.buffer.full():
                self.buffer.get()
            self.buffer.put(msg, True)
        else:
            if not self.buffer.empty():
                last_msg = self.buffer.queue[-1]
                for det in last_msg.detections:
                    direction = det.cy - (self.img_height / 2)
                    self.control_servo(8) if direction>0 else self.control_servo(-8)

    def _servo_init(self):
        GPIO.setmode(GPIO.BOARD)
        self.p = GPIO.PWM(output_pin, 50)
        self.p.start(0)

    def control_servo(self, angle):
        # duty = (angle/180*20)+5
        angle = max(0, min(180, angle))
        duty = 2.5 + (angle / 180.0) * 10
        self.p.ChangeDutyCycle(duty)

    def destroy_node(self):
        if hasattr(self, 'p'):
            self.p.stop()
        GPIO.cleanup()
        self.get_logger().info("Servo Destroy.")
        super().destroy_node()

def main():
    rclpy.init()
    servo_node = Servo_node()
    try:
        rclpy.spin(servo_node)
    except KeyboardInterrupt:
        pass
    finally:
        servo_node.destroy_node()
        rclpy.shutdown()