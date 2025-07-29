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
SDK RDK X5

from hobot_vio import libsrcampy

#create camera object
camera = libsrcampy.Camera()

#create encode object
encode = libsrcampy.Encoder()

#create decode object
decode = libsrcampy.Decoder()

#create display object
display = libsrcampy.Display()
"""


import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from hobot_vio import libsrcampy
import threading

class MipiCam(Node):
    def __init__(self, name="mipicam_node"):
        super().__init__(name)

        self.declare_parameter('publisher_img_topic_name', 'image_raw')
        self.declare_parameter('publisher_compressimg_topic_name', 'image_raw/compress_img')

        self.img_topic_name = self.get_parameter('publisher_img_topic_name').value
        self.compressimg_topic_name = self.get_parameter('publisher_compressimg_topic_name').value

        self.bridge = CvBridge()
        self.camera = libsrcampy.Camera()
        self.camera.open_cam(0, 1, 30, 1920, 1080)

        self.pub_raw = self.create_publisher(Image, self.img_topic_name, 10)
        self.pub_compressed = self.create_publisher(CompressedImage, self.compressimg_topic_name, 10)

        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        self.lock = threading.Lock()

        self.get_logger().info("MipiCam node initialized and publishing.")

    def timer_callback(self):
        with self.lock:
            nv12_img = self.camera.get_img(2)
            if nv12_img is None:
                self.get_logger().warn("未能读取摄像头图像")
                return

            try:
                h, w = 1080, 1920
                bgr_img = cv2.cvtColor(nv12_img.reshape((int(h * 1.5), w)), cv2.COLOR_YUV2BGR_NV12)

                img_msg = self.bridge.cv2_to_imgmsg(bgr_img, encoding="bgr8")
                img_msg.header.stamp = self.get_clock().now().to_msg()
                self.pub_raw.publish(img_msg)

                success, buffer = cv2.imencode('.jpg', bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not success:
                    self.get_logger().warn("图像 JPEG 编码失败")
                    return

                comp_msg = CompressedImage()
                comp_msg.header = img_msg.header
                comp_msg.format = "jpeg"
                comp_msg.data = buffer.tobytes()
                self.pub_compressed.publish(comp_msg)

            except Exception as e:
                self.get_logger().error(f"图像处理失败: {e}")

    def destroy_node(self):
        self.camera.close_cam()
        self.get_logger().info("Camera closed.")
        super().destroy_node()


def main():
    rclpy.init()
    node = MipiCam()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
