import rclpy
from rclpy.node import Node
from identify.msg import YoloDetection, YoloDetections
from builtin_interfaces.msg import Time


class MsgNode(Node):
    def __init__(self):
        super().__init__('msg_node')
        self.get_logger().info('msg_node has been started')

        # 订阅话题
        self.subscription = self.create_subscription(
            YoloDetections,
            'yolo_detections',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg: YoloDetections):
        self.get_logger().info(f"Received YoloDetections at time: {msg.stamp.sec}.{msg.stamp.nanosec}")
        for det in msg.detections:
            self.get_logger().info(
                f"Target: {det.target_name}, Confidence: {det.confidence:.2f}, "
                f"Center: ({det.cx}, {det.cy}), "
                f"Image Size: {det.image_width}x{det.image_height}, "
                f"BBox: ({det.x_min}, {det.y_min}) to ({det.x_max}, {det.y_max})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = MsgNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
