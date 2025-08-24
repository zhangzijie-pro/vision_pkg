import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import json
import cv2

from .camera_rdksrcam import RdkMipiCamera
from .deepsort_lockon import LockOnDeepSort
from .detectors import DetectorAdapter

class TrackAndControlNode(Node):
    def __init__(self):
        super().__init__('track_and_control')
        self.declare_parameter('target_class', 0)
        self.declare_parameter('conf_thres', 0.35)
        self.declare_parameter('cam_id', 0)
        self.declare_parameter('width', 960)
        self.declare_parameter('height', 544)
        self.declare_parameter('visualize', False)
        self.declare_parameter('embedder', 'mobilenet')
        self.declare_parameter('max_age', 15)
        self.declare_parameter('n_init', 3)

        target_class = int(self.get_parameter('target_class').value)
        conf_thres = float(self.get_parameter('conf_thres').value)
        cam_id = int(self.get_parameter('cam_id').value)
        w = int(self.get_parameter('width').value)
        h = int(self.get_parameter('height').value)
        visualize = bool(self.get_parameter('visualize').value)
        embedder = str(self.get_parameter('embedder').value)
        max_age = int(self.get_parameter('max_age').value)
        n_init = int(self.get_parameter('n_init').value)

        self.tracker = LockOnDeepSort(target_class=target_class, embedder=embedder, max_age=max_age, n_init=n_init)
        self.detector = DetectorAdapter(conf_thres=conf_thres, class_whitelist=set([target_class]))
        self.camera = RdkMipiCamera(cam_id, w, h)
        self.visualize = visualize

        self.pub_cmd = self.create_publisher(Twist, '/uav/track_cmd', 10)
        self.pub_info = self.create_publisher(String, '/uav/track_info', 10)

        self.timer = self.create_timer(0.0 if not visualize else 0.001, self.loop_once)
        self.get_logger().info(f"TrackAndControlNode started. Lock-on target_class={target_class}")

    def loop_once(self):
        frame = self.camera.read_bgr()
        if frame is None:
            return

        h, w = frame.shape[:2]
        dets = self.detector.infer(frame)
        dets = [ (xyxy,score,cls) for (xyxy,score,cls) in dets if score >= self.detector.conf_thres and (self.detector.class_whitelist is None or cls in self.detector.class_whitelist) ]

        state = self.tracker.update(dets, frame)
        msg = Twist()
        info = {"locked_id": state["locked_id"], "bbox": None, "class_id": None, "lost": state["lost"]}

        if state["locked_id"] is not None and state["bbox"] is not None:
            x1,y1,x2,y2 = state["bbox"]
            cx = 0.5*(x1+x2)
            cy = 0.5*(y1+y2)
            area = (x2-x1)*(y2-y1) / float(w*h + 1e-6)

            err_x = (cx - w/2) / (w/2)
            err_y = (cy - h/2) / (h/2)

            msg.linear.x = float(err_x)
            msg.linear.y = float(err_y)
            msg.linear.z = float(area)

            info["bbox"] = [float(x1),float(y1),float(x2),float(y2)]
            info["class_id"] = int(self.tracker.target_class)

        self.pub_cmd.publish(msg)
        s = String()
        s.data = json.dumps(info)
        self.pub_info.publish(s)

        if self.visualize:
            self._draw_and_show(frame, state)

    def _draw_and_show(self, frame, state):
        if state["bbox"] is not None:
            x1,y1,x2,y2 = [int(v) for v in state["bbox"]]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, f"id={state['locked_id']}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("track", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = TrackAndControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
