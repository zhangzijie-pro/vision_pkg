# vision_pkg

- identify:  msg pkg
- yolo_detect: yolo model detect 
- mipi_camera : cam
- HDMI-Display: 
- robot_bringup: 


```bash
ros2 topic pub /yolo_detections identify/msg/YoloDetections "{stamp: {sec: 123, nanosec: 456}, detections: [{target_name: 'person', confidence: 0.92, cx: 100, cy: 120, image_height: 480, image_width: 640, x_min: 80, y_min: 100, x_max: 120, y_max: 140}]}"

```