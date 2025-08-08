# vision_pkg

- identify:  msg pkg
- yolo_detect: yolo model detect 
- mipi_camera : cam
- HDMI-Display: 
- robot_bringup: 

## close HDMI 
``` bash
sudo systemctl stop lightdm
```

```bash
ros2 topic pub /yolo_detections identify/msg/YoloDetections "{stamp: {sec: 123, nanosec: 456}, detections: [{target_name: 'person', confidence: 0.92, cx: 100, cy: 120, image_height: 480, image_width: 640, x_min: 80, y_min: 100, x_max: 120, y_max: 140}]}"

```

```bash
{
	"model_path": "/app/pydev_demo/drone_yolov8n_detect_bayese_640x640_nv12/drone_yolov8n_detect_bayese_640x640_nv12_modified.bin",
	"class_num": 1,
	"nms_threshold": 0.7,
	"score_threshold": 0.25,
	"reg_max": 16
}
```