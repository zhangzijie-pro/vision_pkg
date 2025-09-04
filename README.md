# vision_pkg

- backup: old code
- identify:  msg pkg
- mipi_detect: yolo model detect 
```bash
├── LICENSE
├── config
│   ├── config.json
│   ├── m_params.yaml
│   └── params.yaml
├── launch
│   ├── detect.launch.py
│   └── mutil_detect.launch.py
├── mipi_detect
│   ├── __init__.py
│   ├── detect_crop_thread_node.py
│   ├── detect_node.py
│   ├── lib.py
│   ├── track_deepsort.py
│   ├── tracker
│   │   ├── __init__.py
│   │   ├── basetrack.py
│   │   ├── byte_tracker.py
│   │   ├── kalman_filter.py
│   │   └── matching.py
│   └── utils.py
├── package.xml
├── resource
│   └── mipi_detect
├── setup.cfg
├── setup.py
└── test
```
- tools: pid code

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

```bash
source export.sh      # 导出install/config内容到{$PKG_NAME}_config
source back_in.sh	  # copy修改后的config内容到install
```