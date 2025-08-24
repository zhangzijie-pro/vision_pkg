# vision_pkg_deepsort (RDK X5, ROS 2, DeepSORT)
Lock the **first seen target** by class and stay on it — even after brief occlusions or ID switches — using DeepSORT + lightweight relock logic.

## What you get
- **RDK MIPI camera** capture using `hobot_vio.libsrcampy` (no `/image_raw` needed).
- **DeepSORT** (mobilenet embedder by default) for stable tracking.
- **Lock‑on policy**: first track of `target_class` becomes the only target (`locked_id`).
- **Re‑lock after loss**: HSV color histogram + spatial gating to recover the same physical target if DeepSORT reinitializes with a new ID.
- **ROS 2 output**: publishes `/uav/track_cmd` (geometry_msgs/Twist) and `/uav/track_info` (std_msgs/String JSON) for your flight controller.

## Quick start
```bash
# (on RDK X5)
python3 -m pip install -r requirements.txt

# Source ROS 2 (TROS) env first (foxy or humble)
source /opt/tros/setup.bash   # or /opt/tros/humble/setup.bash

# Run (parameters are optional)
ros2 run vision_pkg_deepsort track_and_control   --ros-args -p target_class:=person -p conf_thres:=0.35 -p cam_id:=0 -p visualize:=false
```

## Using your detector
Edit `vision_pkg_deepsort/detectors/detector_adapter.py` and implement `DetectorAdapter.infer(frame_bgr)` to return a list of detections:

```python
# each detection: (xyxy, score, class_id)
# xyxy: [x1, y1, x2, y2] in absolute pixels
return [([x1,y1,x2,y2], float(score), int(class_id)), ...]
```

If you already have a function with signature like `def detect(frame_bgr) -> List[(xyxy,score,class_id)]`, just call it inside `infer()`.
Set `target_class` param to the class ID you want (e.g. `person=0`).

## Topics
- `/uav/track_cmd` : `geometry_msgs/Twist`
    - `linear.x` = normalized horizontal error (right positive)
    - `linear.y` = normalized vertical error (down positive)
    - `linear.z` = bbox area ratio (for distance/altitude control if you want)
- `/uav/track_info` : `std_msgs/String` JSON
    - `{"locked_id": int, "bbox":[x1,y1,x2,y2], "score":float, "class_id":int, "lost": bool}`

## Performance notes (X5)
- NV12→BGR uses OpenCV `cvtColor` (`COLOR_YUV2BGR_NV12`), minimal copies.
- You can disable visualisation via `-p visualize:=false` for max FPS.
- Tune DeepSORT: `max_age`, `n_init`, `nn_budget`, `max_cosine_distance` in `deepsort_lockon.py`.
- If you track **persons**, consider `embedder='torchreid'` (heavier but stronger re-id).

## Known limitations
- Exact libsrcampy camera APIs differ slightly across RDK releases. We handle the common path (NV12 buffer) and provide a fallback. Adjust small API names if needed.
- For best re-lock, ensure target has distinct color/texture; otherwise rely on DeepSORT ID persistence.

## References
- deep-sort-realtime usage and API (see PyPI / GitHub).
- RDK MIPI camera & NV12 hints (mipi_cam & NV12 topics).
