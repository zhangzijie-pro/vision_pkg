from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import deque
import math

from .utils import xyxy_to_ltwh, hsv_hist_signature, hist_distance

class LockOnDeepSort:
    """Wrapper around deep-sort-realtime with first-id lock + relock."""
    def __init__(self,
                 target_class: int,
                 max_age: int = 15,
                 n_init: int = 3,
                 nn_budget: int = 100,
                 max_cosine_distance: float = 0.2,
                 embedder: str = 'mobilenet',
                 half: bool = True,
                 bgr: bool = True):
        self.ds = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=1.0,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
            embedder=embedder,
            half=half,
            bgr=bgr
        )
        self.target_class = int(target_class)
        self.locked_id: Optional[int] = None
        self.lock_hist = None
        self.lock_bbox = None
        self.lost_frames = 0
        self.max_lost_relock = max_age

    def update(self, detections_xyxy: List[Tuple[list, float, int]], frame_bgr) -> Dict[str, Any]:
        # Pre-filter detections
        raw = []
        others = []
        for (xyxy, score, cls) in detections_xyxy:
            if score <= 0:
                continue
            x1,y1,x2,y2 = xyxy
            raw.append( ( [float(x1), float(y1), float(x2-x1), float(y2-y1)], float(score), int(cls) ) )
            others.append( {"xyxy": xyxy, "score": float(score), "class_id": int(cls)} )

        tracks = self.ds.update_tracks(raw, frame=frame_bgr, others=others)

        if self.locked_id is None:
            for t in tracks:
                if not t.is_confirmed():
                    continue
                det = t.get_det_supplementary()
                if det and det.get("class_id") == self.target_class:
                    self._set_lock(t.track_id, det["xyxy"], frame_bgr)
                    break

        locked_bbox = None
        lost = False
        if self.locked_id is not None:
            # Check if same id present
            found = False
            for t in tracks:
                if not t.is_confirmed():
                    continue
                if t.track_id == self.locked_id:
                    det = t.get_det_supplementary()
                    if det:
                        locked_bbox = det["xyxy"]
                        self.lock_bbox = locked_bbox
                        self.lock_hist = hsv_hist_signature(frame_bgr, locked_bbox)
                        found = True
                        self.lost_frames = 0
                        break
            if not found:
                # Relock using hist + spatial gating
                candidate = None
                best_score = 1e9
                for t in tracks:
                    if not t.is_confirmed():
                        continue
                    det = t.get_det_supplementary()
                    if not det or det.get("class_id") != self.target_class:
                        continue
                    cand_bbox = det["xyxy"]
                    cand_hist = hsv_hist_signature(frame_bgr, cand_bbox)
                    d_hist = hist_distance(self.lock_hist, cand_hist)
                    d_spatial = self._spatial_gate(self.lock_bbox, cand_bbox)
                    score = d_hist + 0.5*d_spatial
                    if score < best_score:
                        best_score = score
                        candidate = (t, cand_bbox)
                if candidate and best_score < 0.6:
                    t, cand_bbox = candidate
                    self.locked_id = t.track_id
                    locked_bbox = cand_bbox
                    self.lock_bbox = cand_bbox
                    self.lock_hist = hsv_hist_signature(frame_bgr, cand_bbox)
                    self.lost_frames = 0
                else:
                    self.lost_frames += 1
                    lost = True
                    if self.lost_frames > self.max_lost_relock:
                        self.locked_id = None
                        self.lock_hist = None
                        self.lock_bbox = None
                        self.lost_frames = 0

        return {"locked_id": self.locked_id, "bbox": locked_bbox, "lost": lost}

    def _spatial_gate(self, prev_bbox, cand_bbox):
        if prev_bbox is None or cand_bbox is None:
            return 1.0
        px1,py1,px2,py2 = prev_bbox
        cx1,cy1,cx2,cy2 = cand_bbox
        pcx, pcy = ( (px1+px2)*0.5, (py1+py2)*0.5 )
        ccx, ccy = ( (cx1+cx2)*0.5, (cy1+cy2)*0.5 )
        d = ((ccx-pcx)**2 + (ccy-pcy)**2) ** 0.5
        diag = ((px2-px1)**2 + (py2-py1)**2) ** 0.5 + 1e-6
        return float(d / diag)

    def _set_lock(self, track_id: int, bbox_xyxy, frame_bgr):
        self.locked_id = int(track_id)
        self.lock_bbox = bbox_xyxy
        self.lock_hist = hsv_hist_signature(frame_bgr, bbox_xyxy)
        self.lost_frames = 0
