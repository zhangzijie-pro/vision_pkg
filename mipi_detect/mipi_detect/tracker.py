import cv2
import numpy as np
from collections import deque, defaultdict
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18

image_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

def iou(bb1, bb2):
    """
    bb: bbox [x1,y1,x2,y2]
    """
    xa1, ya1, xa2, ya2 = bb1
    xb1, yb1, xb2, yb2 = bb2
    w1 = xa2 - xa1
    h1 = ya2 - ya1
    w2 = xb2 - xb1
    h2 = yb2 - yb1
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    iw = max(0, xi2 - xi1)
    ih = max(0, yi2 - yi1)
    inter = iw * ih
    union = (w1 * h1) + (w2 * h2) - inter
    return inter / union if union>0 else 0.0

# ============= IOU Tracker ===============
class IOUTracker:
    def __init__(self, iou_threshold = 0.3, max_age = 10):
        self.next_id = 1
        self.tracks = {}
        self.iou_th = iou_threshold
        self.max_age = max_age

    def update(self, detections):
        """
        
        """
        det_bbs = [[d.x_min, d.y_min, d.x_max, d.y_max] for d in detections]
        det_conf = [d.confidence for d in detections]

        assigned = {}
        unmatched_dets = set(range(len(det_bbs)))
        unmatched_trackers = set(self.tracks.keys())

        if len(self.tracks) > 0 and len(det_bbs) > 0:
            t_ids = list(self.tracks.keys())
            iou_mat = np.zeros((len(t_ids), len(det_bbs)), dtype=np.int8)
            for i, tid in enumerate(t_ids):
                for j, db in enumerate(det_bbs):
                    iou_mat[i,j] = iou(self.tracks[tid]['bbox'], db)
        
            if HAS_SCIPY:
                row_ind, col_ind = linear_sum_assignment(-iou_mat)
                for r,c in zip(row_ind, col_ind):
                    if iou_mat[r,c] >= self.iou_th:
                        tid = t_ids[r]
                        assigned[tid] = c
                        unmatched_dets.discard(c)
                        unmatched_trackers.discard(tid)

            else:
                pairs = []
                for i in range(iou_mat.shape[0]):
                    for j in range(iou_mat.shape[1]):
                        pairs.append((iou_mat[i, j], i, j))
                pairs.sort(reverse=True)
                used_r = set(); used_c = set()
                for val, r, c in pairs:
                    if val < self.iou_th: break
                    if r in used_r or c in used_c: continue
                    used_r.add(r); used_c.add(c)
                    tid = t_ids[r]
                    assigned[tid] = c
                    unmatched_dets.discard(c)
                    unmatched_trackers.discard(tid)
        for tid, det_idx in assigned.items():
            self.tracks[tid]['bbox'] = det_bbs[det_idx]
            self.tracks[tid]['conf'] = det_conf[det_idx]
            self.tracks[tid]['age'] += 1
            self.tracks[tid]['miss'] = 0

        for d in list(unmatched_dets):
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {
                'bbox':det_bbs[d],
                'age': 1,
                'miss':0,
                'conf':det_conf[d]
            }
        
        for tid in list(unmatched_trackers):
            self.tracks[tid]['miss'] += 1
        
        to_del = [tid for tid, t in self.tracks.items() if t['miss'] > self.max_age]
        for tid in to_del:
            del self.tracks[tid]

        out = []
        for tid, t in self.tracks.items():
            out.append({'track_id': tid, 'bbox': t['bbox'], 'conf':t.get('conf', 1.0)})

        return out
    
# =============  Global ID Manager (Re-ID) ===============
class GlobalIDManager:
    def __init__(self, feature_model=None, sim_thresh=0.75, pos_thresh=120, max_invisible=150):
        self.global_targets = {}
        self.get_gid = 1
        self.model = feature_model
        self.sim_thresh = sim_thresh
        self.pos_thresh = pos_thresh
        self.max_invisible = max_invisible

        self.transform = image_preprocess

    def extract_feat(self, crop):
        if self.model is None:
            return None
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        x = self.transform(crop).unsqueeze(0)
        with torch.no_grad():
            feat = self.model(x).squeeze()
        # normalize
        feat = feat / (feat.norm() + 1e-6)
        return feat
    
    def match_or_register(self, tracker_id, bbox, frame, frame_idx):
        """
        bbox: [x,y,x,y]
        """
        x1,y1, x2,y2 = bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        crop = None
        feat = None
        if self.model is not None:
            try:
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                feat = self.extract_feat(crop)
            except Exception:
                feat = None

        best_gid = None
        best_score = -1.0
        for gid, data in self.global_targets.items():
            # position distance
            px, py = data['pos']
            dist = np.sqrt((px - cx)**2 + (py - cy)**2)
            sim = 0.0
            if feat is not None and data.get('feat') is not None:
                sim = float(F.cosine_similarity(feat, data['feat'], dim=0).item())
            # Require both conditions-ish: prefer high sim and close dist
            if (sim >= self.sim_thresh and dist <= self.pos_thresh) or (dist <= self.pos_thresh*0.5 and sim >= 0.5):
                # compute combined score
                score = sim * 0.7 + max(0, (self.pos_thresh - dist) / self.pos_thresh) * 0.3
                if score > best_score:
                    best_score = score
                    best_gid = gid

        if best_gid is not None:
            # update
            if feat is not None and self.global_targets[best_gid].get('feat') is not None:
                # EMA update
                self.global_targets[best_gid]['feat'] = 0.7 * self.global_targets[best_gid]['feat'] + 0.3 * feat
                # renormalize
                self.global_targets[best_gid]['feat'] = self.global_targets[best_gid]['feat'] / (self.global_targets[best_gid]['feat'].norm() + 1e-6)
            elif feat is not None:
                self.global_targets[best_gid]['feat'] = feat
            self.global_targets[best_gid]['pos'] = (cx, cy)
            self.global_targets[best_gid]['last_seen'] = frame_idx
            self.global_targets[best_gid]['cur_tracker'] = tracker_id
            return best_gid

        # register new gid
        gid = self.next_gid
        self.next_gid += 1
        self.global_targets[gid] = {
            'feat': feat,
            'pos': (cx, cy),
            'last_seen': frame_idx,
            'cur_tracker': tracker_id
        }
        return gid


    def cleanup(self, frame_idx):
        to_del = [gid for gid, d in self.global_targets.items() if frame_idx - d['last_seen'] > self.max_invisible]
        for gid in to_del:
            del self.global_targets[gid]
        
# =============  BirdFilter  ===============
class BirdFilter:
    def __init__(
            self,
            image_shape, 
            use_similarity=False, 
            similarity_model=None,
            drone_template=None,
            vote_len=10, 
            vote_thresh=0.6
        ):
        self.image_shape = image_shape
        self.use_similarity = use_similarity
        self.similarity_model = similarity_model
        self.template_feat = None
        self.track_history = defaultdict(lambda: deque(maxlen=20))  # gid -> list of centers
        self.bbox_history = defaultdict(lambda: deque(maxlen=20))   # gid -> list of (w,h)
        self.bird_votes = defaultdict(lambda: deque(maxlen=vote_len))
        self.vote_len = vote_len
        self.vote_thresh = vote_thresh

        self.transform = image_preprocess

        
        if use_similarity and similarity_model is not None and drone_template is not None:
            self.similarity_model = similarity_model
            self.template_feat = self._extract_feat_cv(drone_template)

    def _extract_feat_cv(self, img):
        x = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            feat = self.similarity_model(x).squeeze()
        feat = feat / (feat.norm() + 1e-6)
        return feat
    
    def _cos_sim(self, a,b):
        if a is None or b is None:
            return 0.0
        return float(F.cosine_similarity(a,b, dim=0).item())
    
    def update_and_check(self, gid, bbox, conf, frame):
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        img_h, img_w = self.image_shape

        self.track_history[gid].append((cx, cy))
        self.bbox_history[gid].append((w, h))

        is_bird_now = False

        # 规则1: 置信度
        if conf < 0.25:
            is_bird_now = True

        # 规则2: 面积/比例
        area = w * h
        if area < 30 or area > 20000:
            is_bird_now = True
        aspect = w / (h + 1e-6)
        if aspect < 0.15 or aspect > 6:
            is_bird_now = True

        # 规则3: 运动速度（短时）
        traj = list(self.track_history[gid])
        if len(traj) >= 4:
            dx = traj[-1][0] - traj[0][0]
            dy = traj[-1][1] - traj[0][1]
            speed = np.sqrt(dx*dx + dy*dy) / max(1, len(traj))
            if speed > 30:  # 像素/帧，阈值可调
                is_bird_now = True
            # 航向抖动：计算方向标准差
            dirs = []
            for i in range(1, len(traj)):
                vx = traj[i][0] - traj[i-1][0]
                vy = traj[i][1] - traj[i-1][1]
                ang = np.arctan2(vy, vx)
                dirs.append(ang)
            if len(dirs) >= 3:
                if np.std(dirs) > 0.8:
                    is_bird_now = True

        # 规则4: bbox 波动率（挥翅导致）
        sizes = list(self.bbox_history[gid])
        if len(sizes) >= 6:
            ws = np.array([s[0] for s in sizes])
            hs = np.array([s[1] for s in sizes])
            var_w = np.std(ws) / (np.mean(ws) + 1e-6)
            var_h = np.std(hs) / (np.mean(hs) + 1e-6)
            if var_w > 0.28 or var_h > 0.28:
                is_bird_now = True

        # 规则5: 图像相似度（template） 
        if self.use_similarity and self.template_feat is not None:
            try:
                crop = frame[int(y):int(y+h), int(x):int(x+w)]
                if crop.shape[0] > 8 and crop.shape[1] > 8:
                    x_in = self.transform(crop).unsqueeze(0)
                    with torch.no_grad():
                        feat = self.similarity_model(x_in).squeeze()
                    feat = feat / (feat.norm() + 1e-6)
                    sim = self._cos_sim(feat, self.template_feat)
                    if sim < 0.55:
                        is_bird_now = True
            except Exception:
                pass

        self.bird_votes[gid].append(1 if is_bird_now else 0)
        ratio = sum(self.bird_votes[gid]) / (len(self.bird_votes[gid]) + 1e-9)

        if len(self.bird_votes[gid]) >= max(3, self.vote_len//2) and ratio >= self.vote_thresh:
            return True
        return False
    
    