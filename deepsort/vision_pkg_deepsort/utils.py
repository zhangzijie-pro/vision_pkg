import cv2
import numpy as np

def nv12_to_bgr(nv12: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert NV12 (H*1.5 x W) to BGR.
    """
    if nv12 is None:
        return None
    try:
        y_plane = nv12[0:height, :]
        uv_plane = nv12[height:height + height//2, :]
        yuv = np.vstack((y_plane, uv_plane))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        return bgr
    except Exception:
        # Fallback: sometimes buffer is provided as contiguous 1-D bytes
        nv12 = nv12.reshape((int(height*1.5), width))
        return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)

def xyxy_to_ltwh(x1, y1, x2, y2):
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

def bbox_center(x1, y1, x2, y2):
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

def clip_box(x1,y1,x2,y2,w,h):
    return [max(0,int(x1)), max(0,int(y1)), min(w-1,int(x2)), min(h-1,int(y2))]

def hsv_hist_signature(img_bgr, bbox, bins=(16,16)):
    """Return L1-normalized HSV histogram signature for lightweight ReID."""
    x1,y1,x2,y2 = [int(v) for v in bbox]
    h, w = img_bgr.shape[:2]
    x1,y1,x2,y2 = clip_box(x1,y1,x2,y2,w,h)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1],None,bins,[0,180,0,256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def hist_distance(h1, h2):
    if h1 is None or h2 is None:
        return 1.0
    # Bhattacharyya distance
    return float(cv2.compareHist(h1.astype('float32'), h2.astype('float32'), cv2.HISTCMP_BHATTACHARYYA))
