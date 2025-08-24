import numpy as np
import logging
import cv2

try:
    from hobot_vio import libsrcampy  # RDK camera lib
except Exception as e:
    libsrcampy = None
    logging.warning("libsrcampy not available in this env: %s", e)

class RdkMipiCamera:
    """Thin wrapper over libsrcampy.Camera for NV12 frames -> BGR output."""
    def __init__(self, cam_id: int = 0, width: int = 960, height: int = 544):
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self._cam = None

    def open(self):
        if libsrcampy is None:
            raise RuntimeError("libsrcampy not available")
        self._cam = libsrcampy.Camera()
        try:
            self._cam.open(self.cam_id)
        except TypeError:
            self._cam.open(camera_id=self.cam_id)
        self._cam.start()

    def read_bgr(self):
        if self._cam is None:
            self.open()
        frame = self._cam.get_image()
        if frame is None:
            return None
        try:
            nv12 = frame.get_nv12_image()
        except AttributeError:
            nv12 = getattr(frame, "nv12", None) or getattr(frame, "buffer", None)
        if nv12 is None:
            return None
        arr = np.frombuffer(nv12, dtype=np.uint8)
        arr = arr.reshape((int(self.height*1.5), self.width))
        bgr = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_NV12)
        return bgr

    def release(self):
        if self._cam is not None:
            try:
                self._cam.stop()
            except Exception:
                pass
            try:
                self._cam.close()
            except Exception:
                pass
            self._cam = None
