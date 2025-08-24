from typing import List, Tuple

class DetectorAdapter:
    def __init__(self, conf_thres: float = 0.35, class_whitelist=None):
        self.conf_thres = conf_thres
        self.class_whitelist = class_whitelist  # Optional set of ints

    def infer(self, frame_bgr) -> List[Tuple[list, float, int]]:
        """
        Implement this using your detection model.
        Expected return: list of ([x1,y1,x2,y2], score, class_id)
        """
        return []
