import numpy as np
import cv2
from app.services.feature_extractors.base import FeatureExtractor

class SIFTExtractor(FeatureExtractor):
    def __init__(self, max_kp: int = 128):
        self.sift = cv2.SIFT_create()
        self.max_kp = max_kp

    def extract(self, image_bgr) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        kps, desc = self.sift.detectAndCompute(gray, None)
        if desc is None:
            return np.zeros((self.max_kp * 128,), dtype=np.float32)

<<<<<<< Updated upstream
        desc = desc[: self.max_kp]  # recorta
        # padding para vector de tamaño fijo
        if desc.shape[0] < self.max_kp:
            pad = np.zeros((self.max_kp - desc.shape[0], 128), dtype=desc.dtype)
            desc = np.vstack([desc, pad])
=======
        # si no hay keypoints → vector cero (128D = dimensión SIFT estándar)
        if desc is None or len(desc) == 0:
            return np.zeros(128, dtype=np.float32)
>>>>>>> Stashed changes

        return desc.reshape(-1).astype(np.float32)
