import cv2
import numpy as np
from app.services.feature_extractors.base import FeatureExtractor


class SIFTExtractor(FeatureExtractor):
    def __init__(self, max_kp: int = 128):
        self.sift = cv2.SIFT_create()
        self.max_kp = max_kp

    def extract(self, views: dict) -> np.ndarray:
        """Devuelve un vector SIFT global de 128 dimensiones (mean pooling)."""

        gray = views["gray"]

        kps, desc = self.sift.detectAndCompute(gray, None)

        # si no hay keypoints → vector cero
        if desc is None or len(desc) == 0:
            return np.zeros(128, dtype=np.float32)

        # 🔥 promedio → 128D fijo
        feat = desc.mean(axis=0)

        return feat.reshape(-1).astype(np.float32)
