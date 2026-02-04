import numpy as np
import cv2
from app.services.feature_extractors.base import FeatureExtractor


class SIFTExtractor(FeatureExtractor):
    def __init__(self, max_kp: int = 128):
        self.sift = cv2.SIFT_create()
        self.max_kp = max_kp

    def extract(self, views: dict) -> np.ndarray:
        """
        Extrae características SIFT de la vista preprocesada.
        Devuelve vector de tamaño fijo: max_kp * 128 dimensiones.
        """
        gray = views["gray"]
        kps, desc = self.sift.detectAndCompute(gray, None)
        
        # Si no hay keypoints → vector cero
        if desc is None or len(desc) == 0:
            return np.zeros((self.max_kp * 128,), dtype=np.float32)
        
        # Recortar a max_kp descriptores
        desc = desc[:self.max_kp]
        
        # Padding para vector de tamaño fijo
        if desc.shape[0] < self.max_kp:
            pad = np.zeros((self.max_kp - desc.shape[0], 128), dtype=desc.dtype)
            desc = np.vstack([desc, pad])
        
        return desc.reshape(-1).astype(np.float32)
