import numpy as np
import cv2
from app.services.feature_extractors.base import FeatureExtractor

class MomentsExtractor(FeatureExtractor):
    def extract(self, views: dict) -> np.ndarray:
        """Extrae momentos de la vista de bordes preprocesada."""
        # Usa la vista 'edges' que ya aplicó Canny y Morfología
        edges = views["edges"]
        m = cv2.moments(edges)
        # vector fijo (ordenado) para estabilidad
        keys = ["m00","m10","m01","m20","m11","m02","m30","m21","m12","m03",
                "mu20","mu11","mu02","mu30","mu21","mu12","mu03",
                "nu20","nu11","nu02","nu30","nu21","nu12","nu03"]
        v = np.array([m[k] for k in keys], dtype=np.float32)
        return v
