import numpy as np
from skimage.feature import hog

from app.services.feature_extractors.base import FeatureExtractor


class HOGExtractor(FeatureExtractor):
    def extract(self, views: dict) -> np.ndarray:
        """Extrae características HOG de la vista preprocesada."""
        # Usa la vista 'gray' ya preprocesada y normalizada
        gray = views["gray"]
        feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return feat.astype(np.float32)
