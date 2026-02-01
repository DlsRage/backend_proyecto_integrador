import numpy as np
from skimage.feature import hog
from skimage.color import rgb2gray

from app.services.feature_extractors.base import FeatureExtractor

class HOGExtractor(FeatureExtractor):
    def extract(self, image_bgr) -> np.ndarray:
        # image_bgr is OpenCV BGR; convert to RGB then gray
        img_rgb = image_bgr[:, :, ::-1]
        gray = rgb2gray(img_rgb)
        feat = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        return feat.astype(np.float32)
