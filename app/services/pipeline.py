import cv2
import numpy as np
from PIL import Image
import io

from app.services.feature_extractors.base import FeatureExtractor

def decode_image_to_bgr(image_bytes: bytes) -> np.ndarray:
    # PIL -> RGB -> OpenCV BGR
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    return arr[:, :, ::-1].copy()

def preprocess(image_bgr: np.ndarray) -> np.ndarray:
    # Aquí metes tu “limpieza de ruido” real.
    # Demo: resize para control de RAM/tiempo.
    return cv2.resize(image_bgr, (256, 256), interpolation=cv2.INTER_AREA)

def extract_features(image_bgr: np.ndarray, extractor: FeatureExtractor) -> np.ndarray:
    return extractor.extract(image_bgr)
