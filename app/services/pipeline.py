import cv2
import numpy as np
from PIL import Image
import io

from app.services.feature_extractors.base import FeatureExtractor
from app.services.preprocessing.preprocessor import UniversalPreprocessPipeline

# Instancia global del preprocesador
_preprocessor = UniversalPreprocessPipeline(out_size=(256, 256))

def decode_image_to_bgr(image_bytes: bytes) -> np.ndarray:
    # PIL -> RGB -> OpenCV BGR
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    return arr[:, :, ::-1].copy()

def preprocess(image_bgr: np.ndarray) -> dict:
    """
    Preprocesa la imagen y devuelve vistas múltiples.
    Retorna dict con: canon_bgr, gray, edges
    """
    views = _preprocessor.preprocess(image_bgr)
    return views

def extract_features(views: dict, extractor: FeatureExtractor) -> np.ndarray:
    """Extrae características usando las vistas preprocesadas."""
    return extractor.extract(views)
