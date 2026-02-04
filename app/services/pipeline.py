import io

import cv2
import numpy as np
from app.services.feature_extractors.base import FeatureExtractor
from app.services.preprocessing.preprocessor import UniversalPreprocessPipeline
from PIL import Image

# Instancia global del preprocesador
_preprocessor = UniversalPreprocessPipeline(out_size=(256, 256))


def decode_image_to_bgr(image_bytes: bytes) -> np.ndarray:
    """Decodifica bytes de imagen a formato BGR de OpenCV."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    arr = np.array(img)
    return arr[:, :, ::-1].copy()


def preprocess(image_bgr: np.ndarray) -> dict:
    """
    Preprocesa la imagen y devuelve vistas múltiples.

    Returns:
        dict con keys: canon_bgr, gray, mask
        o None si la imagen es inválida
    """
    views = _preprocessor.preprocess(image_bgr)
    return views


def extract_features(views: dict, extractor: FeatureExtractor) -> np.ndarray:
    """
    Extrae características usando las vistas preprocesadas.

    Args:
        views: dict con vistas del preprocesador (canon_bgr, gray, mask)
        extractor: instancia de FeatureExtractor

    Returns:
        Vector de características 1D
    """
    return extractor.extract(views)
