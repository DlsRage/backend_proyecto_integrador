import cv2
import mahotas
import numpy as np
from app.services.feature_extractors.base import FeatureExtractor


class MomentsExtractor(FeatureExtractor):
    """Extractor de Momentos de Zernike.

    Los momentos de Zernike son ortogonales y robustos al ruido,
    utilizados en análisis de forma y reconocimiento de patrones.
    """

    def extract(self, views: dict) -> np.ndarray:
        """Extrae momentos de Zernike de la vista de bordes preprocesada."""
        # Usa la vista 'edges' que ya aplicó Canny y Morfología
        edges = views["edges"]

        # Asegurar que sea una imagen en escala de grises
        if len(edges.shape) == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

        # Convertir a uint8 si no lo está
        if edges.dtype != np.uint8:
            edges = edges.astype(np.uint8)

        # Calcular momentos de Zernike
        try:
            radius = min(edges.shape) // 2
            zernike_moments = mahotas.features.zernike_moments(edges, radius=radius)
            return zernike_moments.astype(np.float32)
        except Exception as e:
            print(f"Error calculando Zernike: {e}")
            # Retornar vector de ceros en caso de error
            return np.zeros(25, dtype=np.float32)
