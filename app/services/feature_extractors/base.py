from abc import ABC, abstractmethod
import numpy as np

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, views: dict) -> np.ndarray:
        """Return 1D float vector.
        
        Args:
            views: Dict con vistas preprocesadas (canon_bgr, gray, edges)
        """
        raise NotImplementedError
