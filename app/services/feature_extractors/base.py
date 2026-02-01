from abc import ABC, abstractmethod
import numpy as np

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, image_bgr) -> np.ndarray:
        """Return 1D float vector."""
        raise NotImplementedError
