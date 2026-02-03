import numpy as np
from app.services.clustering.base import BaseOnlineClustering

class OnlineInverseWeightedKMeans(BaseOnlineClustering):
    def __init__(self, p=2, window_size=200, **kwargs):
        super().__init__(**kwargs)

        self.p = p

        # 🔥 tracking online
        self.window_size = window_size

        self.buffer_X = []
        self.buffer_labels = []


    def _update_step(self, x):

        epsilon = 1e-8
        dists = np.array([np.linalg.norm(x - c) for c in self.centroids])

        inv_dists = 1.0 / (dists**self.p + epsilon)
        weights = inv_dists / inv_dists.sum()

        for i in range(self.k):
            self.centroids[i] += self.lr * weights[i] * (x - self.centroids[i])

        # =========================
        # 🔥 GUARDAR STREAMING
        # =========================
        label = np.argmin(dists)

        self.buffer_X.append(x)
        self.buffer_labels.append(label)

        if len(self.buffer_X) > self.window_size:
            self.buffer_X.pop(0)
            self.buffer_labels.pop(0)
