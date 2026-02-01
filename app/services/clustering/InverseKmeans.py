class OnlineInverseWeightedKMeans(BaseOnlineClustering):
    def __init__(self, p=2, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def _update_step(self, x):
        epsilon = 1e-8
        dists = np.array([np.linalg.norm(x - c) for c in self.centroids])

        inv_dists = 1.0 / (dists**self.p + epsilon)
        weights = inv_dists / inv_dists.sum()

        for i in range(self.k):
            self.centroids[i] += self.lr * weights[i] * (x - self.centroids[i])