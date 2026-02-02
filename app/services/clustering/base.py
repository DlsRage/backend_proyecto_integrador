import numpy as np


class BaseOnlineClustering:
    def __init__(self, n_clusters=3, learning_rate=0.01, epochs=1, random_state=None):
        self.k = n_clusters
        self.lr = learning_rate
        self.epochs = epochs
        self.rng = np.random.RandomState(random_state)
        self.centroids = None

    def _initialize(self, X):
        idx = self.rng.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[idx].copy()

    def fit_predict(self, X):
        X = np.asarray(X)
        self._initialize(X)

        for _ in range(self.epochs):
            indices = self.rng.permutation(len(X))
            for i in indices:
                self._update_step(X[i])

        return self._assign_labels(X)

    def _assign_labels(self, X):
        labels = []
        for x in X:
            dists = [np.linalg.norm(x - c) for c in self.centroids]
            labels.append(np.argmin(dists))
        return np.array(labels)

    def _update_step(self, x):
        raise NotImplementedError
