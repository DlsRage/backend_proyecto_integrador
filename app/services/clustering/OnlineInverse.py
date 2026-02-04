import numpy as np
from sklearn.decomposition import PCA


class OnlineInverseWeightedSize:
    def __init__(self, k=3, learning_rate=0.1, max_size=(30, 30, 30), seed=0, p=2):
        self.k = k
        self.lr = learning_rate
        self.max_size = np.array(max_size)
        self.eps = 1e-8
        self.p = p
        self.rng = np.random.RandomState(seed)

        self.centroids = None
        self.sizes = None
        # Buffer para recolectar las primeras k+1 instancias
        self.buffer = []

    def _init_from_buffer(self, n_trials):
        # Convertimos el buffer a una matriz para calcular dimensiones
        data_buffer = np.array(self.buffer)

        # Obtenemos los límites reales del espacio de datos visto hasta ahora
        mins = data_buffer.min(axis=0)
        maxs = data_buffer.max(axis=0)

        # Generamos centroides aleatorios distribuidos uniformemente en ese rango
        # Esto asegura que los centroides empiecen "dentro" de la zona de datos
        n_features = data_buffer.shape[1]

        best_spread = -np.inf
        best_centroids = None

        for _ in range(n_trials):
            candidate = self.rng.uniform(low=mins, high=maxs, size=(self.k, n_features))
            # medir mínima distancia entre centroides
            pairwise = np.linalg.norm(candidate[:, None] - candidate, axis=2)
            spread = pairwise[np.triu_indices(self.k, 1)].min()
            if spread > best_spread:
                best_spread = spread
                best_centroids = candidate.copy()

        self.centroids = best_centroids
        self.sizes = np.zeros(self.k)

        # Limpiamos el buffer para liberar memoria
        self.buffer = None

    def partial_fit(self, x):
        # Si aún no inicializamos, guardamos en el buffer
        if self.centroids is None:
            self.buffer.append(x)
            # Cuando llegamos a k + 1, inicializamos
            if len(self.buffer) >= self.k:
                self._init_from_buffer(10)
                return -1
            return -1

        # Lógica principal de clustering
        dists = np.linalg.norm(self.centroids - x, axis=1) + self.eps
        weights = 1 / (dists**self.p)

        mask = self.sizes < self.max_size
        if not np.any(mask):
            mask[:] = True
        weights[~mask] = 0

        size_penalty = 1 / (1 + self.sizes)
        weights *= size_penalty
        weights /= weights.sum()

        j = np.argmax(weights)

        # Learning rate adaptativo
        eta = self.lr / (1 + np.sqrt(self.sizes[j]))
        self.centroids[j] += eta * (x - self.centroids[j])
        self.sizes[j] += 1

        return j

    def get_centroids_2d(self):
        """
        Devuelve centroides proyectados a 2D usando PCA.
        Útil para visualización cuando tienes 128 features o más.

        return shape: (k, 2)
        """

        if self.centroids is None:
            return None

        if self.centroids.shape[1] <= 2:
            return self.centroids.copy()

        pca = PCA(n_components=2)
        centroids_2d = pca.fit_transform(self.centroids)

        return centroids_2d
