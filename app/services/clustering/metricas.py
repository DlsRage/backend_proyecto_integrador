import numpy as np
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.spatial.distance import cdist


class ClusteringMetrics:
    """
    Métricas clásicas para clustering (offline/batch)
    """

    # =============================
    # Dunn Index
    # =============================
    @staticmethod
    def dunn_index(X, labels):
        X = np.asarray(X)
        labels = np.asarray(labels)

        unique = np.unique(labels)

        clusters = [X[labels == k] for k in unique]

        # distancia mínima entre clusters
        min_inter = np.inf
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = cdist(clusters[i], clusters[j])
                min_inter = min(min_inter, np.min(d))

        # diámetro máximo intra cluster
        max_intra = 0
        for c in clusters:
            if len(c) > 1:
                d = cdist(c, c)
                max_intra = max(max_intra, np.max(d))

        if max_intra == 0:
            return 0

        return min_inter / max_intra


    # =============================
    # Todas las métricas juntas
    # =============================
    @staticmethod
    def evaluate(X, labels, y_true=None):
        results = {}

        if len(np.unique(labels)) > 1:
            results["silhouette"] = float(silhouette_score(X, labels))
            results["dunn"] = float(ClusteringMetrics.dunn_index(X, labels))

        if y_true is not None:
            results["nmi"] = float(normalized_mutual_info_score(y_true, labels))
            results["ari"] = float(adjusted_rand_score(y_true, labels))
            results["ami"] = float(adjusted_mutual_info_score(y_true, labels))

        return results

