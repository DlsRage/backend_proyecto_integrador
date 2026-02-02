import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Tuple, Optional


def calculate_dunn_index(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Calcula el índice de Dunn para evaluar la calidad del clustering.
    Dunn = min_inter_cluster_distance / max_intra_cluster_distance
    Valores más altos indican mejor separación entre clusters.
    """
    try:
        n_clusters = len(np.unique(labels))
        if n_clusters < 2:
            return 0.0

        # Calcular distancias intra-cluster (máxima distancia dentro de cada cluster)
        max_intra_dist = 0.0
        for k in range(n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                # Distancia máxima desde el centroide a cualquier punto del cluster
                dists = np.linalg.norm(cluster_points - centroids[k], axis=1)
                max_intra_dist = max(max_intra_dist, np.max(dists) if len(dists) > 0 else 0.0)

        # Calcular distancias inter-cluster (mínima distancia entre centroides)
        min_inter_dist = float('inf')
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                min_inter_dist = min(min_inter_dist, dist)

        # Evitar división por cero
        if max_intra_dist == 0.0 or min_inter_dist == float('inf'):
            return 0.0

        dunn = min_inter_dist / max_intra_dist
        return float(dunn)

    except Exception as e:
        print(f"Error calculando Dunn index: {e}")
        return 0.0


def calculate_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Calcula el coeficiente de silueta promedio.
    Rango: [-1, 1]
    - Valores cercanos a 1: buen clustering
    - Valores cercanos a 0: clusters superpuestos
    - Valores negativos: asignaciones incorrectas
    """
    try:
        n_clusters = len(np.unique(labels))
        if n_clusters < 2 or n_clusters >= len(X):
            return 0.0

        score = silhouette_score(X, labels, metric='euclidean')
        return float(score)

    except Exception as e:
        print(f"Error calculando Silhouette: {e}")
        return 0.0


def reduce_to_2d(X: np.ndarray, method: str = 'pca') -> np.ndarray:
    """
    Reduce datos de alta dimensionalidad a 2D para visualización.

    Args:
        X: Array de forma (n_samples, n_features)
        method: 'pca' para PCA (más rápido)

    Returns:
        Array de forma (n_samples, 2)
    """
    try:
        if X.shape[1] <= 2:
            # Ya es 2D o menos
            if X.shape[1] == 2:
                return X
            elif X.shape[1] == 1:
                return np.hstack([X, np.zeros((X.shape[0], 1))])

        if method == 'pca':
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
            return X_2d
        else:
            raise ValueError(f"Método desconocido: {method}")

    except Exception as e:
        print(f"Error en reducción de dimensionalidad: {e}")
        # Fallback: retornar primeras 2 dimensiones
        if X.shape[1] >= 2:
            return X[:, :2]
        else:
            return np.hstack([X, np.zeros((X.shape[0], 1))])


def calculate_metrics(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> dict:
    """
    Calcula todas las métricas de clustering.

    Returns:
        dict con 'dunn' y 'silhouette'
    """
    dunn = calculate_dunn_index(X, labels, centroids)
    silhouette = calculate_silhouette(X, labels)

    return {
        'dunn': dunn,
        'silhouette': silhouette
    }


def project_centroids_to_2d(
    centroids: np.ndarray,
    X: np.ndarray,
    method: str = 'pca'
) -> np.ndarray:
    """
    Proyecta centroides al mismo espacio 2D que los datos.

    Args:
        centroids: Centroides en espacio original (k, n_features)
        X: Datos originales (n_samples, n_features)
        method: Método de reducción

    Returns:
        Centroides en 2D (k, 2)
    """
    try:
        # Limpiar NaN/Inf de entrada
        centroids = np.nan_to_num(centroids, nan=0.0, posinf=1e10, neginf=-1e10)
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

        if centroids.shape[1] <= 2:
            if centroids.shape[1] == 2:
                return centroids
            elif centroids.shape[1] == 1:
                return np.hstack([centroids, np.zeros((centroids.shape[0], 1))])

        if method == 'pca':
            # Ajustar PCA con los datos completos (datos + centroides)
            combined = np.vstack([X, centroids])

            # Verificar que no hay NaN después de combinar
            if np.any(np.isnan(combined)) or np.any(np.isinf(combined)):
                print("⚠️ Combined data contains NaN/Inf, cleaning...")
                combined = np.nan_to_num(combined, nan=0.0, posinf=1e10, neginf=-1e10)

            pca = PCA(n_components=2, random_state=42)
            combined_2d = pca.fit_transform(combined)

            # Extraer solo los centroides proyectados
            centroids_2d = combined_2d[-len(centroids):]

            # Limpiar resultado
            centroids_2d = np.nan_to_num(centroids_2d, nan=0.0, posinf=1e10, neginf=-1e10)

            return centroids_2d
        else:
            raise ValueError(f"Método desconocido: {method}")

    except Exception as e:
        print(f"Error proyectando centroides: {e}")
        # Fallback: crear puntos espaciados en 2D
        k = centroids.shape[0]
        # Crear una cuadrícula simple
        angle = np.linspace(0, 2 * np.pi, k, endpoint=False)
        fallback_centroids = np.column_stack([np.cos(angle), np.sin(angle)])
        return fallback_centroids
