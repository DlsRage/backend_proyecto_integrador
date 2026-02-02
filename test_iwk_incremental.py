"""
Script de prueba para verificar el algoritmo IWK en modo incremental.
Simula el procesamiento paso a paso de imágenes.
"""
import numpy as np
from app.services.clustering.InverseKmeans import OnlineInverseWeightedKMeans
from app.services.metrics import calculate_metrics, project_centroids_to_2d

def test_iwk_incremental():
    print("=" * 60)
    print("TEST: IWK Incremental con datos sintéticos")
    print("=" * 60)

    # Generar datos de prueba (3 clusters bien separados)
    np.random.seed(42)

    # Cluster 1: alrededor de (0, 0)
    cluster1 = np.random.randn(30, 50) * 0.5

    # Cluster 2: alrededor de (5, 5)
    cluster2 = np.random.randn(30, 50) * 0.5 + 5

    # Cluster 3: alrededor de (-5, 5)
    cluster3 = np.random.randn(30, 50) * 0.5 + np.array([-5, 5] + [0]*48)

    # Combinar todos
    X = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.array([0]*30 + [1]*30 + [2]*30)

    print(f"\n📊 Dataset de prueba:")
    print(f"   - {len(X)} muestras")
    print(f"   - {X.shape[1]} dimensiones")
    print(f"   - 3 clusters verdaderos")

    # Inicializar IWK
    clusterer = OnlineInverseWeightedKMeans(
        n_clusters=3,
        learning_rate=0.05,
        p=2,
        random_state=42
    )

    # Lista para guardar resultados de cada paso
    results = []

    def callback(step, centroids, labels):
        # Proyectar a 2D
        centroids_2d = project_centroids_to_2d(centroids, X, method='pca')

        # Calcular métricas (solo sobre muestras procesadas)
        processed_X = X[:step]
        metrics = calculate_metrics(processed_X, labels, centroids)

        results.append({
            'step': step,
            'centroids_2d': centroids_2d,
            'metrics': metrics
        })

        print(f"   Paso {step:3d}/{len(X)}: Dunn={metrics['dunn']:.4f}, Silhouette={metrics['silhouette']:.4f}")

    print(f"\n🔄 Ejecutando clustering incremental...")
    labels = clusterer.fit_predict(X, epoch_callback=callback)

    print(f"\n✅ Clustering completado!")
    print(f"   - {len(results)} actualizaciones reportadas")
    print(f"   - Centroides finales en 2D:")

    if results:
        final_centroids = results[-1]['centroids_2d']
        for i, c in enumerate(final_centroids):
            print(f"      Cluster {i}: ({c[0]:.4f}, {c[1]:.4f})")

        final_metrics = results[-1]['metrics']
        print(f"\n📊 Métricas finales:")
        print(f"   - Dunn Index: {final_metrics['dunn']:.4f}")
        print(f"   - Silhouette: {final_metrics['silhouette']:.4f}")

    # Verificar progresión de métricas
    print(f"\n📈 Progresión de Silhouette:")
    for i in [0, len(results)//4, len(results)//2, 3*len(results)//4, -1]:
        if i < len(results):
            r = results[i]
            print(f"   Paso {r['step']:3d}: {r['metrics']['silhouette']:.4f}")

    print("\n" + "=" * 60)
    print("TEST COMPLETADO")
    print("=" * 60)

if __name__ == "__main__":
    test_iwk_incremental()
