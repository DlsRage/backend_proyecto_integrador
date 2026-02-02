import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from app.core.config import settings
from app.services.clustering.InverseKmeans import OnlineInverseWeightedKMeans
from app.services.event_bus import JobEventBus
from app.services.feature_extractors.hog import HOGExtractor
from app.services.feature_extractors.moments import MomentsExtractor
from app.services.feature_extractors.sift import SIFTExtractor
from app.services.metrics import calculate_metrics, project_centroids_to_2d
from app.services.pipeline import decode_image_to_bgr, extract_features, preprocess
from app.services.storage import StorageService


@dataclass
class Job:
    id: str
    created_at: float
    image_keys: List[str] = field(default_factory=list)
    status: str = "created"  # created | running | done | failed | cancelled
    result: Optional[Dict[str, Any]] = None
    auto_delete: bool = True
    extractor: str = "hog"  # hog | sift | moments (embedding lo agregas)
    n_clusters: int = 3
    learning_rate: float = 0.01
    p: int = 2
    random_state: Optional[int] = None


class JobManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, Job] = {}
        self.bus = JobEventBus()
        self.storage = StorageService()

    def create_job(
        self,
        extractor: str,
        n_clusters: int,
        learning_rate: float,
        p: int,
        random_state: Optional[int],
        auto_delete: bool,
    ) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            created_at=asyncio.get_event_loop().time(),
            extractor=extractor,
            n_clusters=n_clusters,
            learning_rate=learning_rate,
            p=p,
            random_state=random_state,
            auto_delete=auto_delete,
        )
        self.jobs[job_id] = job
        self.bus.ensure(job_id)
        return job

    def get_job(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise KeyError("job_not_found")
        return self.jobs[job_id]

    def register_images(self, job_id: str, keys: List[str]) -> None:
        job = self.get_job(job_id)
        job.image_keys.extend(keys)
        # orden estable => resultados más reproducibles
        job.image_keys = sorted(set(job.image_keys))

    async def start(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job.status not in ("created",):
            return
        job.status = "running"
        await self.bus.publish(job_id, {"type": "status", "status": "running"})

        # ✅ Clustering en background, para NO bloquear SSE
        asyncio.create_task(self._run_job(job_id))

    def _make_extractor(self, name: str):
        if name == "hog":
            return HOGExtractor()
        if name == "sift":
            return SIFTExtractor(max_kp=128)
        if name == "moments":
            return MomentsExtractor()
        raise ValueError("unknown_extractor")

    async def _process_image_batch(self, keys, extractor, semaphore):
        """
        Procesa un lote de imágenes en paralelo limitado por semáforo.
        """
        tasks = []
        for idx, key in enumerate(keys):
            tasks.append(self._process_single_image(key, extractor, semaphore))
        return await asyncio.gather(*tasks)

    async def _process_single_image(self, key, extractor, semaphore):
        async with semaphore:
            try:
                # 1) descargar bytes
                # Timeout de 30s para descarga
                img_bytes = await asyncio.wait_for(
                    asyncio.to_thread(self.storage.get_object_bytes, key), timeout=30.0
                )

                # 2) decode + preprocess + features (CPU) en thread
                # Timeout de 30s para procesamiento
                feat = await asyncio.wait_for(
                    asyncio.to_thread(self._extract_one, img_bytes, extractor),
                    timeout=30.0,
                )
                return feat
            except asyncio.TimeoutError:
                print(f"❌ Timeout processing image {key}")
                return None
            except Exception as e:
                print(f"❌ Error processing image {key}: {e}")
                return None

    async def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        try:
            extractor = self._make_extractor(job.extractor)

            # Usar random_state del job si está definido, sino usar el de settings
            random_seed = (
                job.random_state
                if job.random_state is not None
                else settings.RANDOM_SEED
            )

            clusterer = OnlineInverseWeightedKMeans(
                n_clusters=job.n_clusters,
                learning_rate=job.learning_rate,
                p=job.p,
                random_state=random_seed,
            )

            keys = job.image_keys
            total = len(keys)
            if total == 0:
                raise RuntimeError("no_images_registered")

            await self.bus.publish(
                job_id, {"type": "phase", "name": "feature_extraction"}
            )

            # Recolectar todas las características en paralelo
            # Limitamos la concurrencia a 10 para no saturar CPU/Red
            semaphore = asyncio.Semaphore(10)

            # Procesar en chunks para poder reportar progreso
            chunk_size = 10
            all_feats = []

            for i in range(0, total, chunk_size):
                chunk_keys = keys[i : i + chunk_size]
                chunk_results = await self._process_image_batch(
                    chunk_keys, extractor, semaphore
                )

                # Filtrar errores (None)
                valid_results = [r for r in chunk_results if r is not None]
                all_feats.extend(valid_results)

                # Reportar progreso
                done_count = min(i + chunk_size, total)
                await self.bus.publish(
                    job_id,
                    {
                        "type": "progress",
                        "done": done_count,
                        "total": total,
                        "pct": round(100 * done_count / total, 2),
                        "phase": "feature_extraction",
                    },
                )

            if not all_feats:
                raise RuntimeError("No features extracted successfully")

            # Ejecutar clustering sobre todas las características
            await self.bus.publish(job_id, {"type": "phase", "name": "clustering"})
            X = np.vstack(all_feats)
            # Limpiar X de NaNs/Infs para evitar errores en clustering
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Lista para almacenar resultados de cada iteración
            iteration_results = []

            def iteration_callback(iter_num, centroids, labels):
                # Limpiar centroides de entrada de NaN/Inf
                centroids = np.nan_to_num(centroids, nan=0.0, posinf=1e10, neginf=-1e10)

                # Verificar que centroides son válidos
                if np.all(centroids == 0):
                    print(
                        f"⚠️ Centroides todos en 0 en iteración {iter_num}, omitiendo..."
                    )
                    return

                # Calcular métricas sobre todas las muestras
                try:
                    metrics = calculate_metrics(X, labels, centroids)
                except Exception as e:
                    print(f"⚠️ Error calculando métricas en iteración {iter_num}: {e}")
                    metrics = {"dunn": 0.0, "silhouette": 0.0}

                print(
                    f"✅ Iteración {iter_num}: Dunn={metrics['dunn']:.4f}, Silhouette={metrics['silhouette']:.4f}"
                )

                # Proyectar centroides a 2D para visualización
                try:
                    visual_centroids = project_centroids_to_2d(
                        centroids, X, method="pca"
                    )
                except Exception as e:
                    print(
                        f"⚠️ Error proyectando centroides en iteración {iter_num}: {e}"
                    )
                    # Fallback: usar primeras 2 dimensiones
                    if centroids.shape[1] >= 2:
                        visual_centroids = centroids[:, :2]
                    else:
                        # Crear puntos espaciados en círculo
                        k = centroids.shape[0]
                        angle = np.linspace(0, 2 * np.pi, k, endpoint=False)
                        visual_centroids = np.column_stack(
                            [np.cos(angle), np.sin(angle)]
                        )

                # Sanitize outputs (replace NaNs with 0.0)
                visual_centroids = np.nan_to_num(
                    visual_centroids, nan=0.0, posinf=10.0, neginf=-10.0
                )
                raw_centroids = np.nan_to_num(
                    centroids, nan=0.0, posinf=1e10, neginf=-1e10
                )

                # Sanitize metrics
                clean_metrics = {
                    k: (0.0 if np.isnan(v) or np.isinf(v) else float(v))
                    for k, v in metrics.items()
                }

                iteration_results.append(
                    {
                        "iteration": iter_num,  # Número de iteración del algoritmo
                        "centroids": visual_centroids.tolist(),  # Proyección 2D para visualización
                        "raw_centroids": raw_centroids.tolist(),  # Centroides originales en alta dimensión
                        "n_clusters": job.n_clusters,
                        "metrics": clean_metrics,
                    }
                )

            labels = clusterer.fit_predict(X)

            # Publicar resultados de cada iteración
            for iter_result in iteration_results:
                await self.bus.publish(job_id, {"type": "epoch_result", **iter_result})

            # Guardar resultados
            all_labels: Dict[str, int] = {}
            for k, lab in zip(keys, labels):
                all_labels[k] = int(lab)

            job.result = {
                "n_clusters": job.n_clusters,
                "extractor": job.extractor,
                "learning_rate": job.learning_rate,
                "p": job.p,
                "random_state": random_seed,
                "labels": all_labels,  # key -> cluster
            }
            job.status = "done"
            await self.bus.publish(job_id, {"type": "status", "status": "done"})

            if job.auto_delete:
                # cleanup storage (demo)
                for k in keys:
                    await asyncio.to_thread(self.storage.delete_object, k)

        except Exception as e:
            import traceback

            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"❌ Error en job {job_id}: {error_msg}")
            print(traceback.format_exc())
            job.status = "failed"
            await self.bus.publish(
                job_id, {"type": "status", "status": "failed", "error": error_msg}
            )

    def _extract_one(self, img_bytes: bytes, extractor) -> np.ndarray:
        img_bgr = decode_image_to_bgr(img_bytes)
        views = preprocess(img_bgr)  # Ahora devuelve dict de vistas
        feat = extract_features(views, extractor)
        # normaliza para estabilidad numérica
        feat = feat.astype(np.float32)
        denom = np.linalg.norm(feat) + 1e-8
        return (feat / denom).reshape(1, -1)
