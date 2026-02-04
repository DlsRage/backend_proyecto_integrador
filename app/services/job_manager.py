import asyncio
import os
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from app.core.config import settings
from app.services.clustering.metricas import ClusteringMetrics
from app.services.clustering.OnlineInverse import OnlineInverseWeightedSize
from app.services.event_bus import JobEventBus
from app.services.feature_extractors.embedding import EmbeddingExtractor
from app.services.feature_extractors.hog import HOGExtractor
from app.services.feature_extractors.moments import MomentsExtractor
from app.services.feature_extractors.sift import SIFTExtractor
from app.services.pipeline import decode_image_to_bgr, extract_features, preprocess
from app.services.storage import StorageService


@dataclass
class Job:
    id: str
    created_at: float
    cluster_sizes: List[int]
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
        cluster_sizes: List[int],
        p: int,
        random_state: Optional[int],
    ) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            created_at=asyncio.get_event_loop().time(),
            extractor=extractor,
            n_clusters=n_clusters,
            cluster_sizes=cluster_sizes,
            learning_rate=learning_rate,
            p=p,
            random_state=random_state,
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

        # randomizar
        random.shuffle(job.image_keys)

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
        if name == "embeddings" or name == "cnn":
            return EmbeddingExtractor()
        raise ValueError("unknown_extractor")

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

            clusterer = OnlineInverseWeightedSize(
                k=job.n_clusters,
                learning_rate=job.learning_rate,
                p=job.p,
                seed=random_seed,
                max_size=job.cluster_sizes,
            )

            keys = job.image_keys
            total = len(keys)
            if total == 0:
                raise RuntimeError("no_images_registered")

            # Recolectar todas las características en paralelo
            # Limitamos la concurrencia a 10 para no saturar CPU/Red
            chunk_size = total // 10  # evitar 0
            semaphore = asyncio.Semaphore(1)
            buffer_key = []
            all_feats = []
            all_labels = []
            buffer = []

            for i, key in enumerate(keys):
                feat = await self._process_single_image(key, extractor, semaphore)
                if feat is None:
                    continue

                feat = np.nan_to_num(feat)
                all_feats.append(feat)

                # Extraer solo el nombre del archivo con extensión
                filename = os.path.basename(key)

                if clusterer.centroids is None:
                    clusterer.partial_fit(feat)
                    buffer.append(feat)
                    buffer_key.append(filename)

                    # 2. Justo cuando se llena el buffer (k+1), lo vaciamos y clasificamos
                    if len(buffer) == job.n_clusters:
                        for key_buf, x_buf in enumerate(buffer, start=0):
                            # Ahora partial_fit ya no devolverá -1 porque centroids ya existe
                            res_buf = clusterer.partial_fit(x_buf)
                            await self.bus.publish(
                                job_id,
                                {
                                    "type": "image_label",
                                    "image_key": buffer_key[
                                        key_buf
                                    ],  # Solo nombre del archivo
                                    "cluster": int(res_buf),
                                },
                            )
                            all_labels.append(res_buf)
                        buffer = []  # Vaciamos para no volver a entrar aquí
                    continue  # Pasamos al siguiente x del flujo X

                respuesta = clusterer.partial_fit(feat)
                all_labels.append(respuesta)
                await self.bus.publish(
                    job_id,
                    {
                        "type": "image_label",
                        "image_key": filename,  # Solo nombre del archivo
                        "cluster": int(respuesta),
                    },
                )

                # métricas online cada N
                if (i + 1) % chunk_size == 0:
                    X = np.vstack(all_feats)
                    labels = np.array(all_labels)

                    metrics = ClusteringMetrics.evaluate(X, labels)

                    await self.bus.publish(
                        job_id,
                        {
                            "type": "metrics",
                            "iteration": i + 1,
                            "metrics": metrics,
                            "centroids": clusterer.get_centroids_2d().tolist(),
                        },
                    )
            X = np.vstack(all_feats)
            labels = np.array(all_labels)

            final_metrics = ClusteringMetrics.evaluate(X, labels)

            await self.bus.publish(
                job_id,
                {
                    "type": "final_metrics",
                    "metrics": final_metrics,
                    "centroids": clusterer.centroids.tolist(),
                },
            )
            # Guardar el clusterer en el job para uso incremental
            job.clusterer = clusterer

            job.status = "done"
            await self.bus.publish(job_id, {"type": "status", "status": "done"})

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

        if isinstance(extractor, EmbeddingExtractor):
            views = {"canon_bgr": img_bgr}
        else:
            views = preprocess(img_bgr)

        feat = extract_features(views, extractor)

        feat = feat.astype(np.float32)
        denom = np.linalg.norm(feat) + 1e-8

        normalized = (feat / denom).ravel()

        return normalized

    def get_incremental_state(self, job_id: str) -> dict:
        """
        Obtiene el estado de los centroides de un job para usar en clustering incremental.
        """
        job = self.get_job(job_id)

        if job.status != "done":
            raise ValueError("job_not_completed")

        if not hasattr(job, "clusterer") or job.clusterer is None:
            raise ValueError("no_clusterer_available")

        return {
            "centroids": job.clusterer.centroids.tolist(),
            "cluster_sizes": job.clusterer.sizes.tolist(),
            "n_clusters": job.n_clusters,
            "extractor": job.extractor,
        }

    async def process_incremental_image(
        self, image_bytes: bytes, image_key: str, base_job_id: Optional[str] = None
    ) -> dict:
        """
        Procesa una imagen nueva usando clustering incremental.

        Args:
            image_bytes: Bytes de la imagen
            image_key: Key donde se guardó la imagen en S3
            base_job_id: ID del job base para obtener centroides y configuración

        Returns:
            Dict con cluster asignado, centroides actualizados y tamaños
        """
        # Si hay un job base, cargar su estado
        if base_job_id:
            try:
                base_job = self.get_job(base_job_id)

                # Asegurarse de que el job tiene un clusterer guardado
                if not hasattr(base_job, "clusterer") or base_job.clusterer is None:
                    raise ValueError("El job base no tiene un clusterer disponible")

                clusterer = base_job.clusterer
                extractor_name = base_job.extractor

            except KeyError:
                raise ValueError(f"Job base {base_job_id} no encontrado")
        else:
            raise ValueError("Se requiere un job_id base para clustering incremental")

        # Crear el extractor apropiado
        extractor = self._make_extractor(extractor_name)

        # Extraer features de la nueva imagen
        try:
            feat = await asyncio.to_thread(self._extract_one, image_bytes, extractor)
            feat = np.nan_to_num(feat)
        except Exception as e:
            raise ValueError(f"Error extrayendo features: {str(e)}")

        # Clasificar la imagen con el clusterer existente
        try:
            assigned_cluster = clusterer.partial_fit(feat)
        except Exception as e:
            raise ValueError(f"Error en clustering: {str(e)}")

        # Retornar resultados
        return {
            "assigned_cluster": int(assigned_cluster),
            "centroids": clusterer.centroids.tolist(),
            "cluster_sizes": clusterer.sizes.tolist(),
        }
