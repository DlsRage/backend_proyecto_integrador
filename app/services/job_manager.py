import asyncio
import os
import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.config import settings
from app.services.event_bus import JobEventBus
from app.services.storage import StorageService
from app.services.pipeline import decode_image_to_bgr, preprocess, extract_features
from app.services.feature_extractors.hog import HOGExtractor
from app.services.feature_extractors.sift import SIFTExtractor
from app.services.feature_extractors.moments import MomentsExtractor
from app.services.clustering.InverseKmeans import OnlineInverseWeightedKMeans

@dataclass
class Job:
    id: str
    created_at: float
    image_keys: List[str] = field(default_factory=list)
    status: str = "created"  # created | running | done | failed | cancelled
    result: Optional[Dict[str, Any]] = None
    auto_delete: bool = True
    extractor: str = "hog"   # hog | sift | moments (embedding lo agregas)
    n_clusters: int = 3

class JobManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, Job] = {}
        self.bus = JobEventBus()
        self.storage = StorageService()

    def create_job(self, extractor: str, n_clusters: int, auto_delete: bool) -> Job:
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            created_at=asyncio.get_event_loop().time(),
            extractor=extractor,
            n_clusters=n_clusters,
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

    async def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        try:
            extractor = self._make_extractor(job.extractor)
            clusterer = OnlineInverseWeightedKMeans(
                n_clusters=job.n_clusters,
                random_state=settings.RANDOM_SEED,
            )

            keys = job.image_keys
            total = len(keys)
            if total == 0:
                raise RuntimeError("no_images_registered")

            await self.bus.publish(job_id, {"type": "phase", "name": "feature_extraction"})

            # Recolectar todas las características primero
            all_feats = []

            for idx, key in enumerate(keys, start=1):
                # 1) descargar bytes
                img_bytes = await asyncio.to_thread(self.storage.get_object_bytes, key)

                # 2) decode + preprocess + features (CPU) en thread
                feat = await asyncio.to_thread(self._extract_one, img_bytes, extractor)

                all_feats.append(feat)

                # Reportar progreso de extracción
                await self.bus.publish(job_id, {
                    "type": "progress",
                    "done": idx,
                    "total": total,
                    "pct": round(100 * idx / total, 2),
                    "phase": "feature_extraction"
                })

            # Ejecutar clustering sobre todas las características
            await self.bus.publish(job_id, {"type": "phase", "name": "clustering"})
            X = np.vstack(all_feats)
            labels = clusterer.fit_predict(X)

            # Guardar resultados
            all_labels: Dict[str, int] = {}
            for k, lab in zip(keys, labels):
                all_labels[k] = int(lab)

            job.result = {
                "n_clusters": job.n_clusters,
                "extractor": job.extractor,
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
            await self.bus.publish(job_id, {"type": "status", "status": "failed", "error": error_msg})

    def _extract_one(self, img_bytes: bytes, extractor) -> np.ndarray:
        img_bgr = decode_image_to_bgr(img_bytes)
        views = preprocess(img_bgr)
        feat = extract_features(views, extractor)
        # normaliza para estabilidad numérica
        feat = feat.astype(np.float32)
        denom = (np.linalg.norm(feat) + 1e-8)
        return (feat / denom).reshape(1, -1)
