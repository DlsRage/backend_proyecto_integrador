import asyncio
import uuid
from typing import List, Optional

from app.core.config import settings
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

router = APIRouter()


class IncrementalResponse(BaseModel):
    assigned_cluster: int
    centroids: List[List[float]]
    cluster_sizes: List[int]
    image_key: str


@router.post("/incremental/upload", response_model=IncrementalResponse)
async def upload_incremental_image(
    file: UploadFile = File(...),
    job_id: Optional[str] = None,
    request: Request = None,
):
    """
    Endpoint para subir una imagen nueva y clasificarla usando clustering incremental.

    - Lee la imagen del frontend
    - La sube a DigitalOcean Spaces
    - Usa los centroides del job_id especificado (o crea nuevos)
    - Clasifica la imagen
    - Actualiza los centroides
    - Devuelve el cluster asignado y los centroides actualizados
    """
    jm = request.app.state.job_manager

    # Validar que sea una imagen
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    # Leer los bytes de la imagen
    image_bytes = await file.read()

    # Generar key único para la imagen
    image_key = f"incremental/{uuid.uuid4()}-{file.filename}"

    try:
        # Subir imagen a DigitalOcean Spaces
        await asyncio.to_thread(
            jm.storage.s3.put_object,
            Bucket=settings.SPACES_BUCKET,
            Key=image_key,
            Body=image_bytes,
            ContentType=file.content_type,
        )

        # Procesar la imagen incrementalmente
        result = await jm.process_incremental_image(
            image_bytes=image_bytes, image_key=image_key, base_job_id=job_id
        )

        return IncrementalResponse(
            assigned_cluster=result["assigned_cluster"],
            centroids=result["centroids"],
            cluster_sizes=result["cluster_sizes"],
            image_key=image_key,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error procesando imagen: {str(e)}"
        )


@router.get("/incremental/state/{job_id}")
async def get_incremental_state(job_id: str, request: Request):
    """
    Obtiene el estado actual de los centroides de un job para clustering incremental.
    """
    jm = request.app.state.job_manager

    try:
        state = jm.get_incremental_state(job_id)
        return state
    except KeyError:
        raise HTTPException(status_code=404, detail="job_not_found")
