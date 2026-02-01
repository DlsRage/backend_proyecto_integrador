from fastapi import APIRouter, Request
from pydantic import BaseModel
import uuid

router = APIRouter()

class PresignIn(BaseModel):
    filename: str
    content_type: str = "image/jpeg"

@router.post("/jobs/{job_id}/presign")
async def presign_upload(job_id: str, payload: PresignIn, request: Request):
    jm = request.app.state.job_manager
    jm.get_job(job_id)  # valida exista

    # key único por job
    key = f"jobs/{job_id}/{uuid.uuid4()}-{payload.filename}"
    url = jm.storage.presign_put(key=key, content_type=payload.content_type)
    return {"key": key, "url": url}
