from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()


class JobCreateIn(BaseModel):
    extractor: str = "hog"
    n_clusters: int = 3
    learning_rate: float = 0.01
    p: int = 2
    random_state: int | None = None
    cluster_sizes: list[int]


class RegisterIn(BaseModel):
    keys: List[str]


@router.post("/jobs")
async def create_job(payload: JobCreateIn, request: Request):
    jm = request.app.state.job_manager
    job = jm.create_job(
        extractor=payload.extractor,
        n_clusters=payload.n_clusters,
        learning_rate=payload.learning_rate,
        p=payload.p,
        random_state=payload.random_state,
        cluster_sizes=payload.cluster_sizes,
    )
    return {"job_id": job.id, "status": job.status}


@router.post("/jobs/{job_id}/register")
async def register_images(job_id: str, payload: RegisterIn, request: Request):
    jm = request.app.state.job_manager
    try:
        jm.register_images(job_id, payload.keys)
        return {"ok": True, "count": len(payload.keys)}
    except KeyError:
        raise HTTPException(status_code=404, detail="job_not_found")


@router.post("/jobs/{job_id}/start")
async def start_job(job_id: str, request: Request):
    jm = request.app.state.job_manager
    try:
        await jm.start(job_id)
        return {"ok": True}
    except KeyError:
        raise HTTPException(status_code=404, detail="job_not_found")


@router.get("/jobs/{job_id}/result")
async def get_result(job_id: str, request: Request):
    jm = request.app.state.job_manager
    try:
        job = jm.get_job(job_id)
        return {"status": job.status, "result": job.result}
    except KeyError:
        raise HTTPException(status_code=404, detail="job_not_found")
