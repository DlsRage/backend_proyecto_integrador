from app.api.v1 import events, incremental, jobs, uploads
from fastapi import APIRouter

router = APIRouter()
router.include_router(jobs.router, tags=["jobs"])
router.include_router(uploads.router, tags=["uploads"])
router.include_router(events.router, tags=["events"])
router.include_router(incremental.router, tags=["incremental"])
