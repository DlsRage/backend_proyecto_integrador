from fastapi import APIRouter
from app.api.v1 import jobs, uploads, events

router = APIRouter()
router.include_router(jobs.router, tags=["jobs"])
router.include_router(uploads.router, tags=["uploads"])
router.include_router(events.router, tags=["events"])
