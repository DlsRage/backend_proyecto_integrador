from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

router = APIRouter()

@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    jm = request.app.state.job_manager

    async def event_generator():
        async for ev in jm.bus.subscribe(job_id):
            if await request.is_disconnected():
                break
            yield {"event": ev.get("type", "message"), "data": ev}

    return EventSourceResponse(event_generator())
