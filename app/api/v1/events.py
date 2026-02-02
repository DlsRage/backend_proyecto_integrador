from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
import json

router = APIRouter()

@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    jm = request.app.state.job_manager

    async def event_generator():
        async for ev in jm.bus.subscribe(job_id):
            if await request.is_disconnected():
                break
            
            event_type = ev.get("type", "message")
            # Crear una copia sin el campo 'type' para enviarlo en data
            event_data = {k: v for k, v in ev.items() if k != "type"}
            
            # sse_starlette automáticamente serializa a JSON
            yield {
                "event": event_type,
                "data": json.dumps(event_data)
            }

    return EventSourceResponse(event_generator())
