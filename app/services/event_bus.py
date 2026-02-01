import asyncio
from typing import Any, Dict

class JobEventBus:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[Dict[str, Any]]] = {}

    def ensure(self, job_id: str) -> None:
        if job_id not in self._queues:
            self._queues[job_id] = asyncio.Queue()

    async def publish(self, job_id: str, event: Dict[str, Any]) -> None:
        self.ensure(job_id)
        await self._queues[job_id].put(event)

    async def subscribe(self, job_id: str):
        self.ensure(job_id)
        queue = self._queues[job_id]
        while True:
            ev = await queue.get()
            yield ev
