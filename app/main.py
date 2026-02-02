from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.v1.router import router as v1_router
from app.services.job_manager import JobManager
from app.core.logging import setup_logging


def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI(title=settings.APP_NAME)
    
    # Configurar CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001"],  # URLs del frontend Next.js
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(v1_router, prefix=settings.API_PREFIX)

    @app.on_event("startup")
    async def startup():
        app.state.job_manager = JobManager()

    @app.get("/health")
    async def health():
        return {"ok": True}

    return app

app = create_app()
