"""
FastAPI application entry point / factory.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import (
    admin,
    agent,
    analysis,
    analytics,
    auth,
    chat,
    documents,
    search,
    sessions,
)
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.database.base import init_db

settings = get_settings()
configure_logging(debug=settings.DEBUG)
logger = get_logger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "Production-oriented backend for uploading, searching, and reasoning over "
            "research documents using RAG, hybrid search, and TensorFlow classification. "
            "Protected endpoints require a JWT Bearer token obtained from "
            "/api/v1/auth/login or /api/v1/auth/register. Use the 'Authorize' "
            "button in Swagger to set it."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    @app.on_event("startup")
    def on_startup() -> None:
        settings.ensure_directories()
        init_db()
        logger.info(
            "%s started in '%s' mode.",
            settings.APP_NAME,
            settings.APP_ENV,
        )

    @app.get("/", tags=["Root"])
    def root() -> dict[str, str]:
        return {
            "name": settings.APP_NAME,
            "status": "running",
            "version": "2.0.0",
            "documentation": "/docs",
            "health": "/health",
        }

    @app.get("/health", tags=["Health"])
    def health_check() -> dict[str, str]:
        return {
            "status": "ok",
            "app_name": settings.APP_NAME,
            "environment": settings.APP_ENV,
        }

    app.include_router(auth.router, prefix=settings.API_PREFIX)
    app.include_router(documents.router, prefix=settings.API_PREFIX)
    app.include_router(search.router, prefix=settings.API_PREFIX)
    app.include_router(chat.router, prefix=settings.API_PREFIX)
    app.include_router(analysis.router, prefix=settings.API_PREFIX)
    app.include_router(sessions.router, prefix=settings.API_PREFIX)
    app.include_router(analytics.router, prefix=settings.API_PREFIX)
    app.include_router(agent.router, prefix=settings.API_PREFIX)
    app.include_router(admin.router, prefix=settings.API_PREFIX)

    return app


app = create_app()