"""
MedCareAI FastAPI Application Entry Point.

This module initializes the FastAPI application, configures middleware,
sets up lifespan events, and includes all API routers.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown events:
    - Startup: Initialize database connections, load ML models, setup ChromaDB
    - Shutdown: Close connections and cleanup resources
    """
    # Startup
    print(f"Starting {settings.app_name} in {settings.app_env} mode...")

    # TODO: Initialize database connection pool
    # TODO: Load ML models into memory
    # TODO: Initialize ChromaDB client
    # TODO: Setup Redis connection

    yield

    # Shutdown
    print(f"Shutting down {settings.app_name}...")

    # TODO: Close database connections
    # TODO: Cleanup resources


def create_app() -> FastAPI:
    """
    Application factory function.

    Creates and configures the FastAPI application instance.
    """
    app = FastAPI(
        title=settings.app_name,
        description="Advanced medical decision-support platform combining ML, LLM, and RAG",
        version="0.1.0",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint for load balancers and monitoring."""
        return {
            "status": "healthy",
            "app": settings.app_name,
            "environment": settings.app_env,
        }

    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "message": f"Welcome to {settings.app_name} API",
            "docs": "/docs",
            "health": "/health",
        }

    # Include API routers
    from app.api.v1 import predict

    app.include_router(
        predict.router,
        prefix="/api/v1/predict",
        tags=["Prediction"],
    )

    return app


app = create_app()
