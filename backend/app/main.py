from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import chat, explain, predict
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting MedCareAI...")
    yield
    print("Shutting down MedCareAI...")


def create_app():
    app = FastAPI(
        title=settings.app_name,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    from app.api.v1 import chat, predict, explain

    app.include_router(predict.router, prefix="/api/v1/predict")
    app.include_router(chat.router, prefix="/api/v1/chat")
    app.include_router(explain.router, prefix="/api/v1/explain")

    return app


app = create_app()