import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routes import router, load_model


def create_app():
    app = FastAPI(title="Fraud API", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if os.path.isdir("frontend"):
        app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

    app.include_router(router)

    model_path = os.getenv("MODEL_PATH", "artifacts/models/xgb.joblib")
    load_model(model_path)

    return app


app = create_app()
