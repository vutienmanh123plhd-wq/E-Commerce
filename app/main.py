"""Application factory. Run with:  uvicorn app.main:app"""
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import config
from app.control.search_controller import AppServices, build_real_services, build_router


def create_app(services: AppServices | None = None) -> FastAPI:
    services = services or build_real_services()
    app = FastAPI(title="E-Commerce Multimodal Product Search")
    app.include_router(build_router(services))

    @app.middleware("http")
    async def revalidate_frontend(request, call_next):
        # Without this, browsers reuse stale JS/CSS after the code changes.
        response = await call_next(request)
        if not request.url.path.startswith(("/api/", "/images/", "/uploads/")):
            response.headers["Cache-Control"] = "no-cache"
        return response

    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/images", StaticFiles(directory=config.IMAGES_DIR), name="images")
    app.mount("/uploads", StaticFiles(directory=services.uploads.directory), name="uploads")
    app.mount("/", StaticFiles(directory=config.STATIC_DIR, html=True), name="static")
    return app


def __getattr__(name):  # lazily build the real app so importing this module stays cheap
    if name == "app":
        logging.basicConfig(level=logging.INFO)
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
