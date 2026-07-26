import asyncio
import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from audio_trombone.config import Settings
from audio_trombone.service import TromboneService


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

settings = Settings.from_environment()
service = TromboneService(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await service.start()
    try:
        yield
    finally:
        await service.stop()


app = FastAPI(
    title="Karaoke Anything",
    version="0.1.0",
    description=(
        "UDP media pipeline with a pluggable processor interface. "
        "The default passthrough processor returns media unchanged."
    ),
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    return service.health()


@app.get("/status")
async def status() -> dict:
    return service.status()


@app.get("/processors")
async def processors() -> dict:
    return service.processor_registry.describe()


@app.post("/processor/reset")
async def reset_processor() -> dict:
    await service.reset_processor()
    return {
        "status": "ok",
        "processor": service.processor.name,
        "message": "Processor state reset",
    }


@app.put("/processor/{processor_name}")
async def select_processor(processor_name: str) -> dict:
    try:
        await service.select_processor(processor_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "ok",
        "processor": service.processor.name,
        "message": "Processor selected",
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return service.prometheus_metrics()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.http_host,
        port=settings.http_port,
        log_level=settings.log_level.lower(),
        app_dir="/app/app",
    )
