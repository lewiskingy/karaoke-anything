import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from audio_trombone.config import Settings
from audio_trombone.service import TromboneService


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

settings = Settings.from_environment()
service = TromboneService(settings)


class DemucsSettingsUpdate(BaseModel):
    model: str | None = None
    device: str | None = None
    segment_seconds: float | None = Field(default=None, gt=0)
    overlap: float | None = Field(default=None, ge=0, lt=1)
    shifts: int | None = Field(default=None, ge=0)
    vocal_reduction: float | None = Field(default=None, ge=0, le=1)


class ConvTasNetSettingsUpdate(BaseModel):
    model_path: str | None = None
    device: str | None = None
    segment_seconds: float | None = Field(default=None, gt=0)
    vocal_reduction: float | None = Field(default=None, ge=0, le=1)
    vocal_source_index: int | None = Field(default=None, ge=0)
    accompaniment_source_index: int | None = Field(default=None, ge=0)


class CentreReductionSettingsUpdate(BaseModel):
    reduction: float | None = Field(default=None, ge=0, le=1)


class RuntimeSettingsUpdate(BaseModel):
    processor: str | None = None
    demucs: DemucsSettingsUpdate | None = None
    convtasnet: ConvTasNetSettingsUpdate | None = None
    centre_reduction: CentreReductionSettingsUpdate | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    await service.start()
    try:
        yield
    finally:
        await service.stop()


app = FastAPI(
    title="Karaoke Anything",
    version="0.4.0",
    description=(
        "UDP media pipeline with pluggable processors and runtime controls."
    ),
    lifespan=lifespan,
)


@app.get("/", response_class=FileResponse)
async def settings_page() -> FileResponse:
    return FileResponse(Path(__file__).with_name("settings.html"))


@app.get("/health")
async def health() -> dict:
    return service.health()


@app.get("/status")
async def status() -> dict:
    return service.status()


@app.get("/processors")
async def processors() -> dict:
    return service.processor_registry.describe()


@app.get("/api/settings")
async def get_runtime_settings() -> dict:
    return service.runtime_settings()


def _flatten_updates(payload: RuntimeSettingsUpdate) -> dict[str, object]:
    current = service.settings
    updates: dict[str, object] = {}
    if payload.processor is not None and payload.processor != current.processor:
        updates["processor"] = payload.processor
    if payload.demucs is not None:
        mapping = {
            "model": "demucs_model",
            "device": "demucs_device",
            "segment_seconds": "demucs_segment_seconds",
            "overlap": "demucs_overlap",
            "shifts": "demucs_shifts",
            "vocal_reduction": "demucs_vocal_reduction",
        }
        for request_name, settings_name in mapping.items():
            value = getattr(payload.demucs, request_name)
            if value is not None and value != getattr(current, settings_name):
                updates[settings_name] = value
    if payload.convtasnet is not None:
        mapping = {
            "model_path": "convtasnet_model_path",
            "device": "convtasnet_device",
            "segment_seconds": "convtasnet_segment_seconds",
            "vocal_reduction": "convtasnet_vocal_reduction",
            "vocal_source_index": "convtasnet_vocal_source_index",
            "accompaniment_source_index": "convtasnet_accompaniment_source_index",
        }
        for request_name, settings_name in mapping.items():
            value = getattr(payload.convtasnet, request_name)
            if value is not None and value != getattr(current, settings_name):
                updates[settings_name] = value
    if payload.centre_reduction is not None:
        value = payload.centre_reduction.reduction
        if value is not None and value != current.centre_reduction:
            updates["centre_reduction"] = value
    return updates


async def _apply_settings(payload: RuntimeSettingsUpdate) -> dict:
    updates = _flatten_updates(payload)
    if not updates:
        return {
            "status": "unchanged",
            "processor_restarted": False,
            "settings": service.runtime_settings(),
        }
    try:
        result = await service.update_runtime_settings(updates)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "applied", **result, "settings": service.runtime_settings()}


@app.patch("/api/settings")
async def patch_runtime_settings(payload: RuntimeSettingsUpdate) -> dict:
    return await _apply_settings(payload)


@app.put("/api/settings")
async def put_runtime_settings(payload: RuntimeSettingsUpdate) -> dict:
    return await _apply_settings(payload)


@app.delete("/api/settings")
async def delete_runtime_settings() -> dict:
    try:
        result = await service.restore_startup_settings()
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "restored", **result, "settings": service.runtime_settings()}


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
