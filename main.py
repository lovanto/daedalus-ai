from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from db import close_pool
from ollama.client import OllamaClient
from routers import assist, eval, analyze, chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Daedalus AI Layer",
    description="LLM-powered assist endpoints for the ADLC platform.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.GO_API_URL, "http://localhost:3010"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(assist.router, prefix="/ai")
app.include_router(eval.router, prefix="/ai")
app.include_router(analyze.router, prefix="/ai")
app.include_router(chat.router, prefix="/ai")


# ---------------------------------------------------------------------------
# Request ID + structured request logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def request_id_logger(request: Request, call_next) -> Response:
    req_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    start = time.monotonic()
    response: Response = await call_next(request)
    duration_ms = int((time.monotonic() - start) * 1000)
    response.headers["x-request-id"] = req_id
    logger.info(
        "http method=%s path=%s status=%d duration_ms=%d request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        req_id,
    )
    return response


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup() -> None:
    client = OllamaClient()
    reachable, models = await client.is_reachable()
    if reachable:
        logger.info("Ollama reachable — available models: %s", models)
        if settings.OLLAMA_MODEL not in models:
            logger.warning(
                "Configured model '%s' not found locally. Available: %s",
                settings.OLLAMA_MODEL,
                models,
            )
    else:
        logger.warning(
            "Ollama is NOT reachable at %s — AI endpoints will fail until it starts.",
            settings.OLLAMA_BASE_URL,
        )
    logger.info(
        "daedalus-ai ready on port %s", settings.PYTHON_AI_PORT
    )


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await close_pool()


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/ai/health", tags=["system"])
async def health() -> dict:
    """
    Returns service status and Ollama reachability.
    Pings GET /api/tags on the configured Ollama instance.
    """
    client = OllamaClient()
    reachable, models = await client.is_reachable()
    return {
        "status": "ok",
        "service": "daedalus-ai",
        "ollama_reachable": reachable,
        "ollama_url": settings.OLLAMA_BASE_URL,
        "model": settings.OLLAMA_MODEL,
        "available_models": models,
    }
