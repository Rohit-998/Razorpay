"""PayRevive — FastAPI Application Entry Point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.ml.classifier import classifier
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    settings = get_settings()
    logger.info("payrevive.starting", env=settings.app_env)

    # Load ML model if available
    if classifier.load():
        logger.info("payrevive.model_loaded")
    else:
        logger.warning("payrevive.no_model", hint="Run POST /api/v1/model/train first")

    yield

    logger.info("payrevive.shutdown")


app = FastAPI(
    title="PayRevive",
    description="AI-Powered Payment Recovery Engine — Razorpay AI Buildathon Track 03",
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi.responses import JSONResponse
import traceback
import uuid
from fastapi import Request


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Log the traceback, return a reference to it.

    This used to put `traceback.format_exc()` in the response body. With `allow_origins=["*"]`
    below, that hands any caller the absolute paths of the source tree, the frame locals'
    surrounding code, and the names of internal modules — which is a free map of the service
    for anyone probing it, and it is served on the error path, where probing lands.

    The traceback still goes to the log in full, keyed by an id the caller is given, so
    debugging a real failure is a grep rather than a guess.
    """
    incident = uuid.uuid4().hex[:12]
    logger.error(
        "unhandled_exception",
        incident=incident,
        path=request.url.path,
        error=str(exc),
        traceback=traceback.format_exc(),
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "the request failed inside the service",
            "incident": incident,
            "where_to_look": f"search the server log for incident={incident}",
        },
    )


# CORS — open, because the dashboard is served from a different origin in every
# environment this runs in and none of these endpoints accept credentials. Note the
# combination this forms with the handler above: an open origin policy plus a verbose
# error body is how a stack trace becomes public, which is why the body is terse.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Import and register API routes ───────────────────────────
# `evaluation`, not `eval` — the module serves the eval harness's output, and naming it
# after the harness would shadow the builtin in this namespace.
from app.api import (
    health, webhooks, batch, compliance, dashboard, payments, metrics, model, pipeline,
    evaluation,
)

app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(webhooks.router, prefix="/api/v1", tags=["Webhooks"])
app.include_router(batch.router, prefix="/api/v1", tags=["Batch"])
app.include_router(dashboard.router, prefix="/api/v1", tags=["Dashboard"])
app.include_router(compliance.router, prefix="/api/v1", tags=["Compliance"])
app.include_router(payments.router, prefix="/api/v1", tags=["Payments"])
app.include_router(metrics.router, prefix="/api/v1", tags=["Metrics"])
app.include_router(model.router, prefix="/api/v1", tags=["Model"])
app.include_router(pipeline.router, prefix="/api/v1", tags=["Pipeline"])
app.include_router(evaluation.router, prefix="/api/v1", tags=["Evaluation"])


@app.get("/")
async def root():
    return {
        "name": "PayRevive",
        "version": "1.0.0",
        "description": "AI-Powered Payment Recovery Engine",
        "docs": "/docs",
        "track": "03 — AI Revenue Recovery",
    }
