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

# CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Import and register API routes ───────────────────────────
from app.api import health, webhooks, batch, dashboard, payments, metrics, model

app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(webhooks.router, prefix="/api/v1", tags=["Webhooks"])
app.include_router(batch.router, prefix="/api/v1", tags=["Batch"])
app.include_router(dashboard.router, prefix="/api/v1", tags=["Dashboard"])
app.include_router(payments.router, prefix="/api/v1", tags=["Payments"])
app.include_router(metrics.router, prefix="/api/v1", tags=["Metrics"])
app.include_router(model.router, prefix="/api/v1", tags=["Model"])


@app.get("/")
async def root():
    return {
        "name": "PayRevive",
        "version": "1.0.0",
        "description": "AI-Powered Payment Recovery Engine",
        "docs": "/docs",
        "track": "03 — AI Revenue Recovery",
    }
