"""REST API.

Endpoints:
    POST /analyze          — analyze a batch of events
    GET  /incidents        — list incidents, filterable by IP or analysis
    GET  /incidents/{id}   — full verdict for one incident
    GET  /analyses/{id}    — status of one analysis
    GET  /blocklist        — contained IPs
    GET  /health           — health check, unauthenticated
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import JSONResponse

from app.actuators.blocker import IPBlocker
from app.actuators.notifier import Notifier
from app.config import Settings, get_settings
from app.db import Database
from app.orchestrator import Orchestrator
from app.pipeline.detector import Detector, ModelNotLoaded
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BlocklistEntry,
    HealthResponse,
    IncidentListItem,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
logger = logging.getLogger("detector")


def build_llm_client(settings: Settings) -> Any | None:
    """Instantiate the LLM client for the active provider, or None when there is none.

    Without a key the service keeps working: the agents are replaced by the
    deterministic path. That keeps the project runnable for anyone who just wants
    to see the pipeline work.

    Both branches return an object exposing the same `messages.create` surface, so
    the agent loop, the tools and the prompts are provider-agnostic. See
    `app.agents.providers`.
    """
    provider = settings.active_provider
    if not provider:
        logger.warning(
            "no ANTHROPIC_API_KEY or GEMINI_API_KEY — the agents will stay in "
            "deterministic mode"
        )
        return None
    try:
        if provider == "gemini":
            from app.agents.providers import GeminiClient

            return GeminiClient(settings.gemini_api_key)

        import anthropic

        return anthropic.Anthropic(api_key=settings.anthropic_api_key)
    except Exception:  # pragma: no cover - import/credential failure
        logger.exception("failed to initialize the %s client", provider)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.database = Database(settings.database_path)
    app.state.detector = Detector(settings.model_path, settings.novelty_target_fpr)
    app.state.orchestrator = Orchestrator(
        settings=settings,
        detector=app.state.detector,
        database=app.state.database,
        blocker=IPBlocker(settings.block_mode, settings.block_command),
        notifier=Notifier(settings.telegram_bot_token, settings.telegram_chat_id),
        llm_client=build_llm_client(settings),
    )
    logger.info(
        "service ready | model=%s | agents=%s | blocking=%s",
        "loaded" if app.state.detector.is_loaded else "MISSING",
        "llm" if app.state.orchestrator.triage else "deterministic",
        settings.block_mode,
    )
    yield


app = FastAPI(
    title="Intelligent Agent-Based Detection of Malicious HTTP Logs",
    description=(
        "Funnel detection pipeline: deterministic per-event scoring, correlation "
        "into incidents, and two LLM agents for triage and response."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def require_api_key(x_api_key: str = Header(default="")) -> None:
    settings = get_settings()
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid X-API-Key header",
        )


def get_orchestrator() -> Orchestrator:
    return app.state.orchestrator


def get_database() -> Database:
    return app.state.database


@app.get("/", tags=["meta"])
def root() -> dict[str, Any]:
    return {
        "service": "log-anomaly-detection",
        "version": app.version,
        "docs": "/docs",
        "endpoints": ["/analyze", "/incidents", "/analyses/{id}", "/blocklist", "/health"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    detector: Detector = app.state.detector
    settings: Settings = app.state.settings
    orchestrator: Orchestrator = app.state.orchestrator
    return HealthResponse(
        status="ok" if detector.is_loaded else "degraded",
        model_loaded=detector.is_loaded,
        llm_enabled=orchestrator.triage is not None,
        block_mode=settings.block_mode,
        detail={
            "model_path": str(settings.model_path),
            "load_error": detector.load_error,
            "trained_at": detector.metadata.get("trained_at"),
            "approach": detector.metadata.get("approach"),
            "target_fpr": detector.applied_fpr,
            "llm_provider": settings.active_provider or None,
            "llm_model": settings.active_model if orchestrator.triage else None,
        },
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["analysis"],
    dependencies=[Depends(require_api_key)],
)
def analyze(
    request: AnalyzeRequest,
    background: BackgroundTasks,
    orchestrator: Orchestrator = Depends(get_orchestrator),
) -> Any:
    settings = get_settings()
    if len(request.events) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"the batch holds {len(request.events)} events; the maximum is "
                f"{settings.max_batch_size}"
            ),
        )

    try:
        response, pending = orchestrator.analyze(request)
    except ModelNotLoaded as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"detection model unavailable: {exc}",
        ) from exc

    if pending:
        background.add_task(
            orchestrator.process_pending, response.analysis_id, pending, request.dry_run
        )
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED, content=response.model_dump(mode="json")
        )
    return response


@app.get(
    "/incidents",
    response_model=list[IncidentListItem],
    tags=["analysis"],
    dependencies=[Depends(require_api_key)],
)
def list_incidents(
    limit: int = Query(default=50, ge=1, le=500),
    ip: str | None = None,
    analysis_id: str | None = None,
    database: Database = Depends(get_database),
) -> list[IncidentListItem]:
    rows = database.list_incidents(limit=limit, ip=ip, analysis_id=analysis_id)
    return [IncidentListItem(**row) for row in rows]


@app.get(
    "/incidents/{incident_id}",
    tags=["analysis"],
    dependencies=[Depends(require_api_key)],
)
def get_incident(
    incident_id: str, database: Database = Depends(get_database)
) -> dict[str, Any]:
    incident = database.get_incident(incident_id)
    if incident is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="incident not found"
        )
    return incident


@app.get(
    "/analyses/{analysis_id}",
    tags=["analysis"],
    dependencies=[Depends(require_api_key)],
)
def get_analysis(
    analysis_id: str, database: Database = Depends(get_database)
) -> dict[str, Any]:
    record = database.get_analysis_status(analysis_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="analysis not found"
        )
    record["incidents"] = database.list_incidents(limit=500, analysis_id=analysis_id)
    return record


@app.get(
    "/blocklist",
    response_model=list[BlocklistEntry],
    tags=["response"],
    dependencies=[Depends(require_api_key)],
)
def blocklist(database: Database = Depends(get_database)) -> list[BlocklistEntry]:
    return [BlocklistEntry(**row) for row in database.list_blocklist()]
