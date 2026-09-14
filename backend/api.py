"""FastAPI entry point for the stateless HR policy assistant."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.config import config
from backend.generation import (
    GenerationConfigurationError,
    GenerationContractError,
    GenerationTimeoutError,
    GenerationUnavailableError,
    GeminiGenerator,
)
from backend.ingestion import ingest_directory
from backend.logging_config import configure_logging
from backend.retrieval import EmbeddingRetriever
from backend.schemas import AskRequest, AskResponse, HealthResponse
from backend.service import QAService


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeResources:
    service: QAService
    chunk_count: int
    embedding_model_id: str
    generation_model_id: str


class SafeAPIError(Exception):
    """Error whose code and message are safe to expose to API clients."""

    def __init__(self, *, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def build_runtime() -> RuntimeResources:
    """Validate configuration and initialize shared production resources once."""
    if not config.gemini_api_key:
        raise GenerationConfigurationError("GEMINI_API_KEY is required.")
    if not 1 <= config.top_k <= 5:
        raise ValueError("TOP_K must be between 1 and 5.")
    if not -1.0 <= config.min_relevance_score <= 1.0:
        raise ValueError("MIN_RELEVANCE_SCORE must be between -1 and 1.")

    repository_root = Path(__file__).resolve().parent.parent
    policies_dir = repository_root / config.policies_dir
    chunks = ingest_directory(policies_dir)
    if not chunks:
        raise ValueError("No policy chunks were loaded.")

    retriever = EmbeddingRetriever(
        chunks,
        model_id=config.embedding_model,
        threshold=config.min_relevance_score,
        query_prefix=config.bge_query_prefix,
    )
    generator = GeminiGenerator(
        api_key=config.gemini_api_key,
        model_id=config.generation_model,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        timeout_seconds=config.generation_timeout,
    )
    service = QAService(
        retriever=retriever,
        generator=generator,
        top_k=config.top_k,
    )
    return RuntimeResources(
        service=service,
        chunk_count=len(chunks),
        embedding_model_id=config.embedding_model,
        generation_model_id=config.generation_model,
    )


def create_app(
    initializer: Callable[[], RuntimeResources] | None = None,
) -> FastAPI:
    runtime_initializer = initializer or build_runtime

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        configure_logging(config.log_level)
        application.state.ready = False
        try:
            resources = runtime_initializer()
        except Exception as exc:
            logger.error(
                "event=startup_failed stage=initialization code=startup_failed "
                "exception_type=%s",
                type(exc).__name__,
            )
            raise RuntimeError("Application startup failed.") from exc

        application.state.qa_service = resources.service
        application.state.chunk_count = resources.chunk_count
        application.state.embedding_model_id = resources.embedding_model_id
        application.state.generation_model_id = resources.generation_model_id
        application.state.ready = True
        logger.info(
            "event=startup_complete chunk_count=%d embedding_model=%s "
            "generation_model=%s",
            resources.chunk_count,
            resources.embedding_model_id,
            resources.generation_model_id,
        )
        try:
            yield
        finally:
            application.state.ready = False

    application = FastAPI(
        title="HR Policy Assistant",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @application.exception_handler(SafeAPIError)
    async def safe_error_handler(_: Request, exc: SafeAPIError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _: Request, __: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_request",
                    "message": (
                        "Request body must contain only a non-empty question "
                        "of at most 1,000 characters."
                    ),
                }
            },
        )

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        if not getattr(request.app.state, "ready", False):
            raise SafeAPIError(
                status_code=503,
                code="service_not_ready",
                message="The policy assistant is not ready.",
            )
        logger.debug("event=health_check ready=true")
        return HealthResponse(
            status="ready",
            ready=True,
            loaded_chunk_count=request.app.state.chunk_count,
            embedding_model_id=request.app.state.embedding_model_id,
            generation_model_id=request.app.state.generation_model_id,
        )

    @application.post("/api/v1/ask", response_model=AskResponse)
    def ask(payload: AskRequest, request: Request) -> AskResponse:
        if not getattr(request.app.state, "ready", False):
            raise SafeAPIError(
                status_code=503,
                code="service_not_ready",
                message="The policy assistant is not ready.",
            )
        try:
            return request.app.state.qa_service.ask(payload.question)
        except GenerationContractError as exc:
            raise SafeAPIError(
                status_code=502,
                code="invalid_generation_response",
                message="The answer service returned an invalid response.",
            ) from exc
        except GenerationTimeoutError as exc:
            raise SafeAPIError(
                status_code=504,
                code="generation_timeout",
                message="The answer service timed out. Please try again.",
            ) from exc
        except (GenerationUnavailableError, GenerationConfigurationError) as exc:
            raise SafeAPIError(
                status_code=503,
                code="generation_unavailable",
                message="The answer service is temporarily unavailable.",
            ) from exc
        except SafeAPIError:
            raise
        except Exception as exc:
            logger.error(
                "event=request_failed stage=api code=internal_error exception_type=%s",
                type(exc).__name__,
            )
            raise SafeAPIError(
                status_code=500,
                code="internal_error",
                message="The request could not be processed.",
            ) from exc

    return application


app = create_app()
