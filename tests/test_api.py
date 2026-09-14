"""FastAPI contract tests with injected fakes only."""

from __future__ import annotations

import inspect

import pytest
from fastapi.testclient import TestClient

from backend.api import RuntimeResources, create_app
from backend.generation import (
    CANONICAL_NOT_FOUND,
    GenerationConfigurationError,
    GenerationContractError,
    GenerationTimeoutError,
    GenerationUnavailableError,
)
from backend.schemas import AskResponse, SourceInfo


class FakeService:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.questions = []

    def ask(self, question):
        self.questions.append(question)
        if self.error:
            raise self.error
        return self.response


def answered_response(status="answered"):
    return AskResponse(
        status=status,
        answer="Employees receive 21 days [S1].",
        sources=[
            SourceInfo(
                source_id="S1",
                chunk_id="HR-POL-001@2025-03::entitlement",
                document_id="HR-POL-001",
                document_title="Annual Leave Policy",
                section_title="Entitlement",
                version="2025-03",
                source_file="01-annual-leave.md",
                evidence="Employees receive 21 working days.",
            )
        ],
    )


def make_client(*, response=None, error=None):
    service = FakeService(response=response or answered_response(), error=error)

    def initialize():
        return RuntimeResources(
            service=service,
            chunk_count=81,
            embedding_model_id="BAAI/bge-small-en-v1.5",
            generation_model_id="gemini-3.1-flash-lite",
        )

    return TestClient(create_app(initializer=initialize)), service


def test_health_reports_ready_without_asking_service():
    client, service = make_client()
    with client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "ready": True,
        "loaded_chunk_count": 81,
        "embedding_model_id": "BAAI/bge-small-en-v1.5",
        "generation_model_id": "gemini-3.1-flash-lite",
    }
    assert service.questions == []


def test_ask_returns_stable_success_schema_without_scores():
    client, service = make_client()
    with client:
        response = client.post("/api/v1/ask", json={"question": "  Leave?  "})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["sources"][0]["source_id"] == "S1"
    assert "score" not in body["sources"][0]
    assert service.questions == ["Leave?"]


def test_legitimate_not_found_is_http_200():
    response = AskResponse(
        status="not_found",
        answer=CANONICAL_NOT_FOUND,
        sources=[],
    )
    client, _ = make_client(response=response)
    with client:
        result = client.post("/api/v1/ask", json={"question": "Who is the CEO?"})
    assert result.status_code == 200
    assert result.json()["sources"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 1001},
        {"question": "valid", "top_k": 5},
        {"question": "valid", "unexpected": True},
    ],
)
def test_invalid_requests_use_safe_422_envelope(payload):
    client, service = make_client()
    with client:
        response = client.post("/api/v1/ask", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert service.questions == []


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (GenerationContractError("raw response"), 502, "invalid_generation_response"),
        (GenerationUnavailableError("provider body"), 503, "generation_unavailable"),
        (GenerationConfigurationError("key value"), 503, "generation_unavailable"),
        (GenerationTimeoutError("timeout detail"), 504, "generation_timeout"),
        (RuntimeError("internal detail"), 500, "internal_error"),
    ],
)
def test_failures_map_to_safe_envelopes(error, status, code):
    client, _ = make_client(error=error)
    with client:
        response = client.post("/api/v1/ask", json={"question": "Question"})
    assert response.status_code == status
    body = response.json()
    assert body["error"]["code"] == code
    serialized = response.text
    for unsafe in ["raw response", "provider body", "key value", "internal detail"]:
        assert unsafe not in serialized


def test_only_two_public_routes_are_exposed():
    application = create_app(initializer=lambda: None)  # lifespan is not entered
    routes = {(route.path, tuple(sorted(route.methods or []))) for route in application.routes}
    assert routes == {
        ("/health", ("GET",)),
        ("/api/v1/ask", ("POST",)),
    }


def test_ask_route_is_synchronous():
    application = create_app(initializer=lambda: None)
    ask_route = next(route for route in application.routes if route.path == "/api/v1/ask")
    assert not inspect.iscoroutinefunction(ask_route.endpoint)


def test_startup_failure_is_safe_and_stops_application(capsys):
    def fail():
        raise ValueError("private startup detail")

    client = TestClient(create_app(initializer=fail))
    with pytest.raises(RuntimeError, match="Application startup failed"):
        with client:
            pass
    output = capsys.readouterr().out
    assert "private startup detail" not in output
    assert "exception_type=ValueError" in output
