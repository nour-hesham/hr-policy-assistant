"""Deterministic tests for grounded generation and citation validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from backend.generation import (
    CANONICAL_NOT_FOUND,
    GenerationConfigurationError,
    GenerationContractError,
    GenerationTimeoutError,
    GenerationUnavailableError,
    GeminiGenerator,
    build_grounded_prompt,
    validate_model_answer,
)
from backend.retrieval import RetrievalResult


def source(rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=0.91,
        chunk_id=f"DOC@2025-01::section-{rank}",
        document_id="DOC-001",
        document_title="Leave Policy",
        version="2025-01",
        section_title=f"Section {rank}",
        source_file="policy.md",
        text=f"Trusted policy evidence {rank}.",
    )


class FakeModels:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def generator(response=None, error: Exception | None = None):
    models = FakeModels(response=response, error=error)
    client = SimpleNamespace(models=models)
    return GeminiGenerator(
        api_key="unused-test-key",
        model_id="test-gemini",
        client=client,
    ), models


def test_prompt_labels_sources_and_omits_scores():
    prompt, mapping = build_grounded_prompt("What is the rule?", [source(1), source(2)])
    assert set(mapping) == {"S1", "S2"}
    assert "BEGIN SOURCE S1" in prompt
    assert "Leave Policy" in prompt
    assert "Trusted policy evidence 2." in prompt
    assert "0.91" not in prompt


def test_valid_answered_response_is_returned():
    response = SimpleNamespace(
        parsed={
            "status": "answered",
            "answer": "The policy supports this answer [S1].",
            "cited_source_ids": ["S1"],
        }
    )
    provider, models = generator(response=response)
    result = provider.generate("Question?", [source()])
    assert result.status == "answered"
    assert result.cited_source_ids == ["S1"]
    assert len(models.calls) == 1
    assert models.calls[0]["model"] == "test-gemini"
    config = models.calls[0]["config"]
    assert config.temperature == 0.1
    assert config.max_output_tokens == 400
    assert config.automatic_function_calling.disable is True
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema["additionalProperties"] is False


def test_valid_partial_response_is_returned():
    raw = {
        "status": "partial",
        "answer": "The budget is stated [S1], but spouse eligibility is not specified.",
        "cited_source_ids": ["S1", "S1"],
    }
    result = validate_model_answer(raw, {"S1"})
    assert result.status == "partial"
    assert result.cited_source_ids == ["S1"]


def test_not_found_wording_is_canonical_and_has_no_sources():
    raw = {"status": "not_found", "answer": "Unknown.", "cited_source_ids": []}
    result = validate_model_answer(raw, {"S1"})
    assert result.answer == CANONICAL_NOT_FOUND
    assert result.cited_source_ids == []


@pytest.mark.parametrize(
    "raw",
    [
        {"status": "answered", "answer": "No citation", "cited_source_ids": []},
        {"status": "answered", "answer": "Claim [S2]", "cited_source_ids": ["S2"]},
        {"status": "answered", "answer": "Claim [S1]", "cited_source_ids": ["S1", "S2"]},
        {"status": "not_found", "answer": "Unknown [S1]", "cited_source_ids": ["S1"]},
        {"status": "answered", "answer": "   ", "cited_source_ids": ["S1"]},
        {"status": "answered", "answer": "Claim [S1]", "cited_source_ids": ["S1"], "extra": True},
    ],
)
def test_invalid_schema_or_citations_are_rejected(raw):
    with pytest.raises(GenerationContractError):
        validate_model_answer(raw, {"S1"})


def test_json_text_fallback_is_validated():
    response = SimpleNamespace(
        parsed=None,
        text='{"status":"answered","answer":"Supported [S1]","cited_source_ids":["S1"]}',
    )
    provider, _ = generator(response=response)
    assert provider.generate("Question?", [source()]).status == "answered"


def test_generation_without_sources_is_rejected_before_provider_call():
    provider, models = generator(response=SimpleNamespace(parsed={}))
    with pytest.raises(GenerationContractError):
        provider.generate("Question?", [])
    assert models.calls == []


def test_missing_key_is_a_configuration_error():
    with pytest.raises(GenerationConfigurationError):
        GeminiGenerator(api_key="", model_id="test")


def test_provider_timeout_is_mapped():
    provider, _ = generator(error=TimeoutError("safe synthetic timeout"))
    with pytest.raises(GenerationTimeoutError):
        provider.generate("Question?", [source()])


@pytest.mark.parametrize("status_code", [401, 403, 429, 500, 503])
def test_provider_api_failures_are_mapped(status_code):
    error = genai_errors.APIError(
        status_code,
        {"error": {"code": status_code, "status": "ERROR", "message": "synthetic"}},
    )
    provider, _ = generator(error=error)
    with pytest.raises(GenerationUnavailableError):
        provider.generate("Question?", [source()])


def test_provider_api_timeout_is_mapped():
    error = genai_errors.APIError(
        504,
        {"error": {"code": 504, "status": "DEADLINE_EXCEEDED", "message": "synthetic"}},
    )
    provider, _ = generator(error=error)
    with pytest.raises(GenerationTimeoutError):
        provider.generate("Question?", [source()])
