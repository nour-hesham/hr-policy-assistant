"""QA orchestration tests using only fake dependencies."""

from __future__ import annotations

import logging

import pytest

from backend.generation import (
    CANONICAL_NOT_FOUND,
    GenerationContractError,
    GenerationResult,
    GenerationUnavailableError,
)
from backend.retrieval import RetrievalResponse, RetrievalResult
from backend.service import QAService


def result(rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=0.9 - rank / 100,
        chunk_id=f"DOC@2025-01::section-{rank}",
        document_id="DOC-001",
        document_title="Test Policy",
        version="2025-01",
        section_title=f"Section {rank}",
        source_file="test-policy.md",
        text=f"Canonical evidence {rank}.",
    )


class FakeRetriever:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = []

    def retrieve(self, question, filters=None, top_k=5):
        self.calls.append((question, top_k))
        if self.error:
            raise self.error
        return self.response


class FakeGenerator:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, question, sources):
        self.calls.append((question, sources))
        if self.error:
            raise self.error
        return self.response


def retrieval(evidence_found=True, results=None):
    items = [result()] if results is None and evidence_found else (results or [])
    return RetrievalResponse(
        query="q",
        model_id="test-model",
        top_k=3,
        top_score=0.9 if evidence_found else 0.2,
        evidence_found=evidence_found,
        results=items,
    )


def service(retrieval_response, generation_response=None, generation_error=None):
    retriever = FakeRetriever(response=retrieval_response)
    generator = FakeGenerator(response=generation_response, error=generation_error)
    return QAService(retriever=retriever, generator=generator, top_k=3), retriever, generator


def test_no_evidence_abstains_without_calling_generator():
    qa, retriever, generator = service(retrieval(False))
    response = qa.ask("Unsupported question")
    assert response.status == "not_found"
    assert response.answer == CANONICAL_NOT_FOUND
    assert response.sources == []
    assert generator.calls == []
    assert retriever.calls == [("Unsupported question", 3)]


def test_answer_maps_only_cited_trusted_source_metadata():
    generated = GenerationResult(
        status="answered",
        answer="Supported answer [S2].",
        cited_source_ids=["S2"],
    )
    qa, _, generator = service(retrieval(True, [result(1), result(2)]), generated)
    response = qa.ask("Question")
    assert response.status == "answered"
    assert len(response.sources) == 1
    source = response.sources[0]
    assert source.source_id == "S2"
    assert source.chunk_id == "DOC@2025-01::section-2"
    assert source.evidence == "Canonical evidence 2."
    assert not hasattr(source, "score")
    assert len(generator.calls) == 1


def test_partial_status_and_sources_are_preserved():
    generated = GenerationResult(
        status="partial",
        answer="Supported part [S1]; the remaining part is not specified.",
        cited_source_ids=["S1"],
    )
    qa, _, _ = service(retrieval(), generated)
    response = qa.ask("Compound question")
    assert response.status == "partial"
    assert [s.source_id for s in response.sources] == ["S1"]


def test_generator_not_found_discards_retrieval_sources():
    generated = GenerationResult(
        status="not_found",
        answer=CANONICAL_NOT_FOUND,
        cited_source_ids=[],
    )
    qa, _, _ = service(retrieval(), generated)
    response = qa.ask("Question")
    assert response.status == "not_found"
    assert response.sources == []
    assert response.answer == CANONICAL_NOT_FOUND


def test_missing_citation_mapping_is_contract_error():
    generated = GenerationResult(
        status="answered",
        answer="Bad reference [S9].",
        cited_source_ids=["S9"],
    )
    qa, _, _ = service(retrieval(), generated)
    with pytest.raises(GenerationContractError):
        qa.ask("Question")


def test_generator_error_is_propagated():
    qa, _, _ = service(
        retrieval(),
        generation_error=GenerationUnavailableError("provider unavailable"),
    )
    with pytest.raises(GenerationUnavailableError):
        qa.ask("Question")


def test_safe_completion_log_contains_metrics_not_question_or_answer(caplog):
    generated = GenerationResult(
        status="answered",
        answer="Sensitive generated text [S1].",
        cited_source_ids=["S1"],
    )
    qa, _, _ = service(retrieval(), generated)
    with caplog.at_level(logging.INFO, logger="backend.service"):
        qa.ask("Employee Alice personal question")
    text = caplog.text
    assert "event=request_complete" in text
    assert "status=answered" in text
    assert "gemini_called=true" in text
    assert "question_len=" in text
    assert "Employee Alice" not in text
    assert "Sensitive generated text" not in text
    assert "Canonical evidence" not in text


def test_safe_failure_log_omits_exception_message(caplog):
    qa, _, _ = service(
        retrieval(),
        generation_error=GenerationUnavailableError("secret provider response"),
    )
    with caplog.at_level(logging.ERROR, logger="backend.service"):
        with pytest.raises(GenerationUnavailableError):
            qa.ask("Question containing a private name")
    assert "event=request_failed" in caplog.text
    assert "code=generation_unavailable" in caplog.text
    assert "GenerationUnavailableError" in caplog.text
    assert "secret provider response" not in caplog.text
    assert "private name" not in caplog.text
