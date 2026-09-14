"""Narrow orchestration layer for retrieval-gated policy answers."""

from __future__ import annotations

import logging
import time

from backend.generation import (
    CANONICAL_NOT_FOUND,
    BaseGenerator,
    GenerationConfigurationError,
    GenerationContractError,
    GenerationTimeoutError,
    GenerationUnavailableError,
)
from backend.retrieval import BaseRetriever
from backend.schemas import AskResponse, SourceInfo


logger = logging.getLogger(__name__)


_GENERATION_ERROR_CODES = {
    GenerationContractError: "generation_contract_error",
    GenerationTimeoutError: "generation_timeout",
    GenerationUnavailableError: "generation_unavailable",
    GenerationConfigurationError: "generation_configuration_error",
}


class QAService:
    """Own retrieval, abstention, generation, citation mapping, and timings."""

    def __init__(
        self,
        *,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        top_k: int,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.top_k = top_k

    def ask(self, question: str) -> AskResponse:
        total_started = time.perf_counter()
        generation_ms = 0.0
        gemini_called = False

        retrieval_started = time.perf_counter()
        try:
            retrieval = self.retriever.retrieve(question, top_k=self.top_k)
        except Exception as exc:
            self._log_failure("retrieval", "internal_error", exc)
            raise
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        if not retrieval.evidence_found:
            response = AskResponse(
                status="not_found",
                answer=CANONICAL_NOT_FOUND,
                sources=[],
            )
            self._log_completion(
                response=response,
                evidence_count=0,
                gemini_called=False,
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                total_ms=(time.perf_counter() - total_started) * 1000,
                question_len=len(question),
            )
            return response

        generation_started = time.perf_counter()
        gemini_called = True
        try:
            generated = self.generator.generate(question, retrieval.results)
        except Exception as exc:
            generation_ms = (time.perf_counter() - generation_started) * 1000
            code = next(
                (
                    value
                    for error_type, value in _GENERATION_ERROR_CODES.items()
                    if isinstance(exc, error_type)
                ),
                "internal_error",
            )
            self._log_failure("generation", code, exc)
            raise
        generation_ms = (time.perf_counter() - generation_started) * 1000

        if generated.status == "not_found":
            response = AskResponse(
                status="not_found",
                answer=CANONICAL_NOT_FOUND,
                sources=[],
            )
        else:
            source_map = {
                f"S{index}": result
                for index, result in enumerate(retrieval.results, start=1)
            }
            try:
                sources = [
                    self._to_source_info(source_id, source_map[source_id])
                    for source_id in generated.cited_source_ids
                ]
            except KeyError as exc:
                error = GenerationContractError(
                    "Validated generation referenced a missing retrieval source."
                )
                self._log_failure("citation_mapping", "generation_contract_error", error)
                raise error from exc

            response = AskResponse(
                status=generated.status,
                answer=generated.answer,
                sources=sources,
            )

        self._log_completion(
            response=response,
            evidence_count=len(retrieval.results),
            gemini_called=gemini_called,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=(time.perf_counter() - total_started) * 1000,
            question_len=len(question),
        )
        return response

    @staticmethod
    def _to_source_info(source_id, source) -> SourceInfo:
        return SourceInfo(
            source_id=source_id,
            chunk_id=source.chunk_id,
            document_id=source.document_id,
            document_title=source.document_title,
            section_title=source.section_title,
            version=source.version,
            source_file=source.source_file,
            evidence=source.text,
        )

    @staticmethod
    def _log_failure(stage: str, code: str, exc: Exception) -> None:
        logger.error(
            "event=request_failed stage=%s code=%s exception_type=%s",
            stage,
            code,
            type(exc).__name__,
        )

    @staticmethod
    def _log_completion(
        *,
        response: AskResponse,
        evidence_count: int,
        gemini_called: bool,
        retrieval_ms: float,
        generation_ms: float,
        total_ms: float,
        question_len: int,
    ) -> None:
        logger.info(
            "event=request_complete status=%s evidence_count=%d source_count=%d "
            "gemini_called=%s retrieval_ms=%.1f generation_ms=%.1f total_ms=%.1f "
            "question_len=%d",
            response.status,
            evidence_count,
            len(response.sources),
            str(gemini_called).lower(),
            retrieval_ms,
            generation_ms,
            total_ms,
            question_len,
        )
