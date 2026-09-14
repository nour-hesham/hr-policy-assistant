"""
Deterministic retrieval tests.

These tests exercise the EmbeddingRetriever contract using tiny synthetic
vectors injected directly — no model download, no network access, no API key.

Strategy: subclass EmbeddingRetriever and override _embeddings and _model
with a fake that returns controlled vectors, keeping all ranking/threshold/
deduplication logic under test.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from unittest.mock import MagicMock

from backend.config import config, normalize_optional_path_environment
from backend.ingestion import ChunkRecord
from backend.retrieval import EmbeddingRetriever, RetrievalResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(
    chunk_id: str,
    *,
    document_id: str = "DOC-001",
    document_title: str = "Test Policy",
    version: str = "2025-01",
    section_title: str = "Section",
    section_order: int = 1,
    source_file: str = "test.md",
    content_hash: str = "sha256:abc",
    text: str = "Some text.",
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=document_title,
        version=version,
        section_title=section_title,
        section_order=section_order,
        source_file=source_file,
        content_hash=content_hash,
        text=text,
    )


def make_retriever(
    chunks: list[ChunkRecord],
    doc_embeddings: np.ndarray,
    query_vector: np.ndarray,
    threshold: float = 0.5,
) -> EmbeddingRetriever:
    """
    Build an EmbeddingRetriever with synthetic embeddings injected, bypassing
    SentenceTransformer model loading entirely.
    """
    # Bypass __init__ to avoid loading the real model.
    r = EmbeddingRetriever.__new__(EmbeddingRetriever)
    r.chunks = list(chunks)
    r.model_id = "test-model"
    r.threshold = threshold
    r.query_prefix = ""

    # L2-normalise the supplied synthetic vectors (mirrors production code).
    r._embeddings = EmbeddingRetriever._l2_normalize(doc_embeddings.astype(np.float32))

    # Fake model: encode() returns the preset query vector.
    fake_model = MagicMock()
    q_norm = EmbeddingRetriever._l2_normalize(
        query_vector.astype(np.float32).reshape(1, -1)
    )[0]
    fake_model.encode.return_value = q_norm
    r._model = fake_model

    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmptyCorpus:
    def test_returns_no_evidence_when_no_chunks(self):
        r = make_retriever([], np.empty((0, 3)), np.array([1.0, 0.0, 0.0]))
        resp = r.retrieve("anything")
        assert isinstance(resp, RetrievalResponse)
        assert resp.evidence_found is False
        assert resp.top_score == 0.0
        assert resp.results == []
        assert resp.query == "anything"
        assert resp.model_id == "test-model"


class TestRanking:
    """Correct descending cosine order is returned."""

    def _setup(self):
        # Three unit vectors — known cosine scores against query [1,0,0]:
        #   chunk-A: [1, 0, 0] -> score 1.0  (rank 1)
        #   chunk-B: [0, 1, 0] -> score 0.0  (rank 3, but below threshold)
        #   chunk-C: [0.8, 0.6, 0] (normalised) -> score ~0.8  (rank 2)
        chunks = [
            make_chunk("chunk-A", section_order=1),
            make_chunk("chunk-B", section_order=2),
            make_chunk("chunk-C", section_order=3),
        ]
        doc_emb = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.8, 0.6, 0.0]],
            dtype=np.float32,
        )
        q_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return chunks, doc_emb, q_vec

    def test_results_in_score_descending_order(self):
        chunks, doc_emb, q_vec = self._setup()
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.5)
        resp = r.retrieve("q")
        scores = [res.score for res in resp.results]
        assert scores == sorted(scores, reverse=True), "Results must be score-descending"

    def test_ranks_are_contiguous_from_one(self):
        chunks, doc_emb, q_vec = self._setup()
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.5)
        resp = r.retrieve("q")
        assert [res.rank for res in resp.results] == list(
            range(1, len(resp.results) + 1)
        )

    def test_top_score_matches_rank_one(self):
        chunks, doc_emb, q_vec = self._setup()
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.5)
        resp = r.retrieve("q")
        assert resp.top_score == pytest.approx(resp.results[0].score, abs=1e-5)

    def test_correct_chunk_at_rank_one(self):
        chunks, doc_emb, q_vec = self._setup()
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.5)
        resp = r.retrieve("q")
        assert resp.results[0].chunk_id == "chunk-A"

    def test_second_rank_chunk(self):
        chunks, doc_emb, q_vec = self._setup()
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.5)
        resp = r.retrieve("q")
        # chunk-C has score ~0.8 > chunk-B score 0.0; chunk-B below threshold
        assert resp.results[1].chunk_id == "chunk-C"


class TestThresholdFiltering:
    """Only results >= threshold are returned; top below threshold -> zero results."""

    def test_results_below_threshold_excluded(self):
        chunks = [
            make_chunk("hi", section_order=1),   # score 1.0
            make_chunk("lo", section_order=2),   # score 0.0
        ]
        doc_emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        q_vec = np.array([1.0, 0.0], dtype=np.float32)
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.5)
        resp = r.retrieve("q")
        ids = [res.chunk_id for res in resp.results]
        assert "lo" not in ids
        assert "hi" in ids

    def test_no_results_when_top_score_below_threshold(self):
        chunks = [make_chunk("only", section_order=1)]
        # doc vector orthogonal to query -> cosine score = 0.0
        doc_emb = np.array([[0.0, 1.0]], dtype=np.float32)
        q_vec = np.array([1.0, 0.0], dtype=np.float32)
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.5)
        resp = r.retrieve("q")
        assert resp.evidence_found is False
        assert resp.results == []
        assert resp.top_score == pytest.approx(0.0, abs=1e-5)

    def test_evidence_found_true_when_top_passes(self):
        chunks = [make_chunk("good", section_order=1)]
        doc_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        q_vec = np.array([1.0, 0.0], dtype=np.float32)
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.5)
        resp = r.retrieve("q")
        assert resp.evidence_found is True

    def test_threshold_boundary_inclusive(self):
        """Score exactly equal to threshold must be included."""
        chunks = [make_chunk("exact", section_order=1)]
        # Unit vectors -> cosine = 1.0; threshold = 1.0
        doc_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        q_vec = np.array([1.0, 0.0], dtype=np.float32)
        r = make_retriever(chunks, doc_emb, q_vec, threshold=1.0)
        resp = r.retrieve("q")
        assert resp.evidence_found is True
        assert len(resp.results) == 1


class TestTopK:
    """top_k caps the number of returned results."""

    def test_top_k_limits_results(self):
        n = 6
        chunks = [make_chunk(f"c{i}", section_order=i) for i in range(n)]
        # All vectors in same direction -> all scores near 1.0
        doc_emb = np.ones((n, 3), dtype=np.float32)
        q_vec = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.5)
        resp = r.retrieve("q", top_k=3)
        assert len(resp.results) <= 3
        assert resp.top_k == 3

    def test_fewer_results_than_top_k_when_corpus_small(self):
        chunks = [make_chunk("only", section_order=1)]
        doc_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        q_vec = np.array([1.0, 0.0], dtype=np.float32)
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.5)
        resp = r.retrieve("q", top_k=5)
        assert len(resp.results) == 1


class TestDeduplication:
    """Exact chunk_id duplicates are never returned twice."""

    def test_no_duplicate_chunk_ids(self):
        # Two chunks with the same chunk_id (edge case in test data).
        c1 = make_chunk("dup-id", section_order=1)
        c2 = make_chunk("dup-id", section_order=2)  # same id
        c3 = make_chunk("unique", section_order=3)
        chunks = [c1, c2, c3]
        doc_emb = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32
        )
        q_vec = np.array([1.0, 0.0], dtype=np.float32)
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.5)
        resp = r.retrieve("q")
        returned_ids = [res.chunk_id for res in resp.results]
        assert returned_ids.count("dup-id") == 1, (
            "Duplicate chunk_id must appear at most once"
        )


class TestDeterministicOrdering:
    """Same input always produces the same result ordering."""

    def test_identical_scores_break_by_original_index(self):
        """When two chunks have equal scores, lower original index wins."""
        chunks = [
            make_chunk("first", section_order=1),
            make_chunk("second", section_order=2),
        ]
        # Identical document vectors -> identical cosine scores.
        doc_emb = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        q_vec = np.array([1.0, 0.0], dtype=np.float32)
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.5)
        resp = r.retrieve("q")
        assert resp.results[0].chunk_id == "first"

    def test_repeated_calls_return_same_order(self):
        chunks = [make_chunk(f"c{i}", section_order=i) for i in range(4)]
        doc_emb = np.random.default_rng(42).random((4, 8)).astype(np.float32)
        q_vec = np.ones(8, dtype=np.float32)
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.0)
        resp1 = r.retrieve("q")
        resp2 = r.retrieve("q")
        assert [x.chunk_id for x in resp1.results] == [
            x.chunk_id for x in resp2.results
        ]


class TestResponseContract:
    """Response fields match the accepted retrieval output contract."""

    def test_response_has_required_fields(self):
        chunks = [make_chunk("c1", section_order=1)]
        doc_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        q_vec = np.array([1.0, 0.0], dtype=np.float32)
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.5)
        resp = r.retrieve("test question")
        assert resp.query == "test question"
        assert resp.model_id == "test-model"
        assert isinstance(resp.top_score, float)
        assert isinstance(resp.evidence_found, bool)
        assert isinstance(resp.results, list)

    def test_result_fields_present(self):
        chunks = [make_chunk("c1", document_id="D1", document_title="T1",
                             version="2025-01", section_title="S1",
                             source_file="f.md", text="body")]
        doc_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        q_vec = np.array([1.0, 0.0], dtype=np.float32)
        r = make_retriever(chunks, doc_emb, q_vec, threshold=0.5)
        resp = r.retrieve("q")
        res = resp.results[0]
        assert res.chunk_id == "c1"
        assert res.document_id == "D1"
        assert res.document_title == "T1"
        assert res.version == "2025-01"
        assert res.section_title == "S1"
        assert res.source_file == "f.md"
        assert res.text == "body"
        assert res.rank == 1
        assert isinstance(res.score, float)


class TestLockedProductionConfiguration:
    """Calibration-selected defaults remain explicit and regression-tested."""

    def test_approved_embedding_and_retrieval_defaults(self):
        assert config.embedding_model == "BAAI/bge-small-en-v1.5"
        assert config.top_k == 3
        assert config.min_relevance_score == pytest.approx(0.625)

    def test_blank_hf_home_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("HF_HOME", "   ")
        assert normalize_optional_path_environment("HF_HOME") == ""
        assert "HF_HOME" not in os.environ
