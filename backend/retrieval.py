"""
In-memory semantic retrieval over policy chunks.

Production path:
  EmbeddingRetriever — BAAI/bge-small-en-v1.5 (or any ST model)
  NumPy exact cosine similarity after L2 normalization.

Evaluation-only (not imported by this module at the top level):
  MiniLMRetriever and TFIDFRetriever live in evaluation/evaluate_retrieval.py.

Contract:
  retrieve(question, filters=None, top_k=5) -> RetrievalResponse
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Any

import numpy as np
from pydantic import BaseModel

from backend.ingestion import ChunkRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class RetrievalResult(BaseModel):
    rank: int
    score: float
    chunk_id: str
    document_id: str
    document_title: str
    version: str
    section_title: str
    source_file: str
    text: str


class RetrievalResponse(BaseModel):
    query: str
    model_id: str
    top_k: int
    top_score: float
    evidence_found: bool
    results: List[RetrievalResult]


# ---------------------------------------------------------------------------
# Retriever contract
# ---------------------------------------------------------------------------

class BaseRetriever:
    """Minimal model-independent retriever interface."""

    def retrieve(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> RetrievalResponse:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Production retriever
# ---------------------------------------------------------------------------

class EmbeddingRetriever(BaseRetriever):
    """
    Dense semantic retriever backed by a Sentence Transformers model and an
    exact NumPy cosine search over L2-normalised embeddings.

    Chunk ordering is deterministic: chunks are indexed in the order returned
    by ingest_directory(), which sorts files alphabetically inside that
    function.  Within a file, sections follow document order.
    """

    def __init__(
        self,
        chunks: List[ChunkRecord],
        model_id: str = "BAAI/bge-small-en-v1.5",
        threshold: float = 0.5,
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
    ) -> None:
        # Inline import keeps sentence-transformers out of test collection
        # when tests inject a fake retriever.
        from sentence_transformers import SentenceTransformer  # type: ignore
        import torch

        # One Uvicorn worker and a tiny corpus do not benefit from competing
        # OpenMP worker pools. A single Torch thread also avoids intermittent
        # native access violations observed on Windows CPU during startup.
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch permits setting the inter-op pool only before parallel
            # work starts; repeated in-process construction can reach here.
            pass
        torch.backends.mkldnn.enabled = False

        self.chunks = list(chunks)  # deterministic copy
        self.model_id = model_id
        self.threshold = threshold
        self.query_prefix = query_prefix

        self._model = SentenceTransformer(model_id, device="cpu")

        if self.chunks:
            texts = [
                f"{c.document_title}\n{c.section_title}\n{c.text}"
                for c in self.chunks
            ]
            raw = self._model.encode(
                texts,
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            self._embeddings = self._l2_normalize(raw)
        else:
            self._embeddings = np.empty((0, 0), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        question: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> RetrievalResponse:
        """Return at most top_k unique chunks whose score >= self.threshold."""
        if len(self.chunks) == 0:
            return RetrievalResponse(
                query=question,
                model_id=self.model_id,
                top_k=top_k,
                top_score=0.0,
                evidence_found=False,
                results=[],
            )

        q_raw = self._model.encode(
            self.query_prefix + question,
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        q_vec = self._l2_normalize(q_raw.reshape(1, -1))[0]

        scores: np.ndarray = self._embeddings @ q_vec  # cosine after L2 norm

        # Rank all chunks by score descending (deterministic tie-break by
        # original row index — lower index wins, preserving file/section order).
        # np.lexsort sorts by the LAST key first, so -scores is the primary
        # sort key and np.arange (ascending) is the secondary tiebreaker.
        sorted_indices = np.lexsort(
            (np.arange(len(scores), dtype=np.int64), -scores)
        )

        top_score = float(scores[sorted_indices[0]])

        # Early exit: if the best score fails the threshold return no evidence.
        if top_score < self.threshold:
            logger.debug(
                "no_evidence_found top_score=%.4f threshold=%.4f question_len=%d",
                top_score,
                self.threshold,
                len(question),
            )
            return RetrievalResponse(
                query=question,
                model_id=self.model_id,
                top_k=top_k,
                top_score=top_score,
                evidence_found=False,
                results=[],
            )

        # Collect at most top_k unique results that pass the threshold.
        seen_ids: set[str] = set()
        results: List[RetrievalResult] = []
        rank = 1

        for idx in sorted_indices:
            if rank > top_k:
                break
            score = float(scores[idx])
            if score < self.threshold:
                break  # remaining scores are lower; all fail
            chunk = self.chunks[idx]
            if chunk.chunk_id in seen_ids:
                continue  # deduplicate exact chunk_id
            seen_ids.add(chunk.chunk_id)
            results.append(
                RetrievalResult(
                    rank=rank,
                    score=score,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    version=chunk.version,
                    section_title=chunk.section_title,
                    source_file=chunk.source_file,
                    text=chunk.text,
                )
            )
            rank += 1

        logger.debug(
            "retrieval_complete top_score=%.4f results=%d threshold=%.4f",
            top_score,
            len(results),
            self.threshold,
        )
        return RetrievalResponse(
            query=question,
            model_id=self.model_id,
            top_k=top_k,
            top_score=top_score,
            evidence_found=True,
            results=results,
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalisation; zero vectors stay zero."""
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms != 0)
