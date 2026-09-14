"""
Retrieval evaluation harness.

Calls production ingestion and retrieval components — does not reimplement
their logic.  TF-IDF and MiniLM are evaluation-only and live here, not in
backend/retrieval.py.

Design note on metric separation
---------------------------------
Hit@k and MRR are computed from the UNFILTERED ranked candidates (all
chunks returned by the model regardless of the threshold gate).  This
measures pure retrieval ordering quality independently of whatever
threshold happens to be configured.

Threshold behavior is evaluated separately:
  - evidence_found flag on unsupported questions  -> rejection rate
  - evidence_found flag on supported questions   -> supported evidence-found rate
  - per-case retained expected chunks after filtering -> coverage after threshold

This separation lets us compare ranking quality across models using a
consistent scoring rule, then sweep thresholds to pick the abstention
point without conflating the two concerns.

Usage (from repository root with .venv active):
    python -m evaluation.evaluate_retrieval

Outputs:
    evaluation/results/retrieval_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Make sure backend is importable when run from repository root.
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.ingestion import ChunkRecord, ingest_directory
from backend.retrieval import BaseRetriever, EmbeddingRetriever, RetrievalResponse


# ---------------------------------------------------------------------------
# Evaluation-only retrievers (sklearn kept out of production path)
# ---------------------------------------------------------------------------

class MiniLMRetriever(EmbeddingRetriever):
    """MiniLM challenger — evaluation only."""

    def __init__(self, chunks: list[ChunkRecord], threshold: float = 0.3) -> None:
        super().__init__(
            chunks,
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            threshold=threshold,
            query_prefix="",
        )


class TFIDFRetriever(BaseRetriever):
    """TF-IDF lexical baseline — evaluation only."""

    def __init__(self, chunks: list[ChunkRecord], threshold: float = 0.1) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
        from sklearn.metrics.pairwise import cosine_similarity as sk_cos  # noqa: PLC0415

        self._sk_cos = sk_cos
        self.chunks = list(chunks)
        self.model_id = "TF-IDF"
        self.threshold = threshold

        texts = [
            f"{c.document_title}\n{c.section_title}\n{c.text}" for c in self.chunks
        ]
        self._vectorizer = TfidfVectorizer()
        self._vectors = self._vectorizer.fit_transform(texts) if texts else None

    def retrieve(
        self,
        question: str,
        filters=None,
        top_k: int = 5,
    ) -> RetrievalResponse:
        from backend.retrieval import RetrievalResult  # local import for eval

        if not self.chunks or self._vectors is None:
            return RetrievalResponse(
                query=question,
                model_id=self.model_id,
                top_k=top_k,
                top_score=0.0,
                evidence_found=False,
                results=[],
            )

        q_vec = self._vectorizer.transform([question])
        scores_arr = self._sk_cos(q_vec, self._vectors)[0]

        # Deterministic sort: descending score, ascending original index on ties.
        sorted_indices = np.lexsort(
            (np.arange(len(scores_arr), dtype=np.int64), -scores_arr)
        )
        top_score = float(scores_arr[sorted_indices[0]])

        results = []
        seen: set[str] = set()
        rank = 1
        for idx in sorted_indices:
            if rank > top_k:
                break
            score = float(scores_arr[idx])
            if score < self.threshold:
                break
            chunk = self.chunks[idx]
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
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

        evidence_found = top_score >= self.threshold and bool(results)
        return RetrievalResponse(
            query=question,
            model_id=self.model_id,
            top_k=top_k,
            top_score=top_score,
            evidence_found=evidence_found,
            results=results,
        )


# ---------------------------------------------------------------------------
# Helpers: ranking metrics (threshold-independent)
# ---------------------------------------------------------------------------

def _retrieve_unfiltered(
    retriever: BaseRetriever,
    question: str,
    top_k: int,
    original_threshold: float,
) -> RetrievalResponse:
    """
    Retrieve with threshold=-inf so all candidates are returned, allowing
    Hit@k and MRR to be computed from the full ranked list regardless of
    the configured abstention gate.
    """
    old = retriever.threshold
    retriever.threshold = -999.0  # effectively no threshold gate
    try:
        resp = retriever.retrieve(question, top_k=top_k)
    finally:
        retriever.threshold = old
    return resp


def _hit_at_k(retrieved_ids: list[str], expected_ids: set[str], k: int) -> bool:
    return any(eid in retrieved_ids[:k] for eid in expected_ids)


def _reciprocal_rank(retrieved_ids: list[str], expected_ids: set[str]) -> float:
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in expected_ids:
            return 1.0 / i
    return 0.0


def _all_expected_covered(retrieved_ids: list[str], expected_ids: set[str]) -> bool:
    """True only when every expected chunk appears somewhere in the result list."""
    return expected_ids.issubset(set(retrieved_ids))


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_cases(
    retriever: BaseRetriever,
    cases: list[dict[str, Any]],
    top_k: int = 5,
    ranking_top_k: int | None = None,
) -> dict[str, Any]:
    """
    Run all cases and return per-case raw results plus aggregate metrics.

    Ranking metrics (Hit@k, MRR) are computed from UNFILTERED candidates
    — the full top_k list before any threshold gate — so they reflect pure
    retrieval ordering quality independently of the abstention threshold.

    Threshold metrics (evidence_found, rejection) use the retriever's
    currently configured threshold and measure abstention behavior only.
    """
    ranking_k = ranking_top_k or top_k
    if ranking_k < top_k:
        raise ValueError("ranking_top_k cannot be smaller than top_k")

    per_case: list[dict[str, Any]] = []
    latencies_ms: list[float] = []

    supported = [c for c in cases if c["expected_status"] != "not_found"]
    unsupported = [c for c in cases if c["expected_status"] == "not_found"]

    hit1 = hit3 = hit5 = mrr_sum = 0.0
    saved_threshold = retriever.threshold

    for case in supported:
        expected_ids = set(case["expected_chunk_ids"])

        # Unfiltered retrieval for ranking metrics.
        t0 = time.perf_counter()
        resp_unfiltered = _retrieve_unfiltered(
            retriever, case["question"], ranking_k, saved_threshold
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(latency_ms)

        # Filtered retrieval for threshold metrics (evidence_found, retained chunks).
        resp_filtered = retriever.retrieve(case["question"], top_k=top_k)

        all_ids = [r.chunk_id for r in resp_unfiltered.results]
        retained_ids = [r.chunk_id for r in resp_filtered.results]

        h1 = _hit_at_k(all_ids, expected_ids, 1)
        h3 = _hit_at_k(all_ids, expected_ids, 3)
        h5 = _hit_at_k(all_ids, expected_ids, 5)
        rr = _reciprocal_rank(all_ids, expected_ids)
        all_covered_unfiltered = _all_expected_covered(all_ids, expected_ids)

        # Which expected chunks survive after threshold filtering?
        retained_expected = sorted(expected_ids & set(retained_ids))
        all_covered_after_filter = expected_ids.issubset(set(retained_ids))

        hit1 += h1
        hit3 += h3
        hit5 += h5
        mrr_sum += rr

        per_case.append(
            {
                "id": case["id"],
                "question": case["question"],
                "category": case["category"],
                "expected_status": case["expected_status"],
                "expected_chunk_ids": list(expected_ids),
                # --- ranking (unfiltered) ---
                "unfiltered_retrieved_ids": all_ids,
                "top_score_unfiltered": round(resp_unfiltered.top_score, 6),
                "hit_at_1": h1,
                "hit_at_3": h3,
                "hit_at_5": h5,
                "reciprocal_rank": round(rr, 4),
                "all_expected_in_top_k": all_covered_unfiltered,
                # --- threshold / filtering ---
                "threshold_used": saved_threshold,
                "evidence_found_after_filter": resp_filtered.evidence_found,
                "retained_chunk_ids": retained_ids,
                "retained_expected_chunk_ids": retained_expected,
                "all_expected_retained": all_covered_after_filter,
                "latency_ms": round(latency_ms, 3),
                # --- per-result detail (unfiltered) ---
                "per_result": [
                    {
                        "rank": r.rank,
                        "chunk_id": r.chunk_id,
                        "score": round(r.score, 6),
                        "is_expected": r.chunk_id in expected_ids,
                        "passes_threshold": r.score >= saved_threshold,
                    }
                    for r in resp_unfiltered.results
                ],
            }
        )

    correctly_rejected = 0
    for case in unsupported:
        t0 = time.perf_counter()
        resp = retriever.retrieve(case["question"], top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(latency_ms)

        rejected = not resp.evidence_found
        if rejected:
            correctly_rejected += 1

        per_case.append(
            {
                "id": case["id"],
                "question": case["question"],
                "category": case["category"],
                "expected_status": case["expected_status"],
                "expected_chunk_ids": [],
                "top_score": round(resp.top_score, 6),
                "evidence_found": resp.evidence_found,
                "correctly_rejected": rejected,
                "threshold_used": saved_threshold,
                "retrieved_chunk_ids": [r.chunk_id for r in resp.results],
                "latency_ms": round(latency_ms, 3),
                "per_result": [
                    {
                        "rank": r.rank,
                        "chunk_id": r.chunk_id,
                        "score": round(r.score, 6),
                        "passes_threshold": r.score >= saved_threshold,
                    }
                    for r in resp.results
                ],
            }
        )

    n_sup = len(supported) or 1
    n_uns = len(unsupported) or 1

    aggregate = {
        "hit_at_1": round(hit1 / n_sup, 4),
        "hit_at_3": round(hit3 / n_sup, 4),
        "hit_at_5": round(hit5 / n_sup, 4),
        "mrr": round(mrr_sum / n_sup, 4),
        "rejection_rate": round(correctly_rejected / n_uns, 4),
        "median_latency_ms": round(float(np.median(latencies_ms)), 3),
        "n_supported": len(supported),
        "n_unsupported": len(unsupported),
        "note": (
            "Hit@k and MRR computed from unfiltered top-k candidates. "
            "Rejection rate uses configured threshold."
        ),
    }

    per_case.sort(key=lambda x: x["id"])
    return {"aggregate": aggregate, "per_case": per_case}


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

def threshold_sweep(
    retriever: EmbeddingRetriever,
    cases: list[dict[str, Any]],
    thresholds: list[float],
    top_k: int = 5,
) -> dict[str, Any]:
    """
    For each candidate threshold report:
      - unsupported rejection rate
      - supported evidence-found rate (answerable questions that pass the gate)
      - all-required-evidence coverage: fraction of supported cases where
        ALL expected chunks survive the threshold
      - per-case retained expected chunks at this threshold
    """
    supported = [c for c in cases if c["expected_status"] != "not_found"]
    unsupported = [c for c in cases if c["expected_status"] == "not_found"]

    sweep_results: list[dict[str, Any]] = []

    for t in thresholds:
        retriever.threshold = t

        rej = 0
        for c in unsupported:
            r = retriever.retrieve(c["question"], top_k=top_k)
            if not r.evidence_found:
                rej += 1

        evidence_found_count = 0
        all_covered_count = 0
        per_case_retained: list[dict[str, Any]] = []

        for c in supported:
            expected_ids = set(c["expected_chunk_ids"])
            r = retriever.retrieve(c["question"], top_k=top_k)
            retained_ids = [res.chunk_id for res in r.results]
            retained_expected = sorted(expected_ids & set(retained_ids))
            all_covered = expected_ids.issubset(set(retained_ids))

            if r.evidence_found:
                evidence_found_count += 1
            if all_covered:
                all_covered_count += 1

            per_case_retained.append(
                {
                    "id": c["id"],
                    "evidence_found": r.evidence_found,
                    "retained_expected": retained_expected,
                    "all_expected_retained": all_covered,
                }
            )

        sweep_results.append(
            {
                "threshold": round(t, 4),
                "rejection_rate": round(rej / (len(unsupported) or 1), 4),
                "supported_evidence_found_rate": round(
                    evidence_found_count / (len(supported) or 1), 4
                ),
                "all_required_evidence_coverage": round(
                    all_covered_count / (len(supported) or 1), 4
                ),
                "per_case_retained": per_case_retained,
            }
        )

    return {"sweep": sweep_results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).parent.parent

    print("Loading chunks from policies/...")
    chunks = ingest_directory(repo_root / "policies")
    print(f"  {len(chunks)} chunks loaded")

    cases_path = repo_root / "evaluation" / "cases.json"
    with cases_path.open(encoding="utf-8") as f:
        data = json.load(f)

    cal_cases = data["calibration"]
    # holdout is NOT loaded or used anywhere in this script.
    print(f"  {len(cal_cases)} calibration cases  (holdout not inspected)")

    results_dir = repo_root / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    TOP_K = 5

    # ------------------------------------------------------------------ #
    # BGE                                                                  #
    # ------------------------------------------------------------------ #
    print("\nInitialising BGE (BAAI/bge-small-en-v1.5)...")
    bge = EmbeddingRetriever(
        chunks,
        model_id="BAAI/bge-small-en-v1.5",
        threshold=0.60,
    )

    print("Evaluating BGE (ranking metrics from unfiltered, threshold=0.60 for rejection)...")
    bge.threshold = 0.60
    bge_results = evaluate_cases(bge, cal_cases, top_k=TOP_K)

    # Threshold sweep covering requested values.
    sweep_thresholds = [
        0.600, 0.610, 0.615, 0.617, 0.620, 0.625, 0.630, 0.640, 0.650,
    ]
    print("Running BGE threshold sweep...")
    bge_sweep = threshold_sweep(bge, cal_cases, sweep_thresholds, top_k=TOP_K)
    bge.threshold = 0.60  # restore

    # ------------------------------------------------------------------ #
    # MiniLM                                                               #
    # ------------------------------------------------------------------ #
    print("\nInitialising MiniLM (sentence-transformers/all-MiniLM-L6-v2)...")
    minilm = MiniLMRetriever(chunks, threshold=0.40)

    print("Evaluating MiniLM (threshold=0.40 for rejection)...")
    minilm_results = evaluate_cases(minilm, cal_cases, top_k=TOP_K)

    # ------------------------------------------------------------------ #
    # TF-IDF                                                               #
    # ------------------------------------------------------------------ #
    print("\nInitialising TF-IDF...")
    tfidf = TFIDFRetriever(chunks, threshold=0.10)

    print("Evaluating TF-IDF (threshold=0.10 for rejection)...")
    tfidf_results = evaluate_cases(tfidf, cal_cases, top_k=TOP_K)

    # ------------------------------------------------------------------ #
    # Save                                                                 #
    # ------------------------------------------------------------------ #
    output = {
        "note": (
            "Ranking metrics (Hit@k, MRR) are computed from unfiltered top-k "
            "candidates, independently of threshold. "
            "Threshold sweep and rejection metrics are reported separately."
        ),
        "BGE": {
            "model_id": "BAAI/bge-small-en-v1.5",
            "threshold_for_rejection_eval": 0.60,
            "aggregate": bge_results["aggregate"],
            "per_case": bge_results["per_case"],
            "threshold_sweep": bge_sweep["sweep"],
        },
        "MiniLM": {
            "model_id": "sentence-transformers/all-MiniLM-L6-v2",
            "threshold_for_rejection_eval": 0.40,
            "aggregate": minilm_results["aggregate"],
            "per_case": minilm_results["per_case"],
        },
        "TF-IDF": {
            "model_id": "TF-IDF",
            "threshold_for_rejection_eval": 0.10,
            "aggregate": tfidf_results["aggregate"],
            "per_case": tfidf_results["per_case"],
        },
    }

    out_path = results_dir / "retrieval_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {out_path}")

    # ------------------------------------------------------------------ #
    # Print summary                                                        #
    # ------------------------------------------------------------------ #
    print("\n=== AGGREGATE METRICS (ranking from unfiltered top-5) ===")
    print(f"{'Model':<12} {'Hit@1':>6} {'Hit@3':>6} {'Hit@5':>6} {'MRR':>6} {'Reject':>7} {'ms':>7}")
    for name in ("BGE", "MiniLM", "TF-IDF"):
        agg = output[name]["aggregate"]
        print(
            f"{name:<12} {agg['hit_at_1']:>6.3f} {agg['hit_at_3']:>6.3f} "
            f"{agg['hit_at_5']:>6.3f} {agg['mrr']:>6.3f} "
            f"{agg['rejection_rate']:>7.3f} {agg['median_latency_ms']:>7.1f}"
        )

    print("\n=== BGE THRESHOLD SWEEP ===")
    print(
        f"{'Thresh':>8}  {'Rejection':>10}  {'Sup-EF':>8}  {'All-Cov':>9}"
    )
    for row in bge_sweep["sweep"]:
        print(
            f"{row['threshold']:>8.3f}  "
            f"{row['rejection_rate']:>10.4f}  "
            f"{row['supported_evidence_found_rate']:>8.4f}  "
            f"{row['all_required_evidence_coverage']:>9.4f}"
        )

    print("\n=== BGE PER-CASE DETAILS ===")
    for pc in bge_results["per_case"]:
        if pc["expected_status"] == "not_found":
            ok = "REJECTED" if not pc["evidence_found"] else "FAIL-NOT-REJECTED"
            print(
                f"  [{pc['id']}] {pc['question'][:55]!r:60} "
                f"top={pc['top_score']:.4f} -> {ok}"
            )
        else:
            h5 = "HIT@5" if pc["hit_at_5"] else "MISS"
            cov = "ALL-KEPT" if pc.get("all_expected_retained") else "PARTIAL/DROPPED"
            print(
                f"  [{pc['id']}] {pc['question'][:55]!r:60} "
                f"top={pc['top_score_unfiltered']:.4f} "
                f"RR={pc['reciprocal_rank']:.3f} {h5} {cov}"
            )
            for r in pc["per_result"]:
                marker = " <-- EXPECTED" if r["is_expected"] else ""
                kept = "" if r["passes_threshold"] else " [FILTERED]"
                print(
                    f"         rank={r['rank']} score={r['score']:.4f} "
                    f"{r['chunk_id']}{marker}{kept}"
                )


if __name__ == "__main__":
    main()
