"""One-time retrieval evaluation for the locked five-case holdout split."""

from __future__ import annotations

import json
from pathlib import Path

from backend.config import config
from backend.ingestion import ingest_directory
from backend.retrieval import EmbeddingRetriever
from evaluation.evaluate_retrieval import evaluate_cases


def main() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    with (repository_root / "evaluation" / "cases.json").open(
        encoding="utf-8"
    ) as cases_file:
        holdout_cases = json.load(cases_file)["holdout"]

    chunks = ingest_directory(repository_root / "policies")
    retriever = EmbeddingRetriever(
        chunks,
        model_id=config.embedding_model,
        threshold=config.min_relevance_score,
        query_prefix=config.bge_query_prefix,
    )
    results = evaluate_cases(
        retriever,
        holdout_cases,
        top_k=config.top_k,
        ranking_top_k=5,
    )

    output = {
        "evaluation_split": "holdout",
        "run_policy": (
            "Executed once after locking the production configuration. hold-01 "
            "was corrected before this first run to remove a calibration concept."
        ),
        "configuration": {
            "embedding_model": config.embedding_model,
            "top_k": config.top_k,
            "min_relevance_score": config.min_relevance_score,
        },
        "aggregate": results["aggregate"],
        "per_case": results["per_case"],
    }

    output_path = (
        repository_root / "evaluation" / "results" / "holdout_retrieval_results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")

    aggregate = output["aggregate"]
    print(f"Holdout results saved to {output_path}")
    print(
        "Hit@1={hit_at_1:.4f} Hit@3={hit_at_3:.4f} "
        "Hit@5={hit_at_5:.4f} MRR={mrr:.4f} rejection={rejection_rate:.4f} "
        "median_ms={median_latency_ms:.3f}".format(**aggregate)
    )


if __name__ == "__main__":
    main()
