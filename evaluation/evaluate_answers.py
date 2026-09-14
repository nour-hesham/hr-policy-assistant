"""Run the final shared retrieval/generation pipeline for human answer review."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

from backend.api import build_runtime
from backend.generation import BaseGenerator, GenerationResult
from backend.retrieval import RetrievalResult


class RecordingGenerator(BaseGenerator):
    """Count calls while delegating every request to the production provider."""

    def __init__(self, delegate: BaseGenerator) -> None:
        self.delegate = delegate
        self.call_count = 0

    def generate(
        self,
        question: str,
        sources: Sequence[RetrievalResult],
    ) -> GenerationResult:
        self.call_count += 1
        return self.delegate.generate(question, sources)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=4.5,
        help="Delay between live provider calls to respect free-tier rate limits.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/answer_results.json",
        help="Repository-relative or absolute path for the machine-readable run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds cannot be negative")

    repository_root = Path(__file__).resolve().parent.parent
    requested_output = Path(args.output)
    results_path = (
        requested_output
        if requested_output.is_absolute()
        else repository_root / requested_output
    )
    if results_path.exists():
        raise FileExistsError(
            "answer_results.json already exists; preserve the recorded live run."
        )

    with (repository_root / "evaluation" / "cases.json").open(
        encoding="utf-8"
    ) as cases_file:
        case_data = json.load(cases_file)
    cases = case_data["calibration"] + case_data["holdout"]

    print("Initializing final-system runtime...", flush=True)
    runtime = build_runtime()
    print("Runtime initialized.", flush=True)
    recorder = RecordingGenerator(runtime.service.generator)
    runtime.service.generator = recorder

    records: list[dict[str, Any]] = []
    last_provider_call_finished: float | None = None

    for case in cases:
        print(f"Starting {case['id']}...", flush=True)
        if last_provider_call_finished is not None:
            elapsed = time.perf_counter() - last_provider_call_finished
            if elapsed < args.delay_seconds:
                time.sleep(args.delay_seconds - elapsed)

        calls_before = recorder.call_count
        started = time.perf_counter()
        try:
            response = runtime.service.ask(case["question"])
            provider_called = recorder.call_count > calls_before
            if provider_called:
                last_provider_call_finished = time.perf_counter()
            records.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "question": case["question"],
                    "expected_status": case["expected_status"],
                    "actual_status": response.status,
                    "answer": response.answer,
                    "sources": [source.model_dump() for source in response.sources],
                    "required_facts": case["required_facts"],
                    "provider_called": provider_called,
                    "latency_ms": round(
                        (time.perf_counter() - started) * 1000, 3
                    ),
                    "error_type": None,
                }
            )
            print(
                f"{case['id']}: expected={case['expected_status']} "
                f"actual={response.status} provider_called={str(provider_called).lower()}",
                flush=True,
            )
        except Exception as exc:
            provider_called = recorder.call_count > calls_before
            if provider_called:
                last_provider_call_finished = time.perf_counter()
            records.append(
                {
                    "id": case["id"],
                    "category": case["category"],
                    "question": case["question"],
                    "expected_status": case["expected_status"],
                    "actual_status": "error",
                    "answer": None,
                    "sources": [],
                    "required_facts": case["required_facts"],
                    "provider_called": provider_called,
                    "latency_ms": round(
                        (time.perf_counter() - started) * 1000, 3
                    ),
                    "error_type": type(exc).__name__,
                }
            )
            print(
                f"{case['id']}: expected={case['expected_status']} "
                f"actual=error error_type={type(exc).__name__}",
                flush=True,
            )

    output = {
        "evaluation_method": (
            "One final-system run per fixed case. Answers are preserved unchanged "
            "for direct human review; no LLM judge is used."
        ),
        "configuration": {
            "embedding_model": runtime.embedding_model_id,
            "top_k": runtime.service.top_k,
            "min_relevance_score": runtime.service.retriever.threshold,
            "generation_model": runtime.generation_model_id,
        },
        "case_count": len(records),
        "provider_call_count": recorder.call_count,
        "records": records,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(output, output_file, indent=2, ensure_ascii=False)
        output_file.write("\n")
    print(f"Answer results saved to {results_path}")


if __name__ == "__main__":
    main()
