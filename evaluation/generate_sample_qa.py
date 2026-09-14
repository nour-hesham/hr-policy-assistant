"""Build sample_qa.md from preserved final-system outputs without editing answers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.api import build_runtime


PARTIAL_QUESTION = (
    "What is the annual training budget, and what paid parental leave does the "
    "company offer?"
)


def render_entry(number: int, record: dict[str, Any]) -> str:
    lines = [
        f"## {number}. {record['label']}",
        "",
        "**Question**",
        "",
        record["question"],
        "",
        f"**Status:** `{record['status']}`",
        "",
        "**Produced answer**",
        "",
        record["answer"],
        "",
        "**Sources**",
        "",
    ]
    if not record["sources"]:
        lines.append("None — the system abstained without exposing evidence.")
    else:
        for source in record["sources"]:
            lines.append(
                f"- [{source['source_id']}] {source['document_title']} — "
                f"{source['section_title']} ({source['document_id']}, "
                f"version {source['version']}, `{source['source_file']}`)"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    repository_root = Path(__file__).resolve().parent.parent
    output_path = repository_root / "sample_qa.md"
    if output_path.exists():
        raise FileExistsError("sample_qa.md already exists; preserve recorded outputs.")

    with (
        repository_root / "evaluation" / "results" / "answer_results.json"
    ).open(encoding="utf-8") as results_file:
        answer_results = json.load(results_file)
    by_id = {record["id"]: record for record in answer_results["records"]}

    selections = [
        ("cal-03", "Normal answer: remote work"),
        ("cal-05", "Normal answer: home internet expenses"),
        ("cal-08", "Normal answer: formal grievance"),
        ("cal-06", "Conditional numeric answer: flight class"),
        ("cal-09", "Unsupported question"),
    ]
    records: list[dict[str, Any]] = []
    for case_id, label in selections:
        source_record = by_id[case_id]
        if source_record["actual_status"] == "error":
            raise RuntimeError(f"Cannot sample failed result {case_id}.")
        records.append(
            {
                "label": label,
                "question": source_record["question"],
                "status": source_record["actual_status"],
                "answer": source_record["answer"],
                "sources": source_record["sources"],
            }
        )

    print("Initializing final-system runtime for partial sample...", flush=True)
    runtime = build_runtime()
    print("Runtime initialized; requesting partial sample...", flush=True)
    partial_response = runtime.service.ask(PARTIAL_QUESTION)
    print(f"Partial sample returned status={partial_response.status}", flush=True)
    if partial_response.status != "partial":
        raise RuntimeError(
            "The dedicated partial sample did not return partial; no sample file written."
        )
    records.insert(
        4,
        {
            "label": "Partial answer: training budget and parental leave",
            "question": PARTIAL_QUESTION,
            "status": partial_response.status,
            "answer": partial_response.answer,
            "sources": [source.model_dump() for source in partial_response.sources],
        },
    )

    document = [
        "# Sample HR Policy Q&A",
        "",
        (
            "These six examples preserve unchanged responses from the final shared "
            "retrieval and generation pipeline. Five come from the recorded answer "
            "evaluation; the partial example is one additional final-system run."
        ),
        "",
    ]
    for index, record in enumerate(records, start=1):
        document.append(render_entry(index, record))

    output_path.write_text("\n".join(document), encoding="utf-8", newline="\n")
    print(f"sample_qa.md saved with {len(records)} unchanged outputs")
    print(f"partial_sample_status={partial_response.status}")


if __name__ == "__main__":
    main()
