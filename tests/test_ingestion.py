"""
Deterministic ingestion tests.

No network access, no model downloads, no API keys required.
All fixtures use tmp_path or in-memory content.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest

from backend.ingestion import (
    ChunkRecord,
    ingest_directory,
    normalize_text,
    normalize_date,
    parse_markdown_file,
    slugify,
)
from backend.logging_config import configure_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_policy(tmp_path: Path, filename: str, content: str) -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


VALID_MD = """\
# Annual Leave Policy

**Document ID:** HR-POL-001
**Last updated:** March 2025

## Purpose

This policy sets out annual leave rules.

## Entitlement

Standard entitlement is 21 working days.
"""


# ---------------------------------------------------------------------------
# normalize_date
# ---------------------------------------------------------------------------

class TestNormalizeDate:
    def test_march_2025(self):
        assert normalize_date("March 2025") == "2025-03"

    def test_january(self):
        assert normalize_date("January 2024") == "2024-01"

    def test_december(self):
        assert normalize_date("December 2023") == "2023-12"

    def test_case_insensitive(self):
        assert normalize_date("APRIL 2025") == "2025-04"

    def test_unknown_format_passthrough(self):
        result = normalize_date("Q1 2025")
        assert "2025" in result  # at minimum the year survives


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------

class TestSlugify:
    def test_simple(self):
        assert slugify("Purpose") == "purpose"

    def test_spaces_to_hyphens(self):
        assert slugify("Annual Leave") == "annual-leave"

    def test_special_chars_stripped(self):
        assert slugify("Step 2: Formal Complaint") == "step-2-formal-complaint"

    def test_numbers_preserved(self):
        assert slugify("Section 360-degree feedback") == "section-360-degree-feedback"


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------

class TestNormalizeText:
    def test_crlf_normalized(self):
        result = normalize_text("line1\r\nline2")
        assert "\r" not in result
        assert "line1" in result

    def test_repeated_spaces_collapsed(self):
        result = normalize_text("a  b   c")
        assert "  " not in result

    def test_content_preserved(self):
        text = "USD 25 or more; not reimbursed."
        assert normalize_text(text) == text.strip()


# ---------------------------------------------------------------------------
# parse_markdown_file — valid cases
# ---------------------------------------------------------------------------

class TestParseValidMarkdown:
    def test_two_sections_parsed(self, tmp_path):
        p = write_policy(tmp_path, "policy.md", VALID_MD)
        chunks = parse_markdown_file(p)
        assert len(chunks) == 2

    def test_document_title(self, tmp_path):
        p = write_policy(tmp_path, "policy.md", VALID_MD)
        chunks = parse_markdown_file(p)
        assert chunks[0].document_title == "Annual Leave Policy"

    def test_document_id(self, tmp_path):
        p = write_policy(tmp_path, "policy.md", VALID_MD)
        chunks = parse_markdown_file(p)
        assert chunks[0].document_id == "HR-POL-001"

    def test_version_normalized(self, tmp_path):
        p = write_policy(tmp_path, "policy.md", VALID_MD)
        chunks = parse_markdown_file(p)
        assert chunks[0].version == "2025-03"

    def test_section_titles(self, tmp_path):
        p = write_policy(tmp_path, "policy.md", VALID_MD)
        chunks = parse_markdown_file(p)
        assert chunks[0].section_title == "Purpose"
        assert chunks[1].section_title == "Entitlement"

    def test_section_order_one_based(self, tmp_path):
        p = write_policy(tmp_path, "policy.md", VALID_MD)
        chunks = parse_markdown_file(p)
        assert chunks[0].section_order == 1
        assert chunks[1].section_order == 2

    def test_source_file_is_basename(self, tmp_path):
        p = write_policy(tmp_path, "01-annual-leave.md", VALID_MD)
        chunks = parse_markdown_file(p)
        assert chunks[0].source_file == "01-annual-leave.md"

    def test_chunk_id_format(self, tmp_path):
        p = write_policy(tmp_path, "policy.md", VALID_MD)
        chunks = parse_markdown_file(p)
        assert chunks[0].chunk_id == "HR-POL-001@2025-03::purpose"
        assert chunks[1].chunk_id == "HR-POL-001@2025-03::entitlement"

    def test_content_hash_sha256_prefix(self, tmp_path):
        p = write_policy(tmp_path, "policy.md", VALID_MD)
        chunks = parse_markdown_file(p)
        assert chunks[0].content_hash.startswith("sha256:")

    def test_content_hash_is_deterministic(self, tmp_path):
        p = write_policy(tmp_path, "policy.md", VALID_MD)
        chunks1 = parse_markdown_file(p)
        chunks2 = parse_markdown_file(p)
        assert chunks1[0].content_hash == chunks2[0].content_hash

    def test_content_hash_matches_manual(self, tmp_path):
        """Hash of normalized text must equal sha256 of that text."""
        p = write_policy(tmp_path, "policy.md", VALID_MD)
        chunks = parse_markdown_file(p)
        text = chunks[0].text
        expected = "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert chunks[0].content_hash == expected

    def test_text_contains_original_content(self, tmp_path):
        p = write_policy(tmp_path, "policy.md", VALID_MD)
        chunks = parse_markdown_file(p)
        assert "21 working days" in chunks[1].text

    def test_chunk_record_is_pydantic_model(self, tmp_path):
        p = write_policy(tmp_path, "policy.md", VALID_MD)
        chunks = parse_markdown_file(p)
        assert isinstance(chunks[0], ChunkRecord)

    def test_different_content_produces_different_hash(self, tmp_path):
        p1 = write_policy(tmp_path, "a.md", VALID_MD)
        modified = VALID_MD.replace("21 working days", "25 working days")
        p2 = write_policy(tmp_path, "b.md", modified)
        c1 = parse_markdown_file(p1)
        c2 = parse_markdown_file(p2)
        assert c1[1].content_hash != c2[1].content_hash


# ---------------------------------------------------------------------------
# parse_markdown_file — validation failures
# ---------------------------------------------------------------------------

class TestValidationFailures:
    def test_missing_title_raises(self, tmp_path):
        content = "**Document ID:** HR-POL-001\n**Last updated:** March 2025\n## Section\nContent\n"
        p = write_policy(tmp_path, "policy.md", content)
        with pytest.raises(ValueError, match="No document title found"):
            parse_markdown_file(p)

    def test_missing_document_id_raises(self, tmp_path):
        content = "# Title\n**Last updated:** March 2025\n## Section\nContent\n"
        p = write_policy(tmp_path, "policy.md", content)
        with pytest.raises(ValueError, match="No document ID found"):
            parse_markdown_file(p)

    def test_missing_last_updated_raises(self, tmp_path):
        content = "# Title\n**Document ID:** HR-POL-001\n## Section\nContent\n"
        p = write_policy(tmp_path, "policy.md", content)
        with pytest.raises(ValueError, match="No last updated date found"):
            parse_markdown_file(p)

    def test_no_sections_raises(self, tmp_path):
        content = "# Title\n**Document ID:** HR-POL-001\n**Last updated:** March 2025\n\nJust prose.\n"
        p = write_policy(tmp_path, "policy.md", content)
        with pytest.raises(ValueError, match="No sections found"):
            parse_markdown_file(p)

    def test_empty_section_content_raises(self, tmp_path):
        content = "# Title\n**Document ID:** HR-POL-001\n**Last updated:** March 2025\n## Section\n   \n"
        p = write_policy(tmp_path, "policy.md", content)
        with pytest.raises(ValueError, match="Empty section content"):
            parse_markdown_file(p)


# ---------------------------------------------------------------------------
# ingest_directory
# ---------------------------------------------------------------------------

class TestIngestDirectory:
    def test_sorts_files_alphabetically(self, tmp_path):
        """Chunks from earlier-named files appear before later-named files."""
        # Write in reverse order to confirm sort, not fs order.
        write_policy(tmp_path, "b.md", VALID_MD.replace("HR-POL-001", "DOC-B"))
        write_policy(tmp_path, "a.md", VALID_MD.replace("HR-POL-001", "DOC-A"))
        chunks = ingest_directory(tmp_path)
        doc_ids = [c.document_id for c in chunks]
        first_a = doc_ids.index("DOC-A")
        first_b = doc_ids.index("DOC-B")
        assert first_a < first_b, "a.md chunks must precede b.md chunks"

    def test_all_ten_real_policies(self):
        """Smoke test: all 10 production policies parse without error."""
        policies_dir = Path(__file__).parent.parent / "policies"
        chunks = ingest_directory(policies_dir)
        doc_ids = {c.document_id for c in chunks}
        for n in range(1, 11):
            expected_id = f"HR-POL-{n:03d}"
            assert expected_id in doc_ids, f"{expected_id} not found in ingested chunks"

    def test_chunk_ids_are_unique(self):
        """No two chunks from the real policy set share a chunk_id."""
        policies_dir = Path(__file__).parent.parent / "policies"
        chunks = ingest_directory(policies_dir)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids found"

    def test_ignores_non_md_files(self, tmp_path):
        write_policy(tmp_path, "policy.md", VALID_MD)
        (tmp_path / "notes.txt").write_text("ignored")
        chunks = ingest_directory(tmp_path)
        assert all(c.source_file.endswith(".md") for c in chunks)


# ---------------------------------------------------------------------------
# Logging: UTC timestamps
# ---------------------------------------------------------------------------

class TestLoggingUTC:
    def test_formatter_uses_gmtime(self):
        """
        The logging Formatter must use time.gmtime so timestamps are
        genuinely UTC regardless of host timezone.
        """
        import logging
        import time as time_module

        configure_logging("DEBUG")
        root = logging.getLogger()
        assert root.handlers, "Root logger must have at least one handler"
        handler = root.handlers[0]
        assert hasattr(handler, "formatter"), "Handler must have a formatter"
        formatter = handler.formatter
        assert formatter is not None
        assert formatter.converter is time_module.gmtime, (
            "Formatter.converter must be time.gmtime for genuine UTC timestamps"
        )

    def test_uvicorn_access_log_is_disabled(self):
        import logging

        configure_logging("INFO")
        assert logging.getLogger("uvicorn.access").disabled is True
