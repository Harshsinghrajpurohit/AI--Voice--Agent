"""Knowledge Base parsing, chunking, and schema tests."""

from __future__ import annotations

import pytest

from bank_voice_assistant.config import PROJECT_ROOT
from bank_voice_assistant.errors import KnowledgeBaseError
from bank_voice_assistant.kb import KnowledgeBase, chunk_document, parse_front_matter


def test_parse_front_matter_valid() -> None:
    sample = """---
title: Test Title
category: Testing
tags: [alpha, beta, gamma]
last_updated: 2026-09-01
---

## First Heading
Some body content here.
"""
    meta, body = parse_front_matter(sample, "sample.md")
    assert meta.title == "Test Title"
    assert meta.category == "Testing"
    assert meta.tags == ("alpha", "beta", "gamma")
    assert meta.last_updated == "2026-09-01"
    assert "## First Heading\nSome body content here." == body


def test_parse_front_matter_missing_fences() -> None:
    sample = "No front matter here\nJust text."
    with pytest.raises(KnowledgeBaseError, match="missing leading '---' front-matter block"):
        parse_front_matter(sample, "invalid.md")


def test_parse_front_matter_missing_required_field() -> None:
    sample = """---
title: Incomplete Title
category: Testing
---
Content
"""
    with pytest.raises(KnowledgeBaseError, match="missing required field"):
        parse_front_matter(sample, "incomplete.md")


def test_chunk_document(tmp_path: pytest.TempPathFactory) -> None:
    test_file = tmp_path / "accounts.md"  # type: ignore[operator]
    test_file.write_text(
        """---
title: Bank Accounts
category: Accounts
tags: [savings, charges]
last_updated: 2026-09-01
---

## Savings Account
The savings rate is 4.00% per annum.

## Current Account
Current account earns 0.00% interest.
""",
        encoding="utf-8",
    )

    chunks = chunk_document(test_file)
    assert len(chunks) == 2
    assert chunks[0].heading == "Savings Account"
    assert "4.00%" in chunks[0].content
    assert chunks[0].doc_title == "Bank Accounts"
    assert chunks[1].heading == "Current Account"
    assert "0.00%" in chunks[1].content


def test_load_all_kb_documents() -> None:
    """Verify that all markdown documents in data/kb parse and chunk cleanly."""
    kb_dir = PROJECT_ROOT / "data" / "kb"
    kb = KnowledgeBase(kb_dir)
    chunks = kb.load_all_chunks()

    # We created 6 files, each having 3 to 4 sections
    assert len(chunks) >= 18
    # Assert specific domain contents are present
    all_text = " ".join(c.searchable_text for c in chunks)
    assert "Northwind Classic Savings Account" in all_text
    assert "Fixed Deposit (FD) rates" in all_text
    assert "Cobalt Credit Card" in all_text
    assert "1800-419-0022" in all_text
    assert "₹" in all_text  # INR currency symbol verified


def test_kb_directory_missing(tmp_path: pytest.TempPathFactory) -> None:
    missing_dir = tmp_path / "non_existent_kb"  # type: ignore[operator]
    kb = KnowledgeBase(missing_dir)
    with pytest.raises(KnowledgeBaseError, match="does not exist"):
        kb.load_all_chunks()
