"""Knowledge Base loading, front-matter parsing, and chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import KnowledgeBaseError


@dataclass(frozen=True, slots=True)
class DocMetadata:
    """Document-level metadata parsed from Markdown YAML-style front-matter."""

    title: str
    category: str
    tags: tuple[str, ...]
    last_updated: str


@dataclass(frozen=True, slots=True)
class Chunk:
    """A granular section of a knowledge base document ready for indexing."""

    chunk_id: str
    doc_title: str
    category: str
    heading: str
    content: str
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def searchable_text(self) -> str:
        """Text used for dense embedding and lexical retrieval."""
        tag_str = f" [Tags: {', '.join(self.tags)}]" if self.tags else ""
        return f"{self.doc_title} - {self.heading}{tag_str}\n{self.content}".strip()


def parse_front_matter(raw_text: str, filename: str = "<unknown>") -> tuple[DocMetadata, str]:
    """Parse key: value front-matter between leading '---' fences without external dependencies."""
    pattern = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)$", re.DOTALL)
    match = pattern.match(raw_text)
    if not match:
        raise KnowledgeBaseError(f"File {filename} missing leading '---' front-matter block")

    fm_block, body = match.group(1), match.group(2)
    meta_dict: dict[str, Any] = {}

    for line_idx, line in enumerate(fm_block.splitlines(), start=2):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise KnowledgeBaseError(f"{filename}:{line_idx}: Invalid front-matter line (no colon): {line!r}")
        key, val = line.split(":", 1)
        key = key.strip()
        val = val.strip()

        # Handle list syntax: [a, b, c]
        if val.startswith("[") and val.endswith("]"):
            items = [item.strip().strip("'\"") for item in val[1:-1].split(",") if item.strip()]
            meta_dict[key] = tuple(items)
        else:
            meta_dict[key] = val.strip("'\"")

    required = ("title", "category", "tags", "last_updated")
    for req in required:
        if req not in meta_dict:
            raise KnowledgeBaseError(f"{filename}: Front-matter missing required field: {req!r}")

    tags = meta_dict["tags"]
    if isinstance(tags, str):
        tags = tuple(t.strip() for t in tags.split(",") if t.strip())

    metadata = DocMetadata(
        title=str(meta_dict["title"]),
        category=str(meta_dict["category"]),
        tags=tags,
        last_updated=str(meta_dict["last_updated"]),
    )
    return metadata, body.strip()


def chunk_document(doc_path: Path) -> list[Chunk]:
    """Parse a markdown file into contextual chunks based on H2/H3 headers."""
    raw_text = doc_path.read_text(encoding="utf-8")
    metadata, body = parse_front_matter(raw_text, filename=doc_path.name)

    # Split on H2 or H3 markdown headers: e.g. ## Heading or ### Subheading
    sections = re.split(r"(?m)^(?=#{2,3}\s+)", body)
    chunks: list[Chunk] = []
    slug_base = doc_path.stem

    chunk_idx = 0
    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.splitlines()
        first_line = lines[0].strip()
        if first_line.startswith("##"):
            heading = first_line.lstrip("#").strip()
            content = "\n".join(lines[1:]).strip()
        else:
            heading = "Overview"
            content = section

        if not content:
            continue

        chunk_id = f"{slug_base}:{chunk_idx:03d}"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_title=metadata.title,
                category=metadata.category,
                heading=heading,
                content=content,
                tags=metadata.tags,
            )
        )
        chunk_idx += 1

    return chunks


class KnowledgeBase:
    """Manages loading and chunking of all documents in the KB directory."""

    def __init__(self, kb_dir: Path) -> None:
        self.kb_dir = kb_dir

    def load_all_chunks(self) -> list[Chunk]:
        """Load and chunk all Markdown files in the knowledge base directory."""
        if not self.kb_dir.is_dir():
            raise KnowledgeBaseError(f"Knowledge base directory does not exist: {self.kb_dir}")

        md_files = sorted(self.kb_dir.glob("*.md"))
        if not md_files:
            raise KnowledgeBaseError(f"Knowledge base directory is empty (no .md files): {self.kb_dir}")

        all_chunks: list[Chunk] = []
        for file_path in md_files:
            chunks = chunk_document(file_path)
            all_chunks.extend(chunks)

        return all_chunks
